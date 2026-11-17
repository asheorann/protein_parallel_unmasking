"""Sanity checks on a samples.jsonl before trusting or folding it.

Local helper, not part of Anushka's code (blocks/results/ is gitignored).

Checks, in order of how badly each would corrupt the result:

1. MASK LEAKAGE. `Denoiser.decode` writes "X" for an unresolved position and
   `evaluate.Folder.fold` silently rewrites "X" to alanine, so a sampler that
   fails to reveal everything does not crash -- it quietly reports poly-A-ish
   sequences with a real pLDDT. Any "X" invalidates the row.
2. NFE ACCOUNTING. The quality-cost curve is plotted against measured NFE, so
   NFE must equal the number of denoiser calls the partition actually needed:
   `sum_b steps_b` (+1 per block that hit the argmax safety net).
   `autoregressive` has a floor of one call per residue and so must be flat in
   `steps_requested`; `single` must track it.
3. SAFETY NET USE. `sample_sequence` finishes a block with an argmax when the
   tau-leap left positions masked. Argmax is not sampling, so if this fires
   often the samples are part greedy decode and diversity collapses.
4. DEGENERACY. Duplicate rate, mean pairwise identity, longest homopolymer run
   and residue entropy per strategy. A partition can "win" on recovery or
   pLDDT by collapsing onto one consensus sequence; that is a failure mode,
   not a result.
5. Recovery in range and, for reference, the identity a null model would get.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

import numpy as np

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"


def longest_run(s: str) -> int:
    best = run = 1
    for a, b in zip(s, s[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if s else 0


def entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    p = np.array([v / len(s) for v in c.values()])
    return float(-(p * np.log(p)).sum())


def pairwise_identity(seqs: list[str]) -> float:
    vals = []
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            a, b = seqs[i], seqs[j]
            n = min(len(a), len(b))
            if n:
                vals.append(sum(x == y for x, y in zip(a[:n], b[:n])) / n)
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--block-size", type=int, default=16,
                    help="must match the value the samples were generated with")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples) if l.strip()]
    if not rows:
        print("no rows")
        return 1
    budgets = sorted({r["steps_requested"] for r in rows})
    print(f"{len(rows)} rows, {len({r['name'] for r in rows})} proteins, "
          f"{len({r['strategy'] for r in rows})} strategies, steps {budgets}")

    fails = []

    # ---- 1. mask leakage -------------------------------------------------
    leaked = [r for r in rows if "X" in r["sequence"]]
    nX = sum(r["sequence"].count("X") for r in leaked)
    print(f"\n1. mask leakage: {len(leaked)} of {len(rows)} rows contain 'X' "
          f"({nX} positions total)")
    if leaked:
        worst = max(leaked, key=lambda r: r["sequence"].count("X"))
        print(f"   worst: {worst['name']} {worst['strategy']} "
              f"steps={worst['steps_requested']} -> {worst['sequence'].count('X')} "
              f"of {worst['length']}")
        fails.append("mask leakage: Folder.fold would silently turn these into alanine")

    # ---- 2. NFE accounting ----------------------------------------------
    print("\n2. NFE by strategy and requested steps (mean over proteins/seeds)")
    by = defaultdict(list)
    for r in rows:
        by[(r["strategy"], r["steps_requested"])].append(r["nfe"])
    lens = {r["name"]: r["length"] for r in rows}
    print(f"   {'strategy':>14} {'steps':>6} {'nfe mean':>9} {'nfe min':>8} {'nfe max':>8}")
    for (s, st) in sorted(by):
        v = np.array(by[(s, st)])
        print(f"   {s:>14} {st:6d} {v.mean():9.1f} {v.min():8d} {v.max():8d}")

    ar = {st: np.mean(by[("autoregressive", st)])
          for (s, st) in by if s == "autoregressive"}
    if ar and (max(ar.values()) - min(ar.values())) > 1e-9:
        fails.append(f"autoregressive NFE varies with steps_requested: {ar} "
                     "(it has a floor of one call per residue and must be flat)")
    if ar:
        mean_len = float(np.mean(list(lens.values())))
        got = float(np.mean(list(ar.values())))
        print(f"   autoregressive: {got:.1f} calls vs mean length {mean_len:.1f} "
              f"-> {'OK' if abs(got - mean_len) < 1.5 else 'MISMATCH'}")
        if abs(got - mean_len) >= 1.5:
            fails.append("autoregressive NFE != sequence length")

    # `single` is one block of size L, and `_allocate_steps` caps a block's
    # steps at its own size, so the ceiling is min(steps, L) PER PROTEIN, not
    # the requested budget. Averaging min(steps, mean_length) instead is wrong
    # whenever the lengths straddle the budget.
    for st in sorted({b for (s, b) in by if s == "single"}):
        got = [r["nfe"] for r in rows
               if r["strategy"] == "single" and r["steps_requested"] == st]
        exp = [min(st, lens[r["name"]]) for r in rows
               if r["strategy"] == "single" and r["steps_requested"] == st]
        v, e = float(np.mean(got)), float(np.mean(exp))
        flag = "OK" if abs(v - e) < 1.5 else "MISMATCH"
        print(f"   single steps={st}: {v:.1f} calls, ceiling mean(min(steps,L)) "
              f"= {e:.1f} -> {flag}")
        if flag == "MISMATCH":
            fails.append(f"single NFE {v:.1f} != mean(min(steps,L)) {e:.1f}")

    # ---- 3. safety-net use ----------------------------------------------
    # One extra call beyond the allocated steps means the argmax net fired.
    # A partition can never spend more than one call per residue: a block gets
    # at most len(block) steps. So NFE saturates at L and the requested budget
    # stops being the budget once it exceeds the sequence length. This is why
    # the quality-cost curve must be plotted against MEASURED NFE.
    print("\n3. budget saturation (NFE as a fraction of the ceiling L)")
    print(f"   {'strategy':>14} " + " ".join(f"{b:>7}" for b in budgets))
    for s in sorted({r["strategy"] for r in rows}):
        cells = []
        for b in budgets:
            sub = [r for r in rows
                   if r["strategy"] == s and r["steps_requested"] == b]
            cells.append(f"{np.mean([r['nfe'] / r['length'] for r in sub]):7.2f}"
                         if sub else f"{'-':>7}")
        print(f"   {s:>14} " + " ".join(cells))
    sat = [b for b in budgets
           if np.mean([r["nfe"] for r in rows
                       if r["strategy"] == "contiguous"
                       and r["steps_requested"] == b]) < 0.9 * b]
    if sat:
        print(f"   budgets that do NOT deliver the requested calls: {sat} "
              "-> compare strategies at matched measured NFE, never at matched steps")

    # ---- 4. degeneracy ---------------------------------------------------
    print("\n4. degeneracy by strategy (pooled over steps and seeds)")
    print(f"   {'strategy':>14} {'dup rate':>9} {'pair id':>8} {'max run':>8} "
          f"{'entropy':>8} {'recovery':>9}")
    for s in sorted({r["strategy"] for r in rows}):
        sub = [r for r in rows if r["strategy"] == s]
        seqs = [r["sequence"] for r in sub]
        dup = 1.0 - len(set(seqs)) / len(seqs)
        # pairwise identity only within one protein, else lengths differ
        pid = []
        for n in {r["name"] for r in sub}:
            ss = [r["sequence"] for r in sub if r["name"] == n]
            if len(ss) > 1:
                pid.append(pairwise_identity(ss))
        rec = [r["recovery"] for r in sub if "recovery" in r]
        print(f"   {s:>14} {dup:9.3f} {np.mean(pid) if pid else float('nan'):8.3f} "
              f"{max(longest_run(x) for x in seqs):8d} "
              f"{np.mean([entropy(x) for x in seqs]):8.3f} "
              f"{np.mean(rec) if rec else float('nan'):9.3f}")
    print(f"   (entropy of a uniform 20-letter sequence = {np.log(20):.3f}; "
          "natural proteins ~2.9)")

    # ---- 5. recovery sanity ---------------------------------------------
    rec = np.array([r["recovery"] for r in rows if "recovery" in r])
    if rec.size:
        print(f"\n5. recovery: mean {rec.mean():.3f}, range "
              f"[{rec.min():.3f}, {rec.max():.3f}]")
        if rec.max() > 1.0 or rec.min() < 0.0:
            fails.append("recovery outside [0,1]")
        # NULL REFERENCE. Generation here is UNCONDITIONAL: the denoiser is
        # never shown the structure, only a partition derived from it, so a
        # generated sequence has no channel through which to match the
        # particular native it is scored against. The reference is therefore
        # the identity two independent sequences would share by composition
        # alone, sum_a p_gen(a) p_nat(a), not 1/20. Recovery at or below this
        # means the metric carries no signal about the native and cannot
        # separate partitions.
        natives = {r["name"]: r["native"] for r in rows if "native" in r}
        if natives:
            gen = Counter("".join(r["sequence"] for r in rows))
            nat = Counter("".join(natives.values()))
            gtot, ntot = sum(gen.values()), sum(nat.values())
            null = sum((gen[a] / gtot) * (nat[a] / ntot) for a in CANONICAL)
            nat_null = sum((nat[a] / ntot) ** 2 for a in CANONICAL)
            print(f"   composition null (generated vs native, independent): {null:.3f}")
            print(f"   composition null (native vs native): {nat_null:.3f}; "
                  f"uniform 1/20 = 0.050")
            if rec.mean() <= null + 0.01:
                print("   -> recovery is AT the composition null: unconditional "
                      "generation cannot recover a specific native, so this "
                      "metric cannot rank partitions.")

    print("\n" + "=" * 68)
    if fails:
        print("FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS: no mask leakage, NFE accounting consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
