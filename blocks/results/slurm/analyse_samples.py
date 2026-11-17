"""Experiment 2 analysis: does the tightest bound give the best samples?

Local helper, not part of Anushka's code (blocks/results/ is gitignored).

Her `blocks.cli figures --folded` already aggregates quality against NFE. This
adds the three things that run needs to be readable alongside the bound
results, and that the bound analysis showed are necessary here too:

1. PAIRED, NOT MARGINAL. The bound result's whole lesson was that
   between-protein variance (sd 0.26 nats/residue) swamps the between-strategy
   effect (0.02-0.05), so only within-protein contrasts resolve anything. The
   same applies to recovery and pLDDT: every contrast here is paired within
   (protein, step budget, seed) before it is averaged.
2. BUDGET MATCHED ON MEASURED NFE, NOT REQUESTED STEPS. `_allocate_steps`
   gives every block at least one step, so a partition with many blocks spends
   more calls than requested at small budgets. Comparing strategies at equal
   `steps_requested` compares them at unequal compute.
3. THE LINK TO THE BOUND. Per protein, correlate the strategy's bound from
   `bound_kuniform.jsonl` with its sample quality. The hypothesis under test is
   that a tighter bound buys better samples; that is a correlation across
   strategies within a protein, not across proteins.

Degeneracy is reported beside every quality number, because a partition can
raise recovery by collapsing onto a consensus sequence.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def boot(x, n=20000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def longest_run(s):
    best = run = 1
    for a, b in zip(s, s[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if s else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(ROOT / "blocks/results/folded.jsonl"))
    ap.add_argument("--bound", default=str(ROOT / "blocks/results/bound_kuniform.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "blocks/results/experiment2_summary.json"))
    args = ap.parse_args()

    rows = jsonl(args.samples)
    has_fold = any("plddt" in r and r["plddt"] == r["plddt"] for r in rows)
    metrics = ["recovery"] + (["plddt"] if has_fold else [])
    strategies = sorted({r["strategy"] for r in rows})
    names = sorted({r["name"] for r in rows})
    budgets = sorted({r["steps_requested"] for r in rows})
    print(f"{len(rows)} samples, {len(names)} proteins, {len(strategies)} strategies, "
          f"budgets {budgets}, folded={has_fold}")

    out = {"n_samples": len(rows), "folded": has_fold, "strategies": strategies}

    # ---- measured NFE per strategy and budget ----------------------------
    print("\nMEASURED NFE (the x-axis that matters; requested steps are not compute)")
    print(f"  {'strategy':>14} " + " ".join(f"{b:>8}" for b in budgets))
    nfe = {}
    for s in strategies:
        cells = []
        for b in budgets:
            v = [r["nfe"] for r in rows if r["strategy"] == s and r["steps_requested"] == b]
            nfe[(s, b)] = float(np.mean(v)) if v else float("nan")
            cells.append(f"{nfe[(s, b)]:8.0f}")
        print(f"  {s:>14} " + " ".join(cells))
    out["nfe"] = {f"{s}|{b}": nfe[(s, b)] for s in strategies for b in budgets}

    # ---- marginal quality ------------------------------------------------
    for met in metrics:
        print(f"\nMARGINAL {met.upper()} by strategy (mean over proteins, budgets, seeds)")
        print(f"  {'strategy':>14} {'mean':>8} {'95% CI':>20} {'dup':>6} {'maxrun':>7}")
        out[f"marginal_{met}"] = {}
        for s in strategies:
            sub = [r for r in rows if r["strategy"] == s and met in r and r[met] == r[met]]
            if not sub:
                continue
            m, lo, hi = boot([r[met] for r in sub])
            seqs = [r["sequence"] for r in sub]
            dup = 1.0 - len(set(seqs)) / len(seqs)
            mr = max(longest_run(x) for x in seqs)
            print(f"  {s:>14} {m:8.4f} [{lo:8.4f},{hi:8.4f}] {dup:6.3f} {mr:7d}")
            out[f"marginal_{met}"][s] = {"mean": m, "ci": [lo, hi],
                                         "duplicate_rate": dup, "max_run": mr}

    # ---- paired contrasts, matched on (protein, budget, seed) ------------
    key = lambda r: (r["name"], r["steps_requested"], r["seed"])
    for met in metrics:
        table = defaultdict(dict)
        for r in rows:
            if met in r and r[met] == r[met]:
                table[key(r)][r["strategy"]] = r[met]
        print(f"\nPAIRED {met.upper()} CONTRASTS (matched on protein, budget and seed)")
        print(f"  {'contrast':>30} {'mean':>9} {'95% CI':>22} {'n':>5} {'frac>0':>7}")
        out[f"paired_{met}"] = {}
        pairs = [("contiguous", "anticluster"), ("cluster", "anticluster"),
                 ("cluster", "contiguous"), ("sse", "contiguous"),
                 ("random", "anticluster"), ("single", "contiguous"),
                 ("autoregressive", "contiguous")]
        for a, b in pairs:
            d = [v[a] - v[b] for v in table.values() if a in v and b in v]
            if len(d) < 2:
                continue
            m, lo, hi = boot(d)
            print(f"  {a + ' - ' + b:>30} {m:+9.4f} [{lo:+9.4f},{hi:+9.4f}] "
                  f"{len(d):5d} {np.mean(np.array(d) > 0):7.2f}")
            out[f"paired_{met}"][f"{a} - {b}"] = {"mean": m, "ci": [lo, hi], "n": len(d)}

    # ---- does a tighter bound predict better samples? --------------------
    bound = defaultdict(dict)
    for r in jsonl(args.bound):
        bound[r["name"]].setdefault(r["strategy"], []).append(r["nll_per_residue"])
    bound = {n: {s: float(np.mean(v)) for s, v in d.items()} for n, d in bound.items()}

    print("\nBOUND vs SAMPLE QUALITY, within protein (Spearman over strategies)")
    print("  the hypothesis is that a tighter bound (lower) gives better samples,")
    print("  i.e. a NEGATIVE correlation with recovery/pLDDT.")
    from scipy.stats import spearmanr
    out["bound_vs_quality"] = {}
    for met in metrics:
        q = defaultdict(dict)
        for r in rows:
            if met in r and r[met] == r[met]:
                q[r["name"]].setdefault(r["strategy"], []).append(r[met])
        rhos = []
        for n, d in q.items():
            if n not in bound:
                continue
            ss = [s for s in d if s in bound[n]]
            if len(ss) < 4:
                continue
            x = [bound[n][s] for s in ss]
            y = [float(np.mean(d[s])) for s in ss]
            rho = spearmanr(x, y).statistic
            if rho == rho:
                rhos.append(float(rho))
        if rhos:
            m, lo, hi = boot(rhos)
            print(f"  {met:>10}: mean rho {m:+.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"over {len(rhos)} proteins, {np.mean(np.array(rhos) < 0):.2f} negative")
            out["bound_vs_quality"][met] = {"mean_rho": m, "ci": [lo, hi], "n": len(rhos)}

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
