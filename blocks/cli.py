"""Command line interface.

    python -m blocks.cli smoke
    python -m blocks.cli bound   --pdb-dir blocks/data/cath --out blocks/results/bound.jsonl
    python -m blocks.cli sample  --pdb-dir blocks/data/cath --out blocks/results/samples.jsonl
    python -m blocks.cli fold    --samples blocks/results/samples.jsonl --out blocks/results/folded.jsonl
    python -m blocks.cli figures --bound blocks/results/bound.jsonl --folded blocks/results/folded.jsonl

Every subcommand writes JSONL so runs can be resumed, merged across machines,
and traced back to the exact configuration that produced them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Support both `python cli.py` and `python -m package.cli`
try:
    from . import bound as bound_mod
    from . import evaluate as eval_mod
    from . import plots as plot_mod
    from .model import load_denoiser
    from .partitions import STRATEGIES, make_partition
    from .sample import sample_sequence
    from .structure import load_dataset
except ImportError:
    import bound as bound_mod
    import evaluate as eval_mod
    import plots as plot_mod
    from model import load_denoiser
    from partitions import STRATEGIES, make_partition
    from sample import sample_sequence
    from structure import load_dataset

DEFAULT_STRATEGIES = [
    "contiguous", "anticluster", "cluster", "random", "burial", "sse",
    "autoregressive", "single",
]


def _write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


def _read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------

def cmd_bound(args):
    records = load_dataset(args.pdb_dir, limit=args.limit, max_length=args.max_length)
    denoiser = load_denoiser(args.model, args.device, batch_size=args.batch_size)
    rows = []

    if args.check:
        chk = bound_mod.check_autoregressive_exactness(denoiser, records[0].sequence)
        print(f"autoregressive exactness check: {chk}")
        if not chk["passed"]:
            print("WARNING: estimator does not reduce to AR at block size 1")

    for rec in records:
        for strategy in args.strategies:
            part = make_partition(
                strategy, rec, block_size=args.block_size,
                seed=args.seed, order_by=args.order_by,
            )
            res = bound_mod.estimate_bound(
                denoiser, rec.sequence, part, rec.contacts,
                name=rec.name, strategy=strategy,
                n_t=args.n_t, max_blocks=args.max_blocks, seed=args.seed,
            )
            rows.append(res.as_dict())
            print(f"{rec.name:12s} {strategy:15s} "
                  f"bound/res={res.nll_per_residue:.4f} "
                  f"coupling={res.coupling_mass:.3f}")
    _write(args.out, rows)


def cmd_sample(args):
    records = load_dataset(args.pdb_dir, limit=args.limit, max_length=args.max_length)
    denoiser = load_denoiser(args.model, args.device, batch_size=1)
    rows = []
    for rec in records:
        for strategy in args.strategies:
            part = make_partition(
                strategy, rec, block_size=args.block_size,
                seed=args.seed, order_by=args.order_by,
            )
            for steps in args.steps:
                for s in range(args.n_seeds):
                    smp = sample_sequence(
                        denoiser, part, rec.length,
                        total_steps=steps, temperature=args.temperature,
                        seed=args.seed + 1000 * s, name=rec.name, strategy=strategy,
                    )
                    row = smp.as_dict()
                    row["native"] = rec.sequence
                    row["seed"] = args.seed + 1000 * s
                    row["recovery"] = eval_mod.sequence_recovery(
                        smp.sequence, rec.sequence
                    )
                    rows.append(row)
            print(f"{rec.name:12s} {strategy:15s} done")
    _write(args.out, rows)


def cmd_fold(args):
    rows = _read(args.samples)
    folder = eval_mod.Folder(device=args.device, chunk_size=args.chunk_size)
    for i, r in enumerate(rows):
        try:
            r.update(folder.fold(r["sequence"]))
        except Exception as exc:              # OOM on long sequences is normal
            print(f"fold failed on row {i}: {exc}")
            r["plddt"] = float("nan")
            r["ptm"] = float("nan")
        if (i + 1) % 20 == 0:
            print(f"folded {i + 1}/{len(rows)}")
    _write(args.out, rows)


def cmd_figures(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.bound:
        brows = _read(args.bound)
        plot_mod.plot_bound_by_strategy(brows, outdir / "fig1_bound_by_strategy.pdf")
        plot_mod.plot_mechanism(brows, outdir / "fig2_mechanism.pdf")

        summary = []
        for s in sorted({r["strategy"] for r in brows}):
            v = np.array([r["nll_per_residue"] for r in brows if r["strategy"] == s])
            c = np.array([r["coupling_mass"] for r in brows if r["strategy"] == s])
            lo, hi = eval_mod.bootstrap_ci(list(v))
            summary.append({
                "strategy": s, "n": len(v),
                "bound_per_residue": float(v.mean()),
                "ci_low": lo, "ci_high": hi,
                "coupling_mass": float(c.mean()),
            })
        plot_mod.write_markdown_table(
            summary, outdir / "table_bound.md",
            ["strategy", "n", "bound_per_residue", "ci_low", "ci_high", "coupling_mass"],
        )
        print(f"bound figures written to {outdir}")

    if args.folded:
        frows = _read(args.folded)
        agg = []
        keys = {(r["strategy"], r["steps_requested"]) for r in frows}
        for strategy, steps in sorted(keys):
            subset = [r for r in frows
                      if r["strategy"] == strategy and r["steps_requested"] == steps]
            row = eval_mod.summarise(subset)
            row["strategy"] = strategy
            row["steps_requested"] = steps
            agg.append(row)
        plot_mod.plot_quality_cost(
            agg, outdir / "fig3_recovery_vs_nfe.pdf",
            metric="recovery_mean", ylabel="Sequence recovery $\\uparrow$",
        )
        if any("plddt_mean" in r for r in agg):
            plot_mod.plot_quality_cost(
                agg, outdir / "fig4_plddt_vs_nfe.pdf",
                metric="plddt_mean", ylabel="ESMFold pLDDT $\\uparrow$",
            )
        plot_mod.write_markdown_table(
            agg, outdir / "table_quality.md",
            ["strategy", "steps_requested", "n", "nfe_mean", "recovery_mean",
             "recovery_sem", "plddt_mean", "foldability_rate", "duplicate_rate"],
        )
        print(f"quality figures written to {outdir}")


def cmd_smoke(args):
    """End-to-end run on a synthetic record. No structures, no download."""
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    L = 48
    seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=L))
    contacts = np.zeros((L, L), dtype=bool)
    for i in range(L):
        for j in range(i + 3, L):
            if rng.random() < 0.05:
                contacts[i, j] = contacts[j, i] = True
    rec = SimpleNamespace(
        name="synthetic", sequence=seq, length=L,
        sse="c" * L, rsa=rng.random(L).astype(np.float32),
        contacts=contacts, buried=lambda threshold=0.25: rng.random(L) < threshold,
    )

    print("partition shapes and coupling mass:")
    try:
        from .partitions import coupling_mass
    except ImportError:
        from partitions import coupling_mass
    for s in STRATEGIES:
        part = make_partition(s, rec, block_size=8)
        assert sorted(np.concatenate(part).tolist()) == list(range(L)), \
            f"{s} is not a partition"
        print(f"  {s:15s} blocks={len(part):3d} coupling={coupling_mass(part, contacts):.3f}")

    if args.with_model:
        denoiser = load_denoiser(args.model, args.device, batch_size=4)
        chk = bound_mod.check_autoregressive_exactness(denoiser, seq[:24])
        print(f"exactness check: {chk}")
        part = make_partition("contiguous", rec, block_size=8)
        res = bound_mod.estimate_bound(denoiser, seq, part, contacts, n_t=4)
        print(f"bound/residue = {res.nll_per_residue:.4f} ({res.n_forward} forwards)")
        smp = sample_sequence(denoiser, part, L, total_steps=16)
        print(f"sample ({smp.nfe} NFE): {smp.sequence}")
    print("smoke test passed")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(prog="blocks")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--block-size", type=int, default=16)
        sp.add_argument("--order-by", default="natural",
                        choices=["natural", "burial", "size", "random"])
        sp.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES,
                        choices=list(STRATEGIES))
        sp.add_argument("--pdb-dir", required=True)
        sp.add_argument("--limit", type=int, default=50)
        sp.add_argument("--max-length", type=int, default=256)

    sb = sub.add_parser("bound", help="estimate the block-decomposed NLL bound")
    common(sb)
    sb.add_argument("--n-t", type=int, default=8)
    sb.add_argument("--max-blocks", type=int, default=16)
    sb.add_argument("--batch-size", type=int, default=8)
    sb.add_argument("--check", action="store_true",
                    help="run the autoregressive exactness check first")
    sb.add_argument("--out", default="blocks/results/bound.jsonl")
    sb.set_defaults(func=cmd_bound)

    ss = sub.add_parser("sample", help="generate sequences under each partition")
    common(ss)
    ss.add_argument("--steps", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ss.add_argument("--n-seeds", type=int, default=5)
    ss.add_argument("--temperature", type=float, default=1.0)
    ss.add_argument("--out", default="blocks/results/samples.jsonl")
    ss.set_defaults(func=cmd_sample)

    sf = sub.add_parser("fold", help="score sequences with ESMFold")
    sf.add_argument("--samples", required=True)
    sf.add_argument("--out", default="blocks/results/folded.jsonl")
    sf.add_argument("--device", default="cuda")
    sf.add_argument("--chunk-size", type=int, default=64)
    sf.set_defaults(func=cmd_fold)

    sg = sub.add_parser("figures", help="write figures and tables")
    sg.add_argument("--bound")
    sg.add_argument("--folded")
    sg.add_argument("--outdir", default="blocks/figures")
    sg.set_defaults(func=cmd_figures)

    sk = sub.add_parser("smoke", help="fast self-test")
    sk.add_argument("--with-model", action="store_true")
    sk.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    sk.add_argument("--device", default="cpu")
    sk.set_defaults(func=cmd_smoke)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
