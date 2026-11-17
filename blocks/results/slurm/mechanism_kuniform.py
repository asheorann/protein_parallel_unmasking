"""The mechanism experiment run with the k-uniform estimator.

Local helper, not part of Anushka's code (blocks/results/ is gitignored).

Why this exists: with the 1/t estimator at the shipped defaults (NT=8,
MAXB=16) the decisive cluster-vs-anticluster contrast comes out NEGATIVE, and
at NT=32/MAXB=32 it comes out POSITIVE, with the across-protein bootstrap
calling both significant. That bootstrap resamples proteins and so cannot see
the Monte Carlo error of the estimator itself, which here is the dominant
term: the 1/t weight is heavy-tailed, so the sample mean sits below the true
mean until the rare large draws are sampled.

The k-uniform form has the same expectation. Grouping the Bernoulli-masked
sets by size k and integrating t out of the 1/t-weighted objective gives

    L_b = sum_{k=1..n} (1/k) E_{S uniform of size k}[ f(S) ]

so drawing k ~ U{1..n} and S uniform of that size, with weight n/k, is an
unbiased estimator of the same quantity, with no 1/t tail. At block size one
k is forced to 1 and the weight is 1, so it returns the autoregressive NLL
exactly, with zero variance. Measured here: 5-10x lower standard deviation
per protein, and exact on the AR check.

Writes the same JSONL schema as `blocks.cli bound`, so her plots and tables
read it unchanged.

Usage (from the repo root):
    python blocks/results/slurm/mechanism_kuniform.py --pdb-dir blocks/data/cath
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from blocks.cli import DEFAULT_STRATEGIES  # noqa: E402
from blocks.model import MASK, load_denoiser  # noqa: E402
from blocks.partitions import block_size_stats, coupling_mass, make_partition  # noqa: E402
from blocks.structure import load_dataset  # noqa: E402


@torch.no_grad()
def kuniform_bound(denoiser, sequence, partition, *, n_draws, seed, max_blocks=None):
    """Unbiased bound estimate with k ~ U{1..n} and weight n/k."""
    rng = np.random.default_rng(seed)
    x0 = denoiser.encode(sequence)
    dev = denoiser.device
    B = len(partition)
    if max_blocks is not None and B > max_blocks:
        chosen = rng.choice(B, size=max_blocks, replace=False)
        scale = B / max_blocks
    else:
        chosen = np.arange(B)
        scale = 1.0

    per_block = []
    for b in chosen:
        block = np.asarray(partition[b])
        n = len(block)
        # stratify k over the draws so every size is represented evenly
        u = rng.random(n_draws)
        states, weights, idxs = [], [], []
        for j in range(n_draws):
            k = 1 + min(n - 1, int(np.floor(n * (j + u[j]) / n_draws)))
            sel = rng.choice(n, size=k, replace=False)
            z = x0.clone()
            for q in range(b + 1, B):
                z[0, np.asarray(partition[q])] = MASK
            z[0, block[sel]] = MASK
            states.append(z)
            weights.append(n / k)
            idxs.append(torch.as_tensor(block[sel], device=dev))

        logp = denoiser.log_marginals(torch.cat(states, dim=0))
        vals = []
        for s, (w, idx) in enumerate(zip(weights, idxs)):
            target = x0[0, idx]
            nll = -logp[s, idx, :].gather(1, target[:, None]).squeeze(1).sum()
            vals.append(w * nll.item())
        per_block.append(float(np.mean(vals)))

    per_block = np.asarray(per_block)
    total = float(per_block.sum() * scale)
    if len(per_block) > 1:
        se = float(per_block.std(ddof=1) / np.sqrt(len(per_block)))
        se_total = se * len(per_block) * scale
    else:
        se_total = 0.0
    return total, se_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb-dir", default="blocks/data/cath")
    p.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--n-draws", type=int, default=32)
    p.add_argument("--max-blocks", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seeds", type=int, default=2, help="repeats, to expose MC error")
    p.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    p.add_argument("--out", default="blocks/results/bound_kuniform.jsonl")
    args = p.parse_args()

    records = load_dataset(args.pdb_dir, limit=args.limit, max_length=args.max_length)
    den = load_denoiser(args.model, args.device, batch_size=args.batch_size)

    rows = []
    for i, rec in enumerate(records):
        for strategy in args.strategies:
            part = make_partition(strategy, rec, block_size=args.block_size)
            for sd in range(args.seeds):
                total, se = kuniform_bound(
                    den, rec.sequence, part,
                    n_draws=args.n_draws, seed=sd, max_blocks=args.max_blocks,
                )
                row = {
                    "name": rec.name, "strategy": strategy, "length": rec.length,
                    "nll_total": total, "nll_per_residue": total / rec.length,
                    "stderr_per_residue": se / rec.length,
                    "coupling_mass": coupling_mass(part, rec.contacts),
                    "estimator": "kuniform", "seed": sd,
                }
                row.update(block_size_stats(part))
                rows.append(row)
        print(f"[{i+1}/{len(records)}] {rec.name} done", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
