"""Diagnostic for the block-bound estimator in blocks/bound.py.

Local helper, not part of Anushka's code (blocks/results/ is gitignored).

Question: is the estimator unbiased, and does it reduce to the autoregressive
NLL at block size one, which is the check blocks/README.md says must pass?

Three ways of drawing the masked set inside the current block are compared on
the same frozen denoiser, the same states convention (prefix clean, suffix
masked) and the same proteins:

  forced     as written in bound.py: each position masked with probability t,
             and when none is masked one position is forced to be masked;
             weight 1/t.
  bernoulli  the same without the forcing, i.e. the estimator the bound.py
             docstring derives; a draw with nothing masked contributes zero.
  kuniform   draw the NUMBER of masked positions k uniformly from 1..n, then a
             uniform subset of that size; weight n/k. Integrating t out of the
             1/t-weighted objective gives exactly this form, so it has the same
             expectation as `bernoulli`, with no 1/t tail.

Usage (from the repo root):
    python blocks/results/slurm/diag_estimator.py --pdb-dir blocks/data/pdb --limit 6
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

from blocks.bound import exact_autoregressive_nll  # noqa: E402
from blocks.model import MASK, load_denoiser  # noqa: E402
from blocks.partitions import make_partition  # noqa: E402
from blocks.structure import load_dataset  # noqa: E402

MODES = ("forced", "bernoulli", "kuniform")


@torch.no_grad()
def estimate(denoiser, sequence, partition, *, n_t, seed, mode, max_blocks=None):
    """Same structure and RNG call order as bound.estimate_bound."""
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
        draws = []                      # (weight, bool mask over the block)
        if mode in ("forced", "bernoulli"):
            u = rng.random(n_t)
            ts = np.clip((np.arange(n_t) + u) / n_t, 1e-3, 1.0)
            for t in ts:
                flags = rng.random(n) < t
                if not flags.any():
                    if mode == "forced":
                        flags[rng.integers(n)] = True
                draws.append((1.0 / t, flags))
        elif mode == "kuniform":
            u = rng.random(n_t)
            for j in range(n_t):
                k = 1 + min(n - 1, int(np.floor(n * (j + u[j]) / n_t)))
                flags = np.zeros(n, dtype=bool)
                flags[rng.choice(n, size=k, replace=False)] = True
                draws.append((n / k, flags))
        else:
            raise ValueError(mode)

        states, live = [], []
        for w, flags in draws:
            if not flags.any():
                continue                # contributes exactly zero
            z = x0.clone()
            for j in range(b + 1, B):
                z[0, torch.as_tensor(np.asarray(partition[j]), device=dev)] = MASK
            z[0, torch.as_tensor(block[flags], device=dev)] = MASK
            states.append(z)
            live.append((w, flags))

        vals = [0.0] * (len(draws) - len(live))
        if states:
            logp = denoiser.log_marginals(torch.cat(states, dim=0))
            for s, (w, flags) in enumerate(live):
                idx = torch.as_tensor(block[flags], device=dev)
                target = x0[0, idx]
                nll = -logp[s, idx, :].gather(1, target[:, None]).squeeze(1).sum()
                vals.append(w * nll.item())
        per_block.append(float(np.mean(vals)))

    return float(np.sum(per_block) * scale)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb-dir", default="blocks/data/pdb")
    p.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--max-length", type=int, default=130)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-t", type=int, nargs="+", default=[8, 32])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--out", default="blocks/results/diag_estimator.json")
    args = p.parse_args()

    records = load_dataset(args.pdb_dir, limit=args.limit, max_length=args.max_length)
    den = load_denoiser(args.model, args.device, batch_size=64)
    rows = []

    print("\n== block size 1: estimate / exact autoregressive NLL (must be 1) ==")
    for rec in records:
        exact = exact_autoregressive_nll(den, rec.sequence)
        part = [np.array([i]) for i in range(rec.length)]
        for n_t in args.n_t:
            for mode in MODES:
                r = [estimate(den, rec.sequence, part, n_t=n_t, seed=s, mode=mode) / exact
                     for s in range(args.seeds)]
                rows.append(dict(test="ar", name=rec.name, length=rec.length, n_t=n_t,
                                 mode=mode, exact_nats=exact, ratios=r))
                print(f"{rec.name:8s} L={rec.length:3d} n_t={n_t:2d} {mode:9s} "
                      f"ratio mean={np.mean(r):7.4f} sd={np.std(r, ddof=1):6.4f} "
                      f"min={np.min(r):7.4f} max={np.max(r):7.4f}")

    print(f"\n== contiguous blocks of {args.block_size}: bound in nats / residue ==")
    for rec in records:
        part = make_partition("contiguous", rec, block_size=args.block_size)
        for n_t in args.n_t:
            for mode in MODES:
                v = [estimate(den, rec.sequence, part, n_t=n_t, seed=s, mode=mode) / rec.length
                     for s in range(args.seeds)]
                rows.append(dict(test="contiguous", name=rec.name, length=rec.length,
                                 n_t=n_t, mode=mode, per_residue=v))
                print(f"{rec.name:8s} L={rec.length:3d} n_t={n_t:2d} {mode:9s} "
                      f"mean={np.mean(v):7.4f} sd={np.std(v, ddof=1):6.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
