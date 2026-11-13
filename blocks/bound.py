"""Block-decomposed negative log-likelihood bound.

This is the cheap half of the study and the part that carries the theory
credit. No generation and no training: given a frozen denoiser and a held-out
natural protein, we estimate the bound that a given partition implies.

Derivation
----------
The chain rule over an ordered partition is exact:

    log p(x) = sum_b log p(x_{pi_b} | x_{pi_<b})

Each factor is bounded by the MDLM rate loss restricted to that block. With the
linear schedule alpha_t = 1 - t, the reveal weight is w(t) = -alpha_dot / (1 - alpha) = 1/t,
and the bound is

    L(pi) = sum_b E_{t~U(0,1)} [ (1/t) * sum_{i in pi_b, masked} -log d^i_theta(x_0^i | z_t) ]

where the intermediate state z_t has
    prefix blocks  pi_<b   clean,
    current block  pi_b    masked independently with probability t,
    suffix blocks  pi_>b   fully masked.

The suffix is fully masked because at generation time it has not been produced
yet, which is exactly BD3-LM's construction. Since log p(x) does not depend on
the partition, every difference in L(pi) is bound slack, and that is what the
main experiment measures.

Exactness check
---------------
For blocks of size one the estimator must reduce to the autoregressive NLL. A
single-position block contributes a loss term only when that position is
masked, which happens with probability t, and the state z_t is then independent
of t. So the integral collapses:

    E_t[(1/t) * 1{masked} * (-log d)] = integral_0^1 (1/t) * t * (-log d) dt = -log d

`check_autoregressive_exactness` asserts this numerically. It is a genuine
correctness test of the estimator and belongs in the appendix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:
    from .model import MASK, Denoiser
    from .partitions import Partition, block_size_stats, coupling_mass
except ImportError:
    from model import MASK, Denoiser
    from partitions import Partition, block_size_stats, coupling_mass


@dataclass
class BoundResult:
    name: str
    strategy: str
    length: int
    nll_total: float          # nats for the whole sequence
    nll_per_residue: float    # nats per residue, comparable across lengths
    stderr_per_residue: float
    coupling_mass: float
    n_forward: int
    stats: dict

    def as_dict(self) -> dict:
        d = {
            "name": self.name,
            "strategy": self.strategy,
            "length": self.length,
            "nll_total": self.nll_total,
            "nll_per_residue": self.nll_per_residue,
            "stderr_per_residue": self.stderr_per_residue,
            "coupling_mass": self.coupling_mass,
            "n_forward": self.n_forward,
        }
        d.update(self.stats)
        return d


def _build_state(
    x0: torch.Tensor,
    partition: Partition,
    b: int,
    mask_flags: torch.Tensor,
) -> torch.Tensor:
    """z_t for block b: prefix clean, block b partly masked, suffix all masked."""
    z = x0.clone()
    for j in range(b + 1, len(partition)):
        z[0, partition[j]] = MASK
    block = partition[b]
    z[0, block[mask_flags]] = MASK
    return z


@torch.no_grad()
def estimate_bound(
    denoiser: Denoiser,
    sequence: str,
    partition: Partition,
    contacts: np.ndarray,
    *,
    name: str = "",
    strategy: str = "",
    n_t: int = 8,
    max_blocks: int | None = None,
    seed: int = 0,
) -> BoundResult:
    """Monte Carlo estimate of L(pi) for one protein.

    n_t stratified time samples per block. When `max_blocks` is set, blocks are
    subsampled uniformly and the sum is rescaled by B / n_sampled, which keeps
    the estimator unbiased and keeps autoregressive partitions affordable.
    """
    rng = np.random.default_rng(seed)
    x0 = denoiser.encode(sequence)
    L = len(sequence)
    B = len(partition)

    if max_blocks is not None and B > max_blocks:
        chosen = rng.choice(B, size=max_blocks, replace=False)
        scale = B / max_blocks
    else:
        chosen = np.arange(B)
        scale = 1.0

    start = denoiser.n_calls
    per_block_means = []

    for b in chosen:
        block = partition[b]
        # stratified t in (0, 1]
        u = rng.random(n_t)
        ts = (np.arange(n_t) + u) / n_t
        ts = np.clip(ts, 1e-3, 1.0)

        states, weights, masks = [], [], []
        for t in ts:
            flags = torch.from_numpy(rng.random(len(block)) < t).to(denoiser.device)
            if not flags.any():
                flags[rng.integers(len(block))] = True
            states.append(_build_state(x0, partition, b, flags))
            weights.append(1.0 / t)
            masks.append(flags)

        batch = torch.cat(states, dim=0)
        logp = denoiser.log_marginals(batch)           # (n_t, L, 20)

        samples = []
        for s, (w, flags) in enumerate(zip(weights, masks)):
            idx = torch.as_tensor(block, device=denoiser.device)[flags]
            target = x0[0, idx]
            nll = -logp[s, idx, :].gather(1, target[:, None]).squeeze(1).sum()
            samples.append(w * nll.item())
        per_block_means.append(float(np.mean(samples)))

    per_block_means = np.asarray(per_block_means)
    total = float(per_block_means.sum() * scale)
    # standard error across blocks, propagated to the per-residue scale
    if len(per_block_means) > 1:
        se = float(per_block_means.std(ddof=1) / np.sqrt(len(per_block_means)))
        se_total = se * len(per_block_means) * scale
    else:
        se_total = 0.0

    return BoundResult(
        name=name,
        strategy=strategy,
        length=L,
        nll_total=total,
        nll_per_residue=total / L,
        stderr_per_residue=se_total / L,
        coupling_mass=coupling_mass(partition, contacts),
        n_forward=denoiser.n_calls - start,
        stats=block_size_stats(partition),
    )


@torch.no_grad()
def exact_autoregressive_nll(denoiser: Denoiser, sequence: str) -> float:
    """Left-to-right NLL with the suffix masked, in nats.

    This is what the bound must equal when every block has size one.
    """
    x0 = denoiser.encode(sequence)
    L = len(sequence)
    states = []
    for i in range(L):
        z = x0.clone()
        z[0, i:] = MASK
        states.append(z)
    logp = denoiser.log_marginals(torch.cat(states, dim=0))
    total = 0.0
    for i in range(L):
        total -= logp[i, i, x0[0, i]].item()
    return total


def check_autoregressive_exactness(
    denoiser: Denoiser, sequence: str, tol: float = 0.05, n_t: int = 32
) -> dict:
    """Assert the estimator reduces to autoregression at block size one.

    Passing this is evidence the time weighting, the masking distribution and
    the suffix convention are all implemented consistently. Failing it means a
    bug somewhere in the estimator, not an interesting scientific finding.
    """
    partition = [np.array([i]) for i in range(len(sequence))]
    contacts = np.zeros((len(sequence), len(sequence)), dtype=bool)
    est = estimate_bound(
        denoiser, sequence, partition, contacts, strategy="autoregressive", n_t=n_t
    )
    exact = exact_autoregressive_nll(denoiser, sequence)
    rel = abs(est.nll_total - exact) / max(abs(exact), 1e-8)
    return {
        "estimated": est.nll_total,
        "exact": exact,
        "relative_error": rel,
        "passed": bool(rel < tol),
    }
