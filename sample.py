"""Block-wise masked diffusion sampling.

Generation follows the partition in order. For block b the prefix blocks are
clean and fixed, the suffix blocks stay fully masked, and the positions inside
block b are revealed by ordinary masked diffusion over the block's own time
axis. Nothing about the rate matrix, the schedule or the denoiser changes; only
which positions share a block.

Budget
------
NFE is one denoiser call per reveal step summed over blocks. Steps are
allocated to blocks in proportion to block size, so total NFE is a single
controllable number and every partition can be run at the same budget. Note
that an autoregressive partition has a floor of L calls, so the quality-cost
curve for that strategy has exactly one point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import MASK, Denoiser
from .partitions import Partition


@dataclass
class Sample:
    sequence: str
    nfe: int
    strategy: str
    name: str
    steps_requested: int

    def as_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "nfe": self.nfe,
            "strategy": self.strategy,
            "name": self.name,
            "steps_requested": self.steps_requested,
            "length": len(self.sequence),
        }


def _allocate_steps(partition: Partition, total_steps: int, length: int) -> list[int]:
    """Steps per block, proportional to block size, at least one each."""
    alloc = []
    for block in partition:
        n = int(round(total_steps * len(block) / length))
        alloc.append(max(1, min(n, len(block))))
    return alloc


@torch.no_grad()
def sample_sequence(
    denoiser: Denoiser,
    partition: Partition,
    length: int,
    *,
    total_steps: int = 64,
    temperature: float = 1.0,
    seed: int = 0,
    name: str = "",
    strategy: str = "",
) -> Sample:
    """Generate one sequence under the given partition."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.full((1, length), MASK, device=denoiser.device, dtype=torch.long)
    steps = _allocate_steps(partition, total_steps, length)
    nfe_start = denoiser.n_calls

    for block, n_steps in zip(partition, steps):
        block_t = torch.as_tensor(np.asarray(block), device=denoiser.device)
        remaining = block_t.clone()

        for s in range(n_steps):
            if remaining.numel() == 0:
                break
            logp = denoiser.log_marginals(x)[0]           # (L, 20)

            # MDLM tau-leap on the block's own time axis, alpha_t = 1 - t
            t = 1.0 - s / n_steps
            t_next = 1.0 - (s + 1) / n_steps
            reveal_p = 1.0 if t <= 0 else min(1.0, max(0.0, (t - t_next) / t))
            if s == n_steps - 1:
                reveal_p = 1.0

            draws = torch.rand(remaining.shape, generator=g).to(denoiser.device)
            chosen = remaining[draws < reveal_p]
            if chosen.numel() == 0:
                continue

            logits = logp[chosen] / max(temperature, 1e-6)
            probs = logits.softmax(dim=-1).cpu()
            picks = torch.multinomial(probs, num_samples=1, generator=g).squeeze(1)
            x[0, chosen] = picks.to(denoiser.device)

            keep = ~torch.isin(remaining, chosen)
            remaining = remaining[keep]

        if remaining.numel() > 0:                          # safety net
            logp = denoiser.log_marginals(x)[0]
            x[0, remaining] = logp[remaining].argmax(dim=-1)

    return Sample(
        sequence=denoiser.decode(x)[0],
        nfe=denoiser.n_calls - nfe_start,
        strategy=strategy,
        name=name,
        steps_requested=total_steps,
    )
