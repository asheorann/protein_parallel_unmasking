"""Frozen masked diffusion denoiser.

Nothing here is trained. The whole study runs on a frozen protein masked LM,
which is what makes a four-week timeline realistic and what holds the model,
the training data and the objective fixed across every partition so that the
partition is the only thing that varies.

State convention: a sequence is a LongTensor (B, L) over 0..20, where 0..19 are
the canonical amino acids in CANONICAL order and 20 is the mask symbol. ESM
token ids and the <cls>/<eos> offsets stay inside this file.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
K = len(CANONICAL)
MASK = K

DEFAULT_MODEL = "facebook/esm2_t30_150M_UR50D"


class Denoiser:
    """Wraps a HuggingFace masked LM as d^i_theta(a | z_t).

    ESM-2, MeMDLM and DPLM are all time-free: the noisy state carries the noise
    level and no scalar t is fed to the network. We therefore recover the
    schedule from the revealed fraction rather than conditioning on t.
    """

    def __init__(self, model, tokenizer, device: str, batch_size: int = 8):
        self.model = model.eval().to(device)
        self.tok = tokenizer
        self.device = device
        self.batch_size = batch_size
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.aa_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(a) for a in CANONICAL], device=device
        )
        self.mask_id = tokenizer.mask_token_id
        self.cls_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.n_calls = 0

    # -- encoding ---------------------------------------------------------

    def encode(self, sequence: str) -> torch.Tensor:
        lookup = {a: i for i, a in enumerate(CANONICAL)}
        return torch.tensor(
            [lookup.get(c, 0) for c in sequence], device=self.device
        ).unsqueeze(0)

    def decode(self, x: torch.Tensor) -> list[str]:
        return [
            "".join("X" if i == MASK else CANONICAL[i] for i in row)
            for row in x.tolist()
        ]

    def to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        ids = torch.where(
            x == MASK,
            torch.full_like(x, self.mask_id),
            self.aa_ids[x.clamp(max=K - 1)],
        )
        cls = torch.full((B, 1), self.cls_id, device=x.device, dtype=x.dtype)
        eos = torch.full((B, 1), self.eos_id, device=x.device, dtype=x.dtype)
        return torch.cat([cls, ids, eos], dim=1)

    # -- forward ----------------------------------------------------------

    @torch.no_grad()
    def log_marginals(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, 20) log d^i_theta(a | z_t), restricted to canonical residues.

        Restricting to the 20 canonical tokens enforces the SUBS constraint
        d^i_theta(m | z_t) = 0 by construction.

        Counts one network function evaluation per sequence in the batch, which
        is what the NFE budget charges.
        """
        outs = []
        for i in range(0, x.shape[0], self.batch_size):
            chunk = x[i : i + self.batch_size]
            tokens = self.to_tokens(chunk)
            logits = self.model(
                input_ids=tokens, attention_mask=torch.ones_like(tokens)
            ).logits[:, 1:-1, :]
            outs.append(logits[..., self.aa_ids].log_softmax(dim=-1))
            self.n_calls += chunk.shape[0]
        return torch.cat(outs, dim=0)

    def reset_counter(self) -> None:
        self.n_calls = 0


def load_denoiser(
    name: str = DEFAULT_MODEL, device: str = "cuda", batch_size: int = 8
) -> Denoiser:
    """Load a frozen denoiser.

    Any HuggingFace masked-LM checkpoint on the ESM-2 vocabulary works:
      facebook/esm2_t30_150M_UR50D   default, runs on a modest GPU
      facebook/esm2_t33_650M_UR50D   stronger, needs more memory
      ChatterjeeLab/MeMDLM           membrane-protein masked diffusion
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda not available, falling back to cpu")
        device = "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name)
    return Denoiser(model, tok, device, batch_size=batch_size)
