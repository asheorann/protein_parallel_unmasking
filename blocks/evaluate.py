"""Evaluation of generated sequences.

Three families of metric:

  recovery   identity to the native sequence of the protein whose structure
             supplied the partition. Cheap, deterministic, no folding needed.
  diversity  duplicate rate and mean pairwise identity among generated
             sequences. Guards against the degenerate win where a partition
             improves quality by collapsing onto one consensus sequence.
  folding    ESMFold pLDDT and pTM. Expensive and optional.

A caveat that belongs in the paper: ESMFold shares a trunk with ESM-2, so if
the denoiser is an ESM-2 derivative then pLDDT is a partially circular score.
Recovery against the native sequence does not have that problem, which is why
it is the primary metric here and folding is the secondary one.
"""

from __future__ import annotations

import numpy as np
import torch


def sequence_recovery(generated: str, native: str) -> float:
    """Fraction of positions matching the native sequence."""
    n = min(len(generated), len(native))
    if n == 0:
        return 0.0
    return float(sum(a == b for a, b in zip(generated[:n], native[:n])) / n)


def pairwise_identity(sequences: list[str]) -> float:
    """Mean pairwise identity among equal-length sequences."""
    if len(sequences) < 2:
        return 0.0
    vals = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            a, b = sequences[i], sequences[j]
            n = min(len(a), len(b))
            if n:
                vals.append(sum(x == y for x, y in zip(a[:n], b[:n])) / n)
    return float(np.mean(vals)) if vals else 0.0


def duplicate_rate(sequences: list[str]) -> float:
    if not sequences:
        return 0.0
    return float(1.0 - len(set(sequences)) / len(sequences))


class Folder:
    """ESMFold wrapper. Loaded lazily because the checkpoint is large."""

    def __init__(self, device: str = "cuda", chunk_size: int = 64):
        from transformers import AutoTokenizer, EsmForProteinFolding

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True
        )
        self.model = self.model.eval().to(device)
        if device == "cuda":
            self.model.esm = self.model.esm.half()
            self.model.trunk.set_chunk_size(chunk_size)

    @torch.no_grad()
    def fold(self, sequence: str) -> dict:
        clean = sequence.replace("X", "A")      # safety: no mask leakage
        enc = self.tok(
            [clean], return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)
        out = self.model(enc)
        plddt = out["plddt"][0, :, 1].mean().item() * 100.0
        ptm = float(out["ptm"].item()) if "ptm" in out else float("nan")
        return {"plddt": plddt, "ptm": ptm}


def summarise(
    records: list[dict], fold_key: str = "plddt", threshold: float = 70.0
) -> dict:
    """Aggregate per-sequence records into the numbers a table needs."""
    out: dict = {"n": len(records)}
    if not records:
        return out
    for key in ("recovery", fold_key, "ptm", "nfe"):
        vals = [r[key] for r in records if key in r and r[key] == r[key]]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            out[f"{key}_sem"] = (
                float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            )
    folds = [r[fold_key] for r in records if fold_key in r and r[fold_key] == r[fold_key]]
    if folds:
        out["foldability_rate"] = float(np.mean([f > threshold for f in folds]))
    seqs = [r["sequence"] for r in records if "sequence" in r]
    out["duplicate_rate"] = duplicate_rate(seqs)
    out["pairwise_identity"] = pairwise_identity(seqs)
    return out


def bootstrap_ci(
    values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval on the mean."""
    if len(values) < 2:
        v = float(values[0]) if values else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
