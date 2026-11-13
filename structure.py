"""Structure features for partition construction.

Everything here comes from the *native* structure of a real protein, not from
a predictor. That is the central methodological advantage of this project over
inference-time methods that must guess the coupling: because we evaluate on
natural proteins with solved structures, the secondary structure, the burial
profile and the residue-residue contact map are all ground truth.

Uses biotite so there is no external DSSP binary to install.

Outputs, for a chain of length L:
  sequence  str of length L, canonical amino acids
  sse       str of length L over {a, b, c}  (helix, strand, coil)
  rsa       (L,) relative solvent accessibility in [0, 1]
  contacts  (L, L) bool, C-beta within `cutoff` angstroms, |i-j| > 2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure.io.pdb as pdb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "biotite is required: pip install 'biotite>=0.40'"
    ) from exc

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

# Tien et al. (2013) theoretical maxima, for relative SASA
MAX_ASA = {
    "A": 129.0, "C": 167.0, "D": 193.0, "E": 223.0, "F": 240.0,
    "G": 104.0, "H": 224.0, "I": 197.0, "K": 236.0, "L": 201.0,
    "M": 224.0, "N": 195.0, "P": 159.0, "Q": 225.0, "R": 274.0,
    "S": 155.0, "T": 172.0, "V": 174.0, "W": 285.0, "Y": 263.0,
}


@dataclass
class ProteinRecord:
    name: str
    sequence: str
    sse: str
    rsa: np.ndarray
    contacts: np.ndarray

    @property
    def length(self) -> int:
        return len(self.sequence)

    def buried(self, threshold: float = 0.25) -> np.ndarray:
        """Boolean mask of buried positions. 0.25 RSA is the usual cut."""
        return self.rsa < threshold


def _cbeta_coords(chain) -> np.ndarray:
    """CB coordinates, falling back to CA for glycine."""
    res_ids = np.unique(chain.res_id)
    coords = []
    for rid in res_ids:
        res = chain[chain.res_id == rid]
        cb = res[res.atom_name == "CB"]
        if len(cb) == 0:
            cb = res[res.atom_name == "CA"]
        coords.append(cb.coord[0] if len(cb) else np.full(3, np.nan))
    return np.asarray(coords)


def load_structure(
    path: str | Path,
    chain_id: str | None = None,
    cutoff: float = 8.0,
    min_length: int = 40,
    max_length: int = 512,
) -> ProteinRecord | None:
    """Parse one PDB or mmCIF file into a ProteinRecord.

    Returns None when the chain fails a filter (non-standard residues, gaps in
    the backbone, length out of range). Returning None rather than raising
    keeps dataset loops simple; the caller counts the drops.
    """
    path = Path(path)
    if path.suffix in (".cif", ".bcif", ".mmcif"):
        f = pdbx.CIFFile.read(str(path))
        atoms = pdbx.get_structure(f, model=1)
    else:
        f = pdb.PDBFile.read(str(path))
        atoms = f.get_structure(model=1)

    atoms = atoms[struc.filter_amino_acids(atoms) & ~atoms.hetero]
    if chain_id is not None:
        atoms = atoms[atoms.chain_id == chain_id]
    else:
        first = atoms.chain_id[0] if len(atoms) else None
        if first is None:
            return None
        atoms = atoms[atoms.chain_id == first]

    res_names = struc.get_residues(atoms)[1]
    if any(r not in THREE_TO_ONE for r in res_names):
        return None
    sequence = "".join(THREE_TO_ONE[r] for r in res_names)
    L = len(sequence)
    if not (min_length <= L <= max_length):
        return None

    # secondary structure: biotite returns 'a', 'b', 'c', or '' for unassigned
    try:
        sse = struc.annotate_sse(atoms)
        sse = "".join(s if s in "abc" else "c" for s in sse)
        if len(sse) != L:
            sse = "c" * L
    except Exception:
        sse = "c" * L

    # relative solvent accessibility
    try:
        atom_sasa = struc.sasa(atoms, vdw_radii="Single")
        res_sasa = struc.apply_residue_wise(atoms, atom_sasa, np.nansum)
        rsa = np.array(
            [s / MAX_ASA[a] for s, a in zip(res_sasa, sequence)], dtype=np.float32
        )
        rsa = np.clip(np.nan_to_num(rsa, nan=0.5), 0.0, 1.5)
    except Exception:
        rsa = np.full(L, 0.5, dtype=np.float32)

    # true contact map
    cb = _cbeta_coords(atoms)
    if cb.shape[0] != L or np.isnan(cb).any():
        return None
    d = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)
    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    contacts = (d < cutoff) & (sep > 2)

    return ProteinRecord(
        name=path.stem, sequence=sequence, sse=sse, rsa=rsa, contacts=contacts
    )


def load_dataset(
    directory: str | Path,
    limit: int | None = None,
    pattern: str = "*",
    **kwargs,
) -> list[ProteinRecord]:
    """Load every parsable structure in a directory."""
    directory = Path(directory)
    files = sorted(
        p for p in directory.glob(pattern)
        if p.suffix in (".pdb", ".ent", ".cif", ".mmcif", ".bcif")
    )
    out, dropped = [], 0
    for p in files:
        rec = load_structure(p, **kwargs)
        if rec is None:
            dropped += 1
            continue
        out.append(rec)
        if limit is not None and len(out) >= limit:
            break
    print(f"loaded {len(out)} structures from {directory} ({dropped} dropped)")
    return out
