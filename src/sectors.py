"""Which sector of the alignment a column belongs to, and what moved in each.

The study divides the alignment into three disjoint classes of columns:

``red``
    the sector positions, supplied in order of importance by the sector
    analysis. These are the residues the switch is believed to turn on.
``grafting``
    the rest of the grafting segment -- the contiguous window that contains
    every red-sector position but one, and which is what a chimera experiment
    would actually transplant.
``other``
    every remaining column of the alignment.

A run that may mutate the whole alignment is only interesting if one can say
*where* its mutations landed, which is what this module counts. Everything here
works on encoded sequences and is independent of the sampler that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

#: The three classes, in the order they are reported.
SECTORS = ("red", "grafting", "other")


@dataclass(frozen=True)
class Sectors:
    """The sector of every MSA column, plus the residue label it carries."""

    sector_of: np.ndarray  # (L,) of strings
    label_of: np.ndarray  # (L,) of strings: the PDB-style residue name
    red_rank: Dict[int, int]  # column -> 1-based importance rank within the red sector
    source: str = ""

    @property
    def L(self) -> int:
        return int(self.sector_of.shape[0])

    def columns(self, sector: str) -> List[int]:
        """The columns belonging to ``sector``."""
        return [int(c) for c in np.flatnonzero(self.sector_of == sector)]

    def sizes(self) -> Dict[str, int]:
        return {name: len(self.columns(name)) for name in SECTORS}

    def red_order(self) -> List[int]:
        """Red-sector columns from the most to the least important."""
        return [column for column, _ in sorted(self.red_rank.items(), key=lambda kv: kv[1])]

    @classmethod
    def load(cls, path: str | Path, length: int | None = None) -> "Sectors":
        """Read the CSV written beside the alignment.

        Required columns: ``msa_col`` (0-based) and ``sector``. ``pdb_residue``
        and ``red_rank`` are used when present.
        """
        frame = pd.read_csv(path)
        lowered = {str(name).strip().lower(): name for name in frame.columns}
        for required in ("msa_col", "sector"):
            if required not in lowered:
                raise ValueError(f"{path} must have a {required!r} column, found {list(frame.columns)}")
        columns = frame[lowered["msa_col"]].astype(int).to_numpy()
        size = int(columns.max()) + 1 if length is None else int(length)
        if columns.min() < 0 or columns.max() >= size:
            raise ValueError(f"{path} has columns outside an alignment of length {size}")

        sector_of = np.full(size, "other", dtype=object)
        label_of = np.array([str(i) for i in range(size)], dtype=object)
        for column, sector in zip(columns, frame[lowered["sector"]].astype(str)):
            sector = sector.strip().lower()
            if sector not in SECTORS:
                raise ValueError(f"{path}: unknown sector {sector!r}, expected one of {SECTORS}")
            sector_of[column] = sector
        if "pdb_residue" in lowered:
            for column, label in zip(columns, frame[lowered["pdb_residue"]].astype(str)):
                label_of[column] = label.strip()

        red_rank: Dict[int, int] = {}
        if "red_rank" in lowered:
            ranks = pd.to_numeric(frame[lowered["red_rank"]], errors="coerce")
            for column, rank in zip(columns, ranks):
                if not pd.isna(rank):
                    red_rank[int(column)] = int(rank)
        return cls(sector_of=sector_of, label_of=label_of, red_rank=red_rank, source=str(path))


def mutation_counts(
    sequences: np.ndarray,
    reference: np.ndarray,
    sectors: Sectors,
    gap_index: int | None = None,
) -> pd.DataFrame:
    """Per-sequence substitution counts against ``reference``, split by sector.

    ``sequences`` is ``(N, L)`` encoded. Returns one row per sequence with
    ``n_mut_<sector>`` for each sector and, when ``gap_index`` is given, the
    gaps opened and closed relative to the reference -- the two halves of a
    changed gap pattern, which a bare count of gaps cannot separate.
    """
    encoded = np.asarray(sequences, dtype=np.int64)
    if encoded.ndim == 1:
        encoded = encoded[None, :]
    reference = np.asarray(reference, dtype=np.int64)
    if encoded.shape[1] != reference.shape[0]:
        raise ValueError(f"sequences are length {encoded.shape[1]}, the reference {reference.shape[0]}")
    if encoded.shape[1] != sectors.L:
        raise ValueError(f"sequences are length {encoded.shape[1]}, the sector map covers {sectors.L}")

    differs = encoded != reference[None, :]
    out: Dict[str, np.ndarray] = {}
    for name in SECTORS:
        columns = sectors.columns(name)
        out[f"n_mut_{name}"] = (
            differs[:, columns].sum(axis=1).astype(np.int64)
            if columns
            else np.zeros(encoded.shape[0], dtype=np.int64)
        )
    out["n_mut_total"] = differs.sum(axis=1).astype(np.int64)
    if gap_index is not None:
        ref_gap = reference == gap_index
        seq_gap = encoded == gap_index
        out["gaps_opened"] = (seq_gap & ~ref_gap[None, :]).sum(axis=1).astype(np.int64)
        out["gaps_closed"] = (~seq_gap & ref_gap[None, :]).sum(axis=1).astype(np.int64)
    return pd.DataFrame(out)


def per_column_frequency(
    sequences: np.ndarray,
    reference: np.ndarray,
    sectors: Sectors,
    gap_index: int | None = None,
) -> pd.DataFrame:
    """How often each column is mutated, and how often it holds a gap.

    One row per column: its sector, its residue label, the fraction of the
    sequences that differ from the reference there and -- when ``gap_index`` is
    given -- the fraction carrying a gap, against whether the reference has one.
    This is what says whether a run concentrates its mutations on the red sector
    or spreads them, and where in the alignment its gaps moved.
    """
    encoded = np.asarray(sequences, dtype=np.int64)
    if encoded.ndim == 1:
        encoded = encoded[None, :]
    reference = np.asarray(reference, dtype=np.int64)
    differs = encoded != reference[None, :]
    frame = pd.DataFrame(
        {
            "msa_col": np.arange(sectors.L),
            "pdb_residue": sectors.label_of,
            "sector": sectors.sector_of,
            "red_rank": [sectors.red_rank.get(i, np.nan) for i in range(sectors.L)],
            "frac_mutated": differs.mean(axis=0),
            "n_mutated": differs.sum(axis=0).astype(np.int64),
        }
    )
    if gap_index is not None:
        frame["frac_gap"] = (encoded == gap_index).mean(axis=0)
        frame["wt_gap"] = reference == gap_index
    return frame


def enrichment(counts: pd.DataFrame, sectors: Sectors, mutable: Sequence[int] | None = None) -> pd.DataFrame:
    """Observed mutations per sector against what a uniform sampler would give.

    The denominator is the number of *mutable* columns in each sector, not the
    number of columns in it: a run restricted to the grafting segment cannot
    mutate anything else, and comparing it with the whole alignment would
    manufacture an enrichment that is only the site restriction. ``ratio`` above
    1 means the sector takes more mutations than its share of the mutable sites.
    """
    allowed = set(range(sectors.L)) if mutable is None else {int(c) for c in mutable}
    rows = []
    totals = float(sum(counts[f"n_mut_{name}"].sum() for name in SECTORS))
    n_mutable = float(len(allowed))
    for name in SECTORS:
        columns = [c for c in sectors.columns(name) if c in allowed]
        observed = float(counts[f"n_mut_{name}"].sum())
        share_sites = len(columns) / n_mutable if n_mutable else float("nan")
        share_mut = observed / totals if totals else float("nan")
        rows.append(
            {
                "sector": name,
                "mutable_columns": len(columns),
                "share_of_mutable_sites": share_sites,
                "mutations": observed,
                "mean_per_sequence": observed / max(len(counts), 1),
                "share_of_mutations": share_mut,
                "ratio": share_mut / share_sites if share_sites else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def annotate_trajectory(frame: pd.DataFrame, sectors: Sectors) -> pd.DataFrame:
    """Add ``sector`` / ``pdb_residue`` / ``red_rank`` to a ``trajectory.csv``.

    Rows whose ``msa_col_changed`` is empty -- step 0, the wild type -- keep an
    empty sector rather than being assigned one.
    """
    out = frame.copy()
    columns = pd.to_numeric(out.get("msa_col_changed"), errors="coerce")
    out["sector"] = [
        "" if pd.isna(c) else str(sectors.sector_of[int(c)]) for c in columns
    ]
    out["pdb_residue"] = ["" if pd.isna(c) else str(sectors.label_of[int(c)]) for c in columns]
    out["red_rank"] = [
        np.nan if pd.isna(c) else sectors.red_rank.get(int(c), np.nan) for c in columns
    ]
    return out


__all__ = [
    "SECTORS",
    "Sectors",
    "mutation_counts",
    "per_column_frequency",
    "enrichment",
    "annotate_trajectory",
]
