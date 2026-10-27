"""3TGI residue numbering -> MSA column mapping, site selection and chimeras.

The mapping CSV has (at least) the columns ``pdb_residue`` and ``msa_col``.
``pdb_residue`` is a 3TGI (chymotrypsinogen-numbering) label, possibly with an
insertion code such as ``188A``; ``msa_col`` is the MSA column. The column index
is read as **0-based Python** indexing by default; pass ``base=1`` for files
written by the Julia pipeline, where column 1 is the first MSA column.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from encoding import Alphabet

#: The 17 S1-pocket positions of Sho et al., in 3TGI numbering.
SHO_3TGI_POSITIONS: Tuple[str, ...] = (
    "18", "160", "172", "180", "183", "184", "188", "188A",
    "189", "192", "215", "216", "219", "221", "226", "227", "228",
)

#: The specificity-switching site: D in trypsin, S in chymotrypsin.
CRITIC_3TGI_POSITION = "189"

_PDB_COLUMN_ALIASES = ("pdb_residue", "pdb_res", "pdb", "residue", "pdb_residue_3tgi")
_MSA_COLUMN_ALIASES = ("msa_col", "msa_column", "col", "column", "msa_idx")

_LABEL_RE = re.compile(r"^(-?\d+)([A-Za-z]?)$")


@dataclass(frozen=True)
class SiteMapping:
    """3TGI residue labels <-> 0-based MSA columns."""

    label_to_column: Dict[str, int]

    @property
    def labels(self) -> List[str]:
        return list(self.label_to_column)

    def column(self, label: str) -> int:
        key = str(label).strip().upper()
        if key not in self.label_to_column:
            raise KeyError(f"3TGI position {label!r} is not in the mapping file")
        return self.label_to_column[key]

    def column_to_label(self) -> Dict[int, str]:
        return {column: label for label, column in self.label_to_column.items()}


def load_mapping(path: str | Path, base: int = 0) -> SiteMapping:
    """Load the 3TGI -> MSA column mapping CSV.

    Args:
        path: CSV with columns ``pdb_residue`` and ``msa_col``.
        base: index base of ``msa_col`` in the file (0 for Python, 1 for Julia).
    """
    if base not in (0, 1):
        raise ValueError("base must be 0 (Python) or 1 (Julia)")
    frame = pd.read_csv(path)
    pdb_column = _pick_column(frame, _PDB_COLUMN_ALIASES, path, "pdb_residue")
    msa_column = _pick_column(frame, _MSA_COLUMN_ALIASES, path, "msa_col")

    mapping: Dict[str, int] = {}
    for label, column in zip(frame[pdb_column], frame[msa_column]):
        if pd.isna(label) or pd.isna(column):
            continue
        key = str(label).strip().upper()
        value = int(column) - base
        if value < 0:
            raise ValueError(
                f"{path}: MSA column {column} for residue {label} is negative with base={base}"
            )
        mapping[key] = value
    if not mapping:
        raise ValueError(f"{path} holds no usable mapping rows")
    return SiteMapping(mapping)


@dataclass(frozen=True)
class _RangeToken:
    """An inclusive range of 3TGI residue numbers, resolved against the mapping."""

    start: int
    end: int


_RANGE_RE = re.compile(r"^(-?\d+)\s*-\s*(-?\d+)$")


def parse_site_spec(spec: str) -> List[str | _RangeToken]:
    """Expand a ``--allowed-sites`` specification into labels and ranges.

    Accepts ``sho`` (the 17 Sho et al. positions), a comma-separated list of
    labels (``18,160,188A``), inclusive ranges (``160-228``), or any mixture of
    the three. Ranges cover insertion codes too: ``188-190`` includes ``188A``.
    """
    tokens: List[str | _RangeToken] = []
    for token in str(spec).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "sho":
            tokens.extend(SHO_3TGI_POSITIONS)
            continue
        match = _RANGE_RE.match(token)
        if match:
            tokens.append(_RangeToken(int(match.group(1)), int(match.group(2))))
            continue
        tokens.append(token.upper())
    if not tokens:
        raise ValueError(f"Empty site specification: {spec!r}")
    return tokens


def resolve_sites(
    spec: str,
    mapping: SiteMapping,
) -> Tuple[List[str], List[int], List[str]]:
    """Resolve a site specification to ``(labels, msa_columns, missing_labels)``.

    Labels are returned sorted by MSA column; ranges are expanded over the labels
    that the mapping actually contains.
    """
    tokens = parse_site_spec(spec)
    selected: Dict[str, int] = {}
    missing: List[str] = []
    for token in tokens:
        if isinstance(token, _RangeToken):
            for label, column in mapping.label_to_column.items():
                number = _number(label)
                if number is not None and token.start <= number <= token.end:
                    selected[label] = column
            continue
        try:
            selected[token] = mapping.column(token)
        except KeyError:
            missing.append(token)
    ordered = sorted(selected.items(), key=lambda item: item[1])
    return [label for label, _ in ordered], [column for _, column in ordered], missing


def parse_column_spec(spec: str, length: int | None = None) -> List[int]:
    """Expand a specification of raw MSA columns: ``'40-60,72'`` (0-based)."""
    columns: List[int] = []
    for token in str(spec).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        match = _RANGE_RE.match(token)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            columns.extend(range(start, end + 1))
        else:
            columns.append(int(token))
    if not columns:
        raise ValueError(f"Empty column specification: {spec!r}")
    if any(column < 0 for column in columns):
        raise ValueError(f"MSA columns are 0-based and cannot be negative: {spec!r}")
    if length is not None and max(columns) >= length:
        raise ValueError(f"Column {max(columns)} is outside an alignment of length {length}")
    return columns


def mutable_columns(
    columns: Sequence[int],
    sequences: Iterable[np.ndarray],
    alphabet: Alphabet,
) -> List[int]:
    """Drop the columns that are a gap in any of ``sequences``.

    Gap positions are never mutated: they are not part of the structural model
    and a gap is not a residue the design could carry.
    """
    sequences = list(sequences)
    keep: List[int] = []
    for column in columns:
        if any(int(sequence[column]) == alphabet.gap_index for sequence in sequences):
            continue
        keep.append(int(column))
    return keep


def make_chimera(
    source: np.ndarray,
    target: np.ndarray,
    columns: Sequence[int],
    alphabet: Alphabet,
) -> Tuple[np.ndarray, List[int]]:
    """``source`` carrying the ``target`` residue at every non-gap column.

    Returns the chimera and the columns that actually changed.
    """
    chimera = np.array(source, dtype=np.int64, copy=True)
    changed: List[int] = []
    for column in columns:
        if int(source[column]) == alphabet.gap_index or int(target[column]) == alphabet.gap_index:
            continue
        chimera[column] = int(target[column])
        if int(source[column]) != int(target[column]):
            changed.append(int(column))
    return chimera, sorted(changed)


def _pick_column(frame: pd.DataFrame, aliases: Sequence[str], path, expected: str) -> str:
    lowered = {str(name).strip().lower(): name for name in frame.columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    raise ValueError(
        f"{path} has columns {list(frame.columns)}; a {expected!r} column is required"
    )


def _number(label: str) -> int | None:
    match = _LABEL_RE.match(str(label).strip().upper())
    return int(match.group(1)) if match else None
