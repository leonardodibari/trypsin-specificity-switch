"""Alphabet and sequence I/O, delegating to adabmDCA wherever it has a function.

Everything here is a thin layer over the adabmDCA 2.1 API:

* tokens come from :func:`adabmDCA.alphabet.get_tokens` -- the default is
  ``"protein"``, i.e. ``-ACDEFGHIKLMNPQRSTVWY`` with the **gap first** (index 0),
  which is the convention of the parameter files adabmDCA writes;
* encoding and decoding are :func:`adabmDCA.fasta.encode_sequence` and
  :func:`adabmDCA.fasta.decode_sequence`;
* FASTA reading and writing go through :class:`adabmDCA.Alignment`,
  :func:`adabmDCA.read_alignment` and :func:`adabmDCA.write_alignment`.

The only things added on top are the small conveniences adabmDCA has no opinion
about: normalising ``.``/``~`` to ``-``, the index of the gap, and the list of
states a mutation is allowed to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from adabmDCA.alignment import Alignment, read_alignment, write_alignment
from adabmDCA.alphabet import get_tokens
from adabmDCA.fasta import decode_sequence, encode_sequence, validate_alphabet

#: Default alphabet name, resolved by ``adabmDCA.alphabet.get_tokens``.
DEFAULT_ALPHABET = "protein"

#: Characters normalised to the gap when a FASTA record is read.
GAP_CHARACTERS = ".~"


@dataclass(frozen=True)
class Alphabet:
    """An adabmDCA token string plus the index bookkeeping around it."""

    tokens: str = get_tokens(DEFAULT_ALPHABET)
    gap: str = "-"

    @classmethod
    def from_name(cls, name: str = DEFAULT_ALPHABET) -> "Alphabet":
        """``'protein'``, ``'dna'``, ``'rna'`` or an explicit token string."""
        return cls(tokens=get_tokens(name))

    def __post_init__(self) -> None:
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError(f"Duplicated token in alphabet {self.tokens!r}")
        if self.gap not in self.tokens:
            raise ValueError(f"Gap {self.gap!r} is not part of alphabet {self.tokens!r}")

    @property
    def q(self) -> int:
        return len(self.tokens)

    @property
    def gap_index(self) -> int:
        return self.tokens.index(self.gap)

    @property
    def residue_indices(self) -> np.ndarray:
        """Indices of the non-gap tokens -- the states a mutation may propose."""
        return np.array([i for i in range(self.q) if i != self.gap_index], dtype=np.int64)

    def index(self, char: str) -> int:
        char = char.upper()
        if char in GAP_CHARACTERS:
            return self.gap_index
        try:
            return self.tokens.index(char)
        except ValueError as exc:
            raise ValueError(f"Character {char!r} is not in alphabet {self.tokens!r}") from exc

    def encode(self, sequence: str) -> np.ndarray:
        """0-based token indices of one sequence (``adabmDCA.encode_sequence``)."""
        sequence = normalise(sequence)
        validate_alphabet([sequence], tokens=self.tokens)
        return np.asarray(encode_sequence(sequence, tokens=self.tokens), dtype=np.int64)

    def decode(self, encoded: Sequence[int] | np.ndarray) -> str:
        """Inverse of :meth:`encode` (``adabmDCA.decode_sequence``)."""
        return str(decode_sequence(np.asarray(encoded, dtype=int), tokens=self.tokens))

    def is_gap(self, encoded: np.ndarray) -> np.ndarray:
        return np.asarray(encoded) == self.gap_index


def normalise(sequence: str) -> str:
    """Upper-case a sequence and turn ``.``/``~`` into the gap character."""
    sequence = sequence.upper()
    for char in GAP_CHARACTERS:
        sequence = sequence.replace(char, "-")
    return sequence


def read_sequences(path: str | Path) -> List[Tuple[str, str]]:
    """Read an aligned FASTA into ``(name, sequence)`` pairs."""
    alignment = read_alignment(path, format="fasta")
    return [(name, normalise(sequence)) for name, sequence in zip(alignment.names, alignment.sequences)]


def read_single_sequence(path: str | Path) -> Tuple[str, str]:
    """Read a FASTA expected to hold exactly one record (e.g. the wild type)."""
    records = read_sequences(path)
    if len(records) != 1:
        raise ValueError(f"{path} holds {len(records)} records; a single sequence is expected")
    return records[0]


def write_sequences(
    path: str | Path,
    records: Iterable[Tuple[str, str]],
    line_width: int = 60,
) -> Path:
    """Write ``(name, sequence)`` pairs as FASTA (``adabmDCA.write_alignment``)."""
    names, sequences = zip(*records)
    alignment = Alignment(names=names, sequences=sequences)
    return write_alignment(alignment, path, format="fasta", line_width=line_width)


def ungap(sequence: str) -> str:
    """Drop the alignment gaps from a sequence."""
    return sequence.replace("-", "")
