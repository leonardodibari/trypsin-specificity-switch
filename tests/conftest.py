"""Shared fixtures: a tiny random bmDCA model written in the adabmDCA format."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from encoding import Alphabet, write_sequences  # noqa: E402
from potts import PottsModel  # noqa: E402

L = 12
SEED = 7


@pytest.fixture(scope="session")
def alphabet() -> Alphabet:
    return Alphabet()


@pytest.fixture(scope="session")
def model_file(tmp_path_factory, alphabet) -> Path:
    """A random L=12 model, written as adabmDCA ``h``/``J`` text records."""
    rng = np.random.default_rng(SEED)
    q = alphabet.q
    bias = rng.normal(0.0, 0.5, size=(L, q))
    couplings = rng.normal(0.0, 0.2, size=(L, q, L, q))

    path = tmp_path_factory.mktemp("model") / "params.dat"
    with open(path, "w", encoding="utf-8") as handle:
        for i in range(L):
            for j in range(i + 1, L):
                for a, token_a in enumerate(alphabet.tokens):
                    for b, token_b in enumerate(alphabet.tokens):
                        handle.write(f"J {i} {j} {token_a} {token_b} {couplings[i, a, j, b]:.6f}\n")
        for i in range(L):
            for a, token_a in enumerate(alphabet.tokens):
                handle.write(f"h {i} {token_a} {bias[i, a]:.6f}\n")
    return path


@pytest.fixture(scope="session")
def model(model_file, alphabet) -> PottsModel:
    return PottsModel.load(model_file, alphabet=alphabet, dtype="float64")


@pytest.fixture(scope="session")
def mapping_file(tmp_path_factory) -> Path:
    """A 3TGI -> MSA mapping covering columns 0..L-1, including an insertion code."""
    labels = ["18", "160", "172", "180", "183", "184", "188", "188A", "189", "192", "215", "216"]
    path = tmp_path_factory.mktemp("mapping") / "mapping.csv"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("pdb_residue,msa_col\n")
        for column, label in enumerate(labels):
            handle.write(f"{label},{column}\n")
    return path


@pytest.fixture(scope="session")
def wt_file(tmp_path_factory, alphabet) -> Path:
    """A gapped wild type of length L; column 2 is a gap, column 8 (3TGI 189) is D."""
    sequence = list("IVGGYTCQDNSV")
    sequence[2] = "-"
    path = tmp_path_factory.mktemp("wt") / "wt.fasta"
    write_sequences(path, [("WT_TEST", "".join(sequence))])
    return path
