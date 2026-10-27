"""The energy, the mutation energy difference and the conditional must agree."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from encoding import Alphabet, read_single_sequence
from potts import PottsModel
from sites import load_mapping, make_chimera, mutable_columns, resolve_sites


@pytest.fixture
def sequence(model, alphabet) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, alphabet.q, size=model.L)


def test_delta_energy_matches_recomputed_energy(model, sequence):
    rng = np.random.default_rng(11)
    for _ in range(20):
        site = int(rng.integers(0, model.L))
        new_state = int(rng.integers(0, model.q))
        mutant = sequence.copy()
        mutant[site] = new_state
        expected = model.energy(mutant) - model.energy(sequence)
        assert model.delta_energy(sequence, site, new_state) == pytest.approx(expected, abs=1e-8)


def test_conditional_is_the_boltzmann_conditional(model, sequence):
    site = 5
    energies = []
    for state in range(model.q):
        variant = sequence.copy()
        variant[site] = state
        energies.append(model.energy(variant))
    expected = torch.softmax(-torch.tensor(energies, dtype=torch.float64), dim=0).numpy()
    assert model.conditional(site, sequence) == pytest.approx(expected, abs=1e-8)


def test_conditional_ignores_the_residue_at_the_site(model, sequence, alphabet):
    """P(sigma_i | rest) must not depend on sigma_i -- the sampler relies on it."""
    variant = sequence.copy()
    variant[4] = (int(sequence[4]) + 3) % alphabet.q
    assert model.conditional(4, sequence) == pytest.approx(model.conditional(4, variant), abs=1e-10)


def test_log_conditional_matches_conditional(model, sequence, alphabet):
    state = alphabet.index("S")
    probability = model.conditional(6, sequence, beta=1.3)[state]
    assert np.exp(model.log_conditional(6, sequence, state, beta=1.3)) == pytest.approx(probability)


def test_beta_sharpens_the_conditional(model, sequence):
    cold = model.conditional(7, sequence, beta=4.0)
    warm = model.conditional(7, sequence, beta=0.25)
    assert cold.max() > warm.max()


def test_alphabet_roundtrip():
    """The default is adabmDCA's 'protein': the gap is token 0."""
    alphabet = Alphabet()
    assert alphabet.tokens == "-ACDEFGHIKLMNPQRSTVWY"
    assert alphabet.decode(alphabet.encode("ACDWY-")) == "ACDWY-"
    assert alphabet.gap_index == 0
    assert alphabet.q == 21
    assert alphabet.gap_index not in set(alphabet.residue_indices.tolist())
    assert alphabet.encode("acdwy.").tolist() == alphabet.encode("ACDWY-").tolist()


def test_alphabet_conventions_give_the_same_energies(model_file, wt_file):
    """Gap-first and gap-last tokens are a relabelling: energies must match."""
    _, wt_string = read_single_sequence(wt_file)
    energies = []
    for name in ("protein", "ACDEFGHIKLMNPQRSTVWY-"):
        alphabet = Alphabet.from_name(name)
        loaded = PottsModel.load(model_file, alphabet=alphabet, dtype="float64")
        energies.append(loaded.energy(alphabet.encode(wt_string)))
    assert energies[0] == pytest.approx(energies[1], abs=1e-8)


def test_site_resolution_and_gap_filtering(mapping_file, wt_file, alphabet):
    mapping = load_mapping(mapping_file)
    assert mapping.column("189") == 8
    labels, columns, missing = resolve_sites("160-228", mapping)
    assert missing == []
    assert "18" not in labels and "188A" in labels
    assert columns == sorted(columns)

    _, wt_string = read_single_sequence(wt_file)
    wt = alphabet.encode(wt_string)
    kept = mutable_columns(columns, [wt], alphabet)
    assert 2 not in kept  # the gap column is never mutable

    labels_sho, _, _ = resolve_sites("sho", mapping)
    assert "18" in labels_sho


def test_mapping_base_offsets_the_columns(tmp_path):
    """Julia files count MSA columns from 1; the code stores 0-based columns."""
    path = tmp_path / "mapping_julia.csv"
    path.write_text("pdb_residue,msa_col\n189,9\n215,11\n")
    assert load_mapping(path, base=0).column("189") == 9
    assert load_mapping(path, base=1).column("189") == 8
    zero_based = tmp_path / "mapping_python.csv"
    zero_based.write_text("pdb_residue,msa_col\n189,0\n")
    with pytest.raises(ValueError, match="negative"):
        load_mapping(zero_based, base=1)


def test_make_chimera_skips_gaps(alphabet):
    source = alphabet.encode("ACD-FG")
    target = alphabet.encode("WWWWW-")
    chimera, changed = make_chimera(source, target, [0, 1, 2, 3, 4, 5], alphabet)
    assert alphabet.decode(chimera) == "WWW-WG"
    assert changed == [0, 1, 2, 4]
