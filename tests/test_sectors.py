"""The sector map, the per-sector mutation counts, and gaps as a move."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ensemble import EnsembleConfig, run_ensemble
from mcmc import MCMCConfig, run_mcmc
from observables import describe, gap_count
from sectors import (
    SECTORS,
    Sectors,
    annotate_trajectory,
    enrichment,
    mutation_counts,
    per_column_frequency,
)

L = 12


@pytest.fixture
def sector_file(tmp_path):
    """Columns 1 and 3 red, 1..5 the grafting window, the rest other."""
    rows = []
    for column in range(L):
        if column in (1, 3):
            sector, rank = "red", 1 if column == 3 else 2
        elif 1 <= column <= 5:
            sector, rank = "grafting", ""
        else:
            sector, rank = "other", ""
        rows.append(f"{column},res{column},{sector},{rank},A")
    path = tmp_path / "sectors.csv"
    path.write_text("msa_col,pdb_residue,sector,red_rank,wt_residue\n" + "\n".join(rows) + "\n")
    return path


def test_sector_map_round_trips(sector_file):
    sectors = Sectors.load(sector_file, length=L)
    assert sectors.L == L
    # red = {1, 3}; the window 1..5 less the red columns = {2, 4, 5}; the rest.
    assert sectors.sizes() == {"red": 2, "grafting": 3, "other": 7}
    assert sectors.columns("red") == [1, 3]
    # Column 3 has rank 1, so it comes first.
    assert sectors.red_order() == [3, 1]
    assert sectors.label_of[3] == "res3"


def test_sector_map_rejects_unknown_sector(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("msa_col,sector\n0,purple\n")
    with pytest.raises(ValueError, match="unknown sector"):
        Sectors.load(path, length=1)


def test_mutation_counts_split_by_sector(sector_file, alphabet):
    sectors = Sectors.load(sector_file, length=L)
    reference = np.zeros(L, dtype=np.int64)
    sequences = np.zeros((2, L), dtype=np.int64)
    sequences[0, 1] = 5  # red
    sequences[0, 2] = 5  # grafting
    sequences[1, 9] = 5  # other
    sequences[1, 3] = 5  # red

    counts = mutation_counts(sequences, reference, sectors, gap_index=alphabet.gap_index)
    assert counts["n_mut_red"].tolist() == [1, 1]
    assert counts["n_mut_grafting"].tolist() == [1, 0]
    assert counts["n_mut_other"].tolist() == [0, 1]
    assert counts["n_mut_total"].tolist() == [2, 2]
    # The reference is all gaps (index 0), so every substitution closes one.
    assert counts["gaps_closed"].tolist() == [2, 2]
    assert counts["gaps_opened"].tolist() == [0, 0]


def test_mutation_counts_sum_to_total(sector_file):
    sectors = Sectors.load(sector_file, length=L)
    rng = np.random.default_rng(0)
    reference = rng.integers(0, 5, size=L)
    sequences = rng.integers(0, 5, size=(16, L))
    counts = mutation_counts(sequences, reference, sectors)
    per_sector = sum(counts[f"n_mut_{name}"] for name in SECTORS)
    assert (per_sector == counts["n_mut_total"]).all()


def test_enrichment_denominator_is_the_mutable_set(sector_file):
    """A run restricted to the red sector is not 'enriched' for it."""
    sectors = Sectors.load(sector_file, length=L)
    reference = np.zeros(L, dtype=np.int64)
    sequences = np.zeros((4, L), dtype=np.int64)
    sequences[:, 1] = 5  # every mutation inside the red sector

    whole = enrichment(mutation_counts(sequences, reference, sectors), sectors)
    red_whole = whole.set_index("sector").loc["red"]
    assert red_whole["ratio"] > 1  # against the whole alignment, red is over-represented

    restricted = enrichment(
        mutation_counts(sequences, reference, sectors), sectors, mutable=[1, 3]
    )
    red_restricted = restricted.set_index("sector").loc["red"]
    assert red_restricted["share_of_mutable_sites"] == 1.0
    assert red_restricted["ratio"] == pytest.approx(1.0)


def test_per_column_frequency(sector_file):
    sectors = Sectors.load(sector_file, length=L)
    reference = np.zeros(L, dtype=np.int64)
    sequences = np.zeros((4, L), dtype=np.int64)
    sequences[:2, 3] = 7
    frame = per_column_frequency(sequences, reference, sectors)
    assert len(frame) == L
    row = frame.set_index("msa_col").loc[3]
    assert row["frac_mutated"] == pytest.approx(0.5)
    assert row["sector"] == "red"
    assert frame.set_index("msa_col").loc[0, "frac_mutated"] == 0.0


def test_annotate_trajectory_leaves_step_zero_blank(sector_file):
    sectors = Sectors.load(sector_file, length=L)
    frame = pd.DataFrame({"step": [0, 1, 2], "msa_col_changed": ["", 3, 9]})
    annotated = annotate_trajectory(frame, sectors)
    assert annotated["sector"].tolist() == ["", "red", "other"]
    assert annotated["pdb_residue"].tolist() == ["", "res3", "res9"]
    assert annotated["red_rank"].tolist()[1] == 1
    assert np.isnan(annotated["red_rank"].tolist()[2])


# --- insertion codes that differ only in case ----------------------------------


@pytest.fixture
def case_mapping_file(tmp_path):
    """A mapping with both '188a' and '188A', as iter_aln_v2 really has."""
    rows = [("187", 0), ("188", 1), ("188a", 2), ("188b", 3), ("188A", 4), ("189", 5)]
    path = tmp_path / "mapping.csv"
    path.write_text(
        "pdb_residue,msa_col\n" + "\n".join(f"{label},{column}" for label, column in rows) + "\n"
    )
    return path


def test_labels_differing_only_in_case_are_distinct(case_mapping_file):
    from sites import load_mapping

    mapping = load_mapping(case_mapping_file)
    assert mapping.label_to_column["188a"] == 2
    assert mapping.label_to_column["188A"] == 4
    assert mapping.column("188a") == 2
    assert mapping.column("188A") == 4


def test_range_keeps_both_cases(case_mapping_file):
    """A range over 188 must not drop one of the two insertion codes."""
    from sites import load_mapping, resolve_sites

    mapping = load_mapping(case_mapping_file)
    _, columns, missing = resolve_sites("187-189", mapping)
    assert missing == []
    assert columns == [0, 1, 2, 3, 4, 5]


def test_exact_case_wins_over_the_fallback(tmp_path):
    """With both cases present, each label resolves to its own column."""
    from sites import load_mapping

    path = tmp_path / "m.csv"
    path.write_text("pdb_residue,msa_col\n190b,7\n190B,9\n")
    mapping = load_mapping(path)
    assert mapping.column("190b") == 7
    assert mapping.column("190B") == 9


def test_case_insensitive_fallback_when_unambiguous(tmp_path):
    """A lone label still resolves whatever case it is asked for."""
    from sites import load_mapping

    path = tmp_path / "m.csv"
    path.write_text("pdb_residue,msa_col\n191c,4\n")
    mapping = load_mapping(path)
    assert mapping.column("191c") == 4
    assert mapping.column("191C") == 4


def test_unknown_label_still_raises(tmp_path):
    from sites import load_mapping

    path = tmp_path / "m.csv"
    path.write_text("pdb_residue,msa_col\n42,1\n")
    mapping = load_mapping(path)
    with pytest.raises(KeyError, match="not in the mapping"):
        mapping.column("999")


def test_duplicate_label_is_rejected(tmp_path):
    from sites import load_mapping

    path = tmp_path / "dup.csv"
    path.write_text("pdb_residue,msa_col\n42,1\n42,3\n")
    with pytest.raises(ValueError, match="appears twice"):
        load_mapping(path)


# --- gaps as an ordinary state -------------------------------------------------


def _wt_with_gap(alphabet, model):
    sequence = alphabet.encode("IVGGYTCQDNSV")
    sequence[2] = alphabet.gap_index
    assert sequence.shape[0] == model.L
    return sequence


def test_gap_column_is_frozen_by_default(model, alphabet):
    """Without the flag a gap column in the allowed set is an error, not a silent no-op."""
    sequence = _wt_with_gap(alphabet, model)
    with pytest.raises(ValueError, match="gap position"):
        run_mcmc(
            model,
            sequence,
            critic_col=8,
            allowed_cols=[1, 2, 3],
            config=MCMCConfig(lam=0.0, max_steps=10, threshold=2.0),
        )


def test_mcmc_never_proposes_a_gap_by_default(model, alphabet):
    sequence = _wt_with_gap(alphabet, model)
    result = run_mcmc(
        model,
        sequence,
        critic_col=8,
        allowed_cols=[1, 3, 4, 5],
        config=MCMCConfig(lam=0.0, max_steps=4000, threshold=2.0, record_every=1, seed=1),
    )
    for record in result.records:
        # Column 2 starts as a gap and is not in the allowed set, so it stays one;
        # no other column ever acquires a gap, because a gap is never proposed.
        assert record.sequence[2] == "-"
        assert record.sequence.count("-") == 1


def test_mcmc_gap_moves_change_the_gap_pattern(model, alphabet):
    """With the flag a gap column mutates and other columns can become gaps."""
    sequence = _wt_with_gap(alphabet, model)
    result = run_mcmc(
        model,
        sequence,
        critic_col=8,
        allowed_cols=[1, 2, 3, 4, 5],
        config=MCMCConfig(
            lam=0.0, max_steps=8000, threshold=2.0, record_every=1, seed=3, allow_gap_moves=True
        ),
    )
    patterns = {tuple(c == "-" for c in record.sequence) for record in result.records}
    assert len(patterns) > 1, "the gap pattern never changed"
    # The gap column itself was filled at some point.
    assert any(record.sequence[2] != "-" for record in result.records)


def test_ensemble_gap_moves_change_the_gap_pattern(model, alphabet):
    sequence = _wt_with_gap(alphabet, model)
    config = EnsembleConfig(
        lam=0.0, n_chains=64, n_steps=600, seed=5, trace_every=0, allow_gap_moves=True
    )
    result = run_ensemble(
        model, sequence, critic_col=8, allowed_cols=[1, 2, 3, 4, 5], config=config
    )
    gaps = (result.sequences == alphabet.gap_index).sum(axis=1)
    assert gaps.min() != gaps.max(), "every chain kept the same number of gaps"


def test_ensemble_without_gap_moves_keeps_the_pattern(model, alphabet):
    sequence = _wt_with_gap(alphabet, model)
    config = EnsembleConfig(lam=0.0, n_chains=32, n_steps=400, seed=5, trace_every=0)
    result = run_ensemble(model, sequence, critic_col=8, allowed_cols=[1, 3, 4, 5], config=config)
    gaps = result.sequences == alphabet.gap_index
    assert gaps.sum(axis=1).tolist() == [1] * 32
    assert gaps[:, 2].all()


def test_describe_reports_the_gap_count(model, alphabet):
    sequence = _wt_with_gap(alphabet, model)
    batch = np.stack([sequence, sequence])
    batch[1, 4] = alphabet.gap_index
    observables = describe(model, batch, reference=sequence, critic_col=8, target_state=alphabet.index("S"))
    assert observables["n_gaps"].tolist() == [1, 2]
    assert gap_count(model, batch).tolist() == [1, 2]


def test_enrichment_denominator_is_the_mutable_set_not_the_alignment(sector_file):
    """A restricted run must not be scored against columns it could never touch.

    This is the regression for the bug where substitute_critic.py dropped
    `allowed_cols`, annotate_sectors.py fell back to "all columns mutable", and
    the red sector's ratio came out at 2.07 (apparently enriched) when against
    the 79 columns that could actually move it was 0.66 (depleted).
    """
    sectors = Sectors.load(sector_file, length=L)  # red {1,3}, grafting {2,4,5}
    counts = pd.DataFrame({"n_mut_red": [2], "n_mut_grafting": [6], "n_mut_other": [0]})

    restricted = enrichment(counts, sectors, mutable=[1, 2, 3, 4, 5]).set_index("sector")
    assert restricted.loc["red", "mutable_columns"] == 2
    assert restricted.loc["other", "mutable_columns"] == 0
    # red holds 2/5 of the mutable sites and took 2/8 of the mutations.
    assert restricted.loc["red", "ratio"] == pytest.approx((2 / 8) / (2 / 5))
    assert restricted.loc["red", "ratio"] < 1.0

    whole = enrichment(counts, sectors, mutable=None).set_index("sector")
    # The same data scored against all L columns manufactures an enrichment.
    assert whole.loc["red", "ratio"] == pytest.approx((2 / 8) / (2 / L))
    assert whole.loc["red", "ratio"] > 1.0
    assert whole.loc["red", "ratio"] > restricted.loc["red", "ratio"]
