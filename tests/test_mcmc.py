"""End-to-end behaviour of the biased sampler."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from encoding import Alphabet, read_sequences, read_single_sequence
from mcmc import MCMCConfig, lam_dirname, run_mcmc, save_result


@pytest.fixture
def start(wt_file, alphabet) -> np.ndarray:
    _, wt_string = read_single_sequence(wt_file)
    return alphabet.encode(wt_string)


def _allowed(model, exclude=(2,)):
    return [column for column in range(model.L) if column not in exclude]


def test_a_strong_bias_reaches_the_threshold(model, start, alphabet):
    result = run_mcmc(
        model,
        start,
        critic_col=8,
        allowed_cols=_allowed(model),
        config=MCMCConfig(lam=50.0, threshold=0.3, max_steps=20_000, record_every=5, seed=1),
    )
    assert result.converged
    assert math.exp(result.final_logP) >= 0.3
    assert result.records[-1].step == result.accepted_steps
    assert result.first_crossing_step == result.accepted_steps


def test_recorded_energies_match_the_sequences(model, start, alphabet):
    """The chain updates E and logP incrementally; they must not drift."""
    result = run_mcmc(
        model,
        start,
        critic_col=8,
        allowed_cols=_allowed(model),
        config=MCMCConfig(lam=5.0, threshold=0.95, max_steps=3_000, record_every=1, seed=4),
    )
    state = alphabet.index("S")
    for record in result.records[::7]:
        sequence = alphabet.encode(record.sequence)
        assert model.energy(sequence) == pytest.approx(record.E_DCA, abs=1e-6)
        assert model.log_conditional(8, sequence, state) == pytest.approx(record.logP_S, abs=1e-6)
        assert record.H_eff == pytest.approx(record.E_DCA - 5.0 * record.logP_S, abs=1e-9)


def test_gaps_and_forbidden_sites_are_never_touched(model, start, alphabet):
    allowed = [3, 4, 5, 8]
    result = run_mcmc(
        model,
        start,
        critic_col=8,
        allowed_cols=allowed,
        config=MCMCConfig(lam=2.0, threshold=0.999, max_steps=2_000, record_every=1, seed=5),
    )
    final = alphabet.encode(result.final_sequence)
    changed = {int(c) for c in np.flatnonzero(final != start)}
    assert changed <= set(allowed)
    assert alphabet.gap_index not in set(final[allowed].tolist())
    assert result.final_sequence[2] == "-"


def test_zero_lambda_ignores_the_specificity_site(model, start):
    """lambda = 0 is plain Potts sampling: the bias must not enter H."""
    result = run_mcmc(
        model,
        start,
        critic_col=8,
        allowed_cols=_allowed(model),
        config=MCMCConfig(lam=0.0, threshold=1.01, max_steps=500, record_every=10, seed=6),
    )
    for record in result.records:
        assert record.H_eff == pytest.approx(record.E_DCA)


def test_the_same_seed_reproduces_the_chain(model, start):
    config = MCMCConfig(lam=3.0, threshold=0.9, max_steps=1_000, record_every=10, seed=99)
    first = run_mcmc(model, start, 8, _allowed(model), config)
    second = run_mcmc(model, start, 8, _allowed(model), config)
    assert first.final_sequence == second.final_sequence
    assert first.proposed_steps == second.proposed_steps


def test_outputs_are_written(model, start, tmp_path):
    result = run_mcmc(
        model,
        start,
        critic_col=8,
        allowed_cols=_allowed(model),
        config=MCMCConfig(lam=10.0, threshold=0.8, max_steps=5_000, record_every=10, seed=3),
    )
    directory = save_result(result, tmp_path / lam_dirname(10.0))
    assert directory.name == "lam10"

    frame = __import__("pandas").read_csv(directory / "trajectory.csv")
    assert list(frame.columns) == [
        "step", "msa_col_changed", "aa_from", "aa_to", "E_DCA", "logP_S", "H_eff"
    ]
    assert frame["step"].is_monotonic_increasing

    records = read_sequences(directory / "sequences.fasta")
    assert len(records) == len(frame)
    assert records[0][0] == "step_0"
    assert all(len(sequence) == model.L for _, sequence in records)

    summary = json.loads((directory / "summary.json").read_text())
    assert summary["lam"] == 10.0
    assert summary["final_P_S"] == pytest.approx(math.exp(result.final_logP))


def test_an_all_gap_site_is_rejected(model, start):
    with pytest.raises(ValueError, match="gap"):
        run_mcmc(model, start, 8, [2], MCMCConfig(max_steps=10))


# --- the Hamming restraint, single chain ----------------------------------------
# The same term as in :mod:`ensemble`, over the allowed columns and split into its
# substitution and indel parts: H += t_sub * d_sub + t_indel * d_indel. The chain
# carries both incrementally and recomputes them at the end, so a leak between the
# two parts raises rather than quietly repricing the run.


def test_t_zero_reproduces_the_unrestrained_chain(model, start):
    """Every single-chain result sampled before the restraint existed still stands."""
    plain = MCMCConfig(lam=3.0, threshold=1.01, max_steps=2_000, record_every=10, seed=8)
    restrained = MCMCConfig(lam=3.0, threshold=1.01, max_steps=2_000, record_every=10,
                            seed=8, t=0.0, t_sub=None, t_indel=None)
    first = run_mcmc(model, start, critic_col=8, allowed_cols=_allowed(model), config=plain)
    second = run_mcmc(model, start, critic_col=8, allowed_cols=_allowed(model),
                      config=restrained)
    assert first.final_sequence == second.final_sequence
    assert first.accepted_steps == second.accepted_steps
    assert first.final_energy == second.final_energy


def test_the_restraint_holds_the_chain_near_the_wild_type(model, start, alphabet):
    """<d> is monotone decreasing in the price: an identity, not a tendency."""
    distances = []
    for t in (-0.5, 0.0, 2.0, 6.0):
        result = run_mcmc(
            model, start, critic_col=8, allowed_cols=_allowed(model),
            config=MCMCConfig(lam=1.0, threshold=1.01, max_steps=8_000, record_every=10_000,
                              seed=12, t=t),
        )
        final = alphabet.encode(result.final_sequence)
        distances.append(int(np.sum(final != start)))
    assert all(a >= b for a, b in zip(distances, distances[1:])), distances
    assert distances[-1] < distances[0]


def test_the_recorded_H_eff_carries_the_split_restraint(model, start, alphabet):
    """H_eff must equal beta*E - lam*logP + t_sub*d_sub + t_indel*d_indel recomputed
    from the recorded sequence itself -- which is also the check that the restraint
    is measured over the ALLOWED columns and not over all of L."""
    t_sub, t_indel, lam, beta = 0.6, -0.9, 2.0, 1.1
    allowed = _allowed(model)
    gap = alphabet.gap_index
    result = run_mcmc(
        model, start, critic_col=8, allowed_cols=allowed,
        config=MCMCConfig(lam=lam, beta=beta, threshold=1.01, max_steps=3_000,
                          record_every=1, seed=14, t_sub=t_sub, t_indel=t_indel,
                          allow_gap_moves=True),
    )
    state = alphabet.index("S")
    for record in result.records[::11]:
        sequence = alphabet.encode(record.sequence)
        differs = sequence[allowed] != start[allowed]
        indel = differs & ((sequence[allowed] == gap) | (start[allowed] == gap))
        expected = (
            beta * model.energy(sequence)
            - lam * model.log_conditional(8, sequence, state, beta=1.0)
            + t_sub * int((differs & ~indel).sum())
            + t_indel * int(indel.sum())
        )
        assert record.H_eff == pytest.approx(expected, abs=1e-6)


def test_a_dear_indel_price_suppresses_gaps_in_the_single_chain(model, start, alphabet):
    """t_indel acts on the indels alone, which is the whole reason it is separate."""
    allowed = _allowed(model)
    gap = alphabet.gap_index

    def indel_count(t_indel):
        result = run_mcmc(
            model, start, critic_col=8, allowed_cols=allowed,
            config=MCMCConfig(lam=1.0, threshold=1.01, max_steps=12_000, record_every=10_000,
                              seed=15, t_sub=0.0, t_indel=t_indel, allow_gap_moves=True),
        )
        final = alphabet.encode(result.final_sequence)
        differs = final[allowed] != start[allowed]
        indel = differs & ((final[allowed] == gap) | (start[allowed] == gap))
        return int(indel.sum()), int((differs & ~indel).sum())

    free_indel, free_sub = indel_count(-2.0)
    dear_indel, dear_sub = indel_count(8.0)
    assert dear_indel < free_indel, (free_indel, dear_indel)
    # The substitutions were never priced in either run.
    assert dear_sub > 0 and free_sub > 0
