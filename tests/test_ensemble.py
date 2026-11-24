"""The batched sampler must reproduce the distribution it claims to sample."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from ensemble import EnsembleConfig, run_ensemble
from observables import (
    context_dependent_entropy,
    critic_probability,
    energies,
    hamming_distance,
    optimal_mu,
    sample_quality,
)


@pytest.fixture
def start(model, alphabet) -> np.ndarray:
    rng = np.random.default_rng(2)
    sequence = rng.choice(alphabet.residue_indices, size=model.L)
    return sequence.astype(np.int64)


def exact_distribution(model, start, critic_col, sites, lam, beta, target_state, t=0.0,
                       reference=None):
    """Enumerate the sampled space and return (states, probabilities).

    The chain only ever touches ``sites``, so the reachable space is the product
    of the non-gap residues over those sites: small enough to enumerate for two
    sites, which makes the sampled distribution checkable exactly.
    """
    residues = model.alphabet.residue_indices
    states, weights = [], []
    for combination in itertools.product(residues, repeat=len(sites)):
        sequence = np.array(start, copy=True)
        for site, state in zip(sites, combination):
            sequence[site] = state
        energy = model.energy(sequence)
        log_p = model.log_conditional(critic_col, sequence, target_state, beta=beta)
        distance = int(np.sum(sequence != (start if reference is None else reference)))
        states.append(tuple(int(s) for s in combination))
        weights.append(-beta * energy + lam * log_p - t * distance)
    weights = np.asarray(weights)
    weights = np.exp(weights - weights.max())
    return states, weights / weights.sum()


@pytest.mark.parametrize("lam", [0.0, 3.0])
def test_the_ensemble_samples_the_biased_boltzmann_distribution(model, start, alphabet, lam):
    # One mutable site: 20 reachable states, which a few thousand chains pin
    # down tightly enough to compare against the exact weights.
    sites = [4]
    critic_col = 8
    beta = 1.0
    target_state = alphabet.index("S")

    states, expected = exact_distribution(model, start, critic_col, sites, lam, beta, target_state)
    index_of = {state: i for i, state in enumerate(states)}

    result = run_ensemble(
        model,
        start,
        critic_col=critic_col,
        allowed_cols=sites,
        config=EnsembleConfig(lam=lam, beta=beta, n_chains=4000, n_steps=600, seed=11),
    )
    counts = np.zeros(len(states))
    for sequence in result.sequences:
        counts[index_of[tuple(int(sequence[site]) for site in sites)]] += 1
    observed = counts / counts.sum()

    # Multinomial noise alone puts the total variation around 0.02 here.
    total_variation = 0.5 * np.abs(observed - expected).sum()
    assert total_variation < 0.06, f"TV = {total_variation:.3f}"
    assert np.corrcoef(observed, expected)[0, 1] > 0.98


def test_only_the_allowed_sites_move(model, start):
    result = run_ensemble(
        model,
        start,
        critic_col=8,
        allowed_cols=[3, 5],
        config=EnsembleConfig(lam=1.0, n_chains=64, n_steps=200, seed=3),
    )
    changed = np.flatnonzero((result.sequences != start[None, :]).any(axis=0))
    assert set(changed.tolist()) <= {3, 5}
    assert result.sequences.shape == (64, model.L)
    assert 0.0 < result.acceptance_rate <= 1.0


def test_the_same_seed_reproduces_the_batch(model, start):
    config = EnsembleConfig(lam=2.0, n_chains=32, n_steps=100, seed=7)
    first = run_ensemble(model, start, 8, [3, 4, 5], config)
    second = run_ensemble(model, start, 8, [3, 4, 5], config)
    assert np.array_equal(first.sequences, second.sequences)


def test_a_stronger_bias_raises_the_critic_probability(model, start, alphabet):
    state = alphabet.index("S")
    probabilities = []
    for lam in (0.0, 10.0):
        result = run_ensemble(
            model,
            start,
            critic_col=8,
            allowed_cols=[3, 4, 5, 9],
            config=EnsembleConfig(lam=lam, n_chains=256, n_steps=300, seed=5),
        )
        probabilities.append(critic_probability(model, result.sequences, 8, state).mean())
    assert probabilities[1] > probabilities[0]


def test_batched_observables_match_the_single_sequence_versions(model, start, alphabet):
    rng = np.random.default_rng(8)
    batch = np.stack([rng.choice(alphabet.residue_indices, size=model.L) for _ in range(5)])
    state = alphabet.index("S")

    assert energies(model, batch) == pytest.approx([model.energy(s) for s in batch], abs=1e-4)
    assert critic_probability(model, batch, 6, state) == pytest.approx(
        [model.conditional(6, s)[state] for s in batch], abs=1e-5
    )

    expected_cde = []
    for sequence in batch:
        entropies = []
        for site in range(model.L):
            p = model.conditional(site, sequence)
            entropies.append(float(-(p * np.log(np.clip(p, 1e-30, None))).sum()))
        expected_cde.append(np.mean(entropies))
    assert context_dependent_entropy(model, batch) == pytest.approx(expected_cde, abs=1e-5)

    assert hamming_distance(batch, batch[0]).tolist()[0] == 0
    assert hamming_distance(batch, start).max() <= model.L


def test_optimal_mu_minimises_the_variance_of_the_sample_quality():
    rng = np.random.default_rng(1)
    cde = rng.normal(1.0, 0.2, size=500)
    energy = 3.5 * cde + rng.normal(0.0, 0.05, size=500)
    mu = optimal_mu(energy, cde)
    assert mu == pytest.approx(3.5, abs=0.1)
    scan = [sample_quality(energy, cde, m).var() for m in (mu - 0.3, mu, mu + 0.3)]
    assert scan[1] < scan[0] and scan[1] < scan[2]


def test_the_log_trace_schedule_is_dense_early(model, start):
    result = run_ensemble(
        model,
        start,
        critic_col=8,
        allowed_cols=[3, 4, 5],
        config=EnsembleConfig(
            lam=1.0, n_chains=16, n_steps=1000, seed=2,
            trace_every=100, trace_schedule="log", trace_cde=False,
        ),
    )
    steps = result.trace["step"]
    assert steps[0] == 1 and steps[-1] == 1000
    assert np.all(np.diff(steps) > 0)
    # Log spacing puts more than half the points in the first tenth of the run.
    assert (steps <= 100).sum() > len(steps) / 2
    assert "CDE_mean" not in result.trace  # switched off


def test_the_trace_follows_the_batch(model, start):
    result = run_ensemble(
        model,
        start,
        critic_col=8,
        allowed_cols=[3, 4, 5],
        config=EnsembleConfig(lam=1.0, n_chains=16, n_steps=50, seed=2, trace_every=10),
    )
    assert result.trace["step"].tolist() == [1, 10, 20, 30, 40, 50]
    assert np.all((result.trace["P_S_mean"] >= 0) & (result.trace["P_S_mean"] <= 1))
    assert np.all(result.trace["P_S_q25"] <= result.trace["P_S_q75"])
    assert np.all(np.diff(result.trace["accepted_per_chain"]) >= 0)
    for name in ("E_DCA", "hamming_to_wt", "P_S", "CDE"):
        assert f"{name}_mean" in result.trace
    # The traced energy must be the energy of the traced sequences.
    assert result.trace["E_DCA_mean"][-1] == pytest.approx(
        float(np.mean(energies(model, result.sequences))), abs=1e-3
    )
    assert result.trace["hamming_to_wt_mean"][-1] == pytest.approx(
        float(np.mean(hamming_distance(result.sequences, start))), abs=1e-6
    )


# --- the Hamming restraint ------------------------------------------------------
# H carries + t * d_H(sigma, WT): t > 0 holds the chains at the wild type, t < 0
# pays them to leave it. d_H is a sum of single-site indicators, so this is a
# uniform field of +t on every non-wild-type state and the sampled distribution
# is still exactly a Potts model -- which is what makes the enumeration below a
# complete check rather than a smoke test.


@pytest.mark.parametrize("t", [-1.0, 0.5, 2.0])
def test_the_restrained_ensemble_samples_the_restrained_boltzmann_distribution(
    model, start, alphabet, t
):
    """The strongest check available: enumerate the reachable space exactly."""
    sites = [4]
    critic_col = 8
    beta = 1.0
    lam = 2.0
    target_state = alphabet.index("S")

    states, expected = exact_distribution(
        model, start, critic_col, sites, lam, beta, target_state, t=t
    )
    index_of = {state: i for i, state in enumerate(states)}

    result = run_ensemble(
        model,
        start,
        critic_col=critic_col,
        allowed_cols=sites,
        config=EnsembleConfig(lam=lam, beta=beta, t=t, n_chains=4000, n_steps=600, seed=11),
    )
    counts = np.zeros(len(states))
    for sequence in result.sequences:
        counts[index_of[tuple(int(sequence[site]) for site in sites)]] += 1
    observed = counts / counts.sum()

    total_variation = 0.5 * np.abs(observed - expected).sum()
    assert total_variation < 0.06, f"TV = {total_variation:.3f}"
    assert np.corrcoef(observed, expected)[0, 1] > 0.98


def test_t_zero_changes_nothing(model, start):
    """t = 0 must reproduce the unrestrained chain move for move, not merely in
    distribution: every result sampled before the restraint existed still stands."""
    plain = EnsembleConfig(lam=2.0, beta=1.0, n_chains=64, n_steps=400, seed=19)
    restrained = EnsembleConfig(lam=2.0, beta=1.0, t=0.0, n_chains=64, n_steps=400, seed=19)
    first = run_ensemble(model, start, 8, [3, 4, 5], plain)
    second = run_ensemble(model, start, 8, [3, 4, 5], restrained)
    assert np.array_equal(first.sequences, second.sequences)
    assert first.acceptance_rate == second.acceptance_rate


def test_the_mean_distance_decreases_monotonically_in_t(model, start):
    """<d>(t) is monotone decreasing -- it is minus the derivative of a free energy
    in a conjugate field, so this is an identity, and a violation at equilibrium
    means the chains have not equilibrated."""
    means = []
    for t in (-0.5, 0.0, 0.5, 1.5, 4.0):
        result = run_ensemble(
            model, start, critic_col=8, allowed_cols=[3, 4, 5, 9, 10],
            config=EnsembleConfig(lam=1.0, t=t, n_chains=2000, n_steps=1500, seed=4),
        )
        means.append(float(hamming_distance(result.sequences, start).mean()))
    assert all(a >= b - 0.05 for a, b in zip(means, means[1:])), means
    # A large positive t pins the chains at the wild type outright.
    assert means[-1] < 0.5 * means[0]


def test_the_variance_of_the_distance_is_the_susceptibility(model, start):
    """-d<d>/dt = Var(d): the fluctuation-response identity the t scan rests on.

    Checked by finite difference against the variance at the midpoint, which is
    also the sanity check that the restraint enters the Hamiltonian with the
    right sign and magnitude.
    """
    def statistics(t):
        result = run_ensemble(
            model, start, critic_col=8, allowed_cols=[3, 4, 5, 9, 10],
            config=EnsembleConfig(lam=1.0, t=t, n_chains=8000, n_steps=2500, seed=21),
        )
        distances = hamming_distance(result.sequences, start).astype(float)
        return distances.mean(), distances.var()

    step = 0.25
    mean_low, _ = statistics(-step)
    mean_high, _ = statistics(step)
    _, variance = statistics(0.0)
    slope = -(mean_high - mean_low) / (2 * step)
    assert slope == pytest.approx(variance, rel=0.25), (slope, variance)


def test_the_trace_carries_every_observable_a_result_is_read_off(model, start):
    """Saturation has to be checkable on each of them, not inferred from one."""
    result = run_ensemble(
        model, start, critic_col=8, allowed_cols=[3, 4, 5],
        config=EnsembleConfig(lam=1.0, t=0.3, mu=2.5, n_chains=32, n_steps=100, seed=2,
                              trace_every=20),
    )
    for name in ("E_DCA", "hamming_to_wt", "indel_to_wt", "P_S", "LOR", "CDE", "Q"):
        assert f"{name}_mean" in result.trace, name
        assert f"{name}_q25" in result.trace and f"{name}_q75" in result.trace
    # Q is E - mu * CDE on the same chains, not a separately scored quantity.
    assert result.trace["Q_mean"][-1] == pytest.approx(
        result.trace["E_DCA_mean"][-1] - 2.5 * result.trace["CDE_mean"][-1], abs=1e-4
    )


# --- the restraint split by move type -------------------------------------------
# d_H counts substitutions and indels together, so with allow_gap_moves a single t
# prices a gained or lost residue at the substitution price without saying so.
# t_sub and t_indel separate the two. Each is still a single-site indicator, so the
# sum is still exactly a field and the enumeration below is still complete: at a
# site whose reference residue is r the shift is 0 on r, t_indel on the states that
# are a gap-vs-residue difference, and t_sub on the rest.


def split_exact_distribution(
    model, start, critic_col, sites, lam, beta, target_state,
    t_sub=0.0, t_indel=0.0, reference=None, allow_gap_moves=True,
):
    """``exact_distribution`` with the two prices separated and gaps reachable.

    The helper above enumerates non-gap residues only, so it cannot see an indel
    at all; this one includes the gap, which is the whole point of the split.
    """
    gap = model.alphabet.gap_index
    reference = start if reference is None else reference
    pool = list(range(model.q)) if allow_gap_moves else list(model.alphabet.residue_indices)
    states, weights = [], []
    for combination in itertools.product(pool, repeat=len(sites)):
        sequence = np.array(start, copy=True)
        for site, state in zip(sites, combination):
            sequence[site] = state
        differs = sequence[sites] != reference[sites]
        indel = differs & ((sequence[sites] == gap) | (reference[sites] == gap))
        d_sub = int((differs & ~indel).sum())
        d_indel = int(indel.sum())
        energy = model.energy(sequence)
        log_p = model.log_conditional(critic_col, sequence, target_state, beta=beta)
        states.append(tuple(int(s) for s in combination))
        weights.append(-beta * energy + lam * log_p - t_sub * d_sub - t_indel * d_indel)
    weights = np.asarray(weights)
    weights = np.exp(weights - weights.max())
    return states, weights / weights.sum()


def test_t_sub_and_t_indel_fall_back_to_t(model, start):
    """Unset, each follows t -- so every run sampled under the single knob still
    reproduces move for move, not merely in distribution."""
    single = EnsembleConfig(lam=1.0, t=0.7, n_chains=64, n_steps=400, seed=23,
                            allow_gap_moves=True)
    split = EnsembleConfig(lam=1.0, t=0.7, t_sub=0.7, t_indel=0.7, n_chains=64,
                           n_steps=400, seed=23, allow_gap_moves=True)
    assert single.t_sub_effective == single.t_indel_effective == 0.7
    first = run_ensemble(model, start, 8, [3, 4, 5], single)
    second = run_ensemble(model, start, 8, [3, 4, 5], split)
    assert np.array_equal(first.sequences, second.sequences)
    assert first.acceptance_rate == second.acceptance_rate


@pytest.mark.parametrize("t_sub, t_indel", [(1.0, -1.0), (-0.5, 2.0), (0.0, 1.5)])
def test_the_split_restraint_samples_the_split_boltzmann_distribution(
    model, start, alphabet, t_sub, t_indel
):
    """Enumerate the reachable space exactly, gap included.

    The two prices are given opposite signs on purpose: a single t cannot produce
    these distributions at all, so this fails outright if the terms are summed
    into one counter anywhere along the way.
    """
    sites = [4]
    critic_col, beta, lam = 8, 1.0, 2.0
    target_state = alphabet.index("S")

    states, expected = split_exact_distribution(
        model, start, critic_col, sites, lam, beta, target_state,
        t_sub=t_sub, t_indel=t_indel,
    )
    index_of = {state: i for i, state in enumerate(states)}

    result = run_ensemble(
        model, start, critic_col=critic_col, allowed_cols=sites,
        config=EnsembleConfig(lam=lam, beta=beta, t_sub=t_sub, t_indel=t_indel,
                              allow_gap_moves=True, n_chains=4000, n_steps=600, seed=11),
    )
    counts = np.zeros(len(states))
    for sequence in result.sequences:
        counts[index_of[tuple(int(sequence[site]) for site in sites)]] += 1
    observed = counts / counts.sum()

    total_variation = 0.5 * np.abs(observed - expected).sum()
    assert total_variation < 0.06, f"TV = {total_variation:.3f}"
    assert np.corrcoef(observed, expected)[0, 1] > 0.98


def test_t_indel_prices_indels_without_touching_the_substitution_count(model, alphabet):
    """The separation is the point: raising t_indel alone must suppress gaps and
    leave the substitution part of d where it was."""
    rng = np.random.default_rng(5)
    start = rng.choice(alphabet.residue_indices, size=model.L).astype(np.int64)
    sites = [3, 4, 5, 9, 10]

    def counts(t_indel):
        result = run_ensemble(
            model, start, critic_col=8, allowed_cols=sites,
            config=EnsembleConfig(lam=1.0, t_sub=0.0, t_indel=t_indel, allow_gap_moves=True,
                                  n_chains=3000, n_steps=3000, seed=31, trace_every=3000),
        )
        return (result.trace["indel_to_wt_mean"][-1], result.trace["sub_to_wt_mean"][-1])

    free_indel, free_sub = counts(-1.0)
    dear_indel, dear_sub = counts(6.0)
    # A large indel price empties the gap states...
    assert dear_indel < 0.1 * free_indel, (free_indel, dear_indel)
    assert dear_indel < 0.05
    # ...while the substitutions, priced at t_sub = 0 throughout, stay put. They
    # are not identical -- removing the gap frees weight onto the residues -- but
    # they cannot collapse the way the indels did.
    assert dear_sub > 0.8 * free_sub, (free_sub, dear_sub)


def test_the_split_restraint_matches_an_independent_site_model_analytically(
    tmp_path, alphabet
):
    """On a model with no couplings the sites factorise and <d_sub>, <d_indel> are
    closed-form sums of single-site softmaxes -- so the sampler is checked against
    a formula, over many sites at once, rather than against an enumeration of one.
    """
    from potts import PottsModel

    L, q = 12, alphabet.q
    rng = np.random.default_rng(13)
    bias = rng.normal(0.0, 0.8, size=(L, q))
    path = tmp_path / "independent.dat"
    with open(path, "w", encoding="utf-8") as handle:
        # Every J is zero: the sites are independent and the energy is -sum_i h_i.
        for i in range(L):
            for j in range(i + 1, L):
                for token_a in alphabet.tokens:
                    for token_b in alphabet.tokens:
                        handle.write(f"J {i} {j} {token_a} {token_b} 0.000000\n")
        for i in range(L):
            for a, token_a in enumerate(alphabet.tokens):
                handle.write(f"h {i} {token_a} {bias[i, a]:.6f}\n")
    model = PottsModel.load(path, alphabet=alphabet, dtype="float64")

    gap = alphabet.gap_index
    start = rng.choice(alphabet.residue_indices, size=L).astype(np.int64)
    start[7] = gap  # one reference gap, so t_indel prices a RESTORED residue too
    sites = [1, 2, 3, 7, 9]
    beta, t_sub, t_indel = 1.0, 0.4, -0.8

    # <d> = sum over sites of the single-site probability of being away from the
    # reference, each site's distribution being softmax(beta*h - the two prices).
    expected_sub = expected_indel = 0.0
    for site in sites:
        reference_state = int(start[site])
        logits = np.empty(q)
        for state in range(q):
            is_indel = state != reference_state and (state == gap or reference_state == gap)
            is_sub = state != reference_state and not is_indel
            logits[state] = beta * bias[site, state] - t_sub * is_sub - t_indel * is_indel
        probability = np.exp(logits - logits.max())
        probability /= probability.sum()
        for state in range(q):
            if state == reference_state:
                continue
            if state == gap or reference_state == gap:
                expected_indel += probability[state]
            else:
                expected_sub += probability[state]

    result = run_ensemble(
        model, start, critic_col=0, allowed_cols=sites,
        config=EnsembleConfig(lam=0.0, beta=beta, t_sub=t_sub, t_indel=t_indel,
                              allow_gap_moves=True, n_chains=8000, n_steps=6000,
                              seed=17, trace_every=6000),
    )
    assert result.trace["sub_to_wt_mean"][-1] == pytest.approx(expected_sub, abs=0.06)
    assert result.trace["indel_to_wt_mean"][-1] == pytest.approx(expected_indel, abs=0.06)


def test_restricting_d_H_to_the_allowed_columns_only_shifts_H_by_a_constant(
    model, alphabet
):
    """d_H is defined over the allowed columns, not over all L. The two differ only
    where the reference differs from the start OUTSIDE the allowed set -- frozen
    columns, which no move can touch -- so the difference is a constant in H and
    cancels in every Delta H. The sampled distribution is therefore identical; what
    changes is only that the reported d no longer carries a frozen offset.

    This is the case the bracket runs would hit if they were not run over all 265
    columns: a natural-chymotrypsin start against a wild-type reference.
    """
    rng = np.random.default_rng(41)
    start = rng.choice(alphabet.residue_indices, size=model.L).astype(np.int64)
    reference = rng.choice(alphabet.residue_indices, size=model.L).astype(np.int64)
    sites = [3, 4, 5]
    frozen_differences = int(
        np.sum(start[[c for c in range(model.L) if c not in sites]]
               != reference[[c for c in range(model.L) if c not in sites]])
    )
    assert frozen_differences > 0, "the case under test needs a frozen difference"

    config = EnsembleConfig(lam=1.0, t_sub=0.7, t_indel=0.7, n_chains=256, n_steps=900,
                            seed=29, allow_gap_moves=True, trace_every=900)
    restricted = run_ensemble(model, start, 8, sites, config, reference_sequence=reference)

    # Independently: the same chains sampled with d_H taken over all L, which is
    # the restricted d_H plus that constant. Reproduced here by sampling the
    # restricted term and checking the offset is exactly the frozen difference.
    traced = restricted.trace["hamming_to_wt_mean"][-1]
    exact = float(np.mean((restricted.sequences != reference[None, :]).sum(axis=1)))
    assert traced == pytest.approx(exact, abs=1e-6)
    # The all-L distance exceeds the restrained one by the frozen offset alone.
    inside = float(np.mean((restricted.sequences[:, sites] != reference[sites][None, :]).sum(axis=1)))
    assert exact == pytest.approx(inside + frozen_differences, abs=1e-6)
