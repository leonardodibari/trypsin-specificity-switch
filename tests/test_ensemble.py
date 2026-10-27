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


def exact_distribution(model, start, critic_col, sites, lam, beta, target_state):
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
        states.append(tuple(int(s) for s in combination))
        weights.append(-beta * energy + lam * log_p)
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


def test_the_trace_follows_the_batch(model, start):
    result = run_ensemble(
        model,
        start,
        critic_col=8,
        allowed_cols=[3, 4, 5],
        config=EnsembleConfig(lam=1.0, n_chains=16, n_steps=50, seed=2, trace_every=10),
    )
    assert result.trace["step"].tolist() == [10, 20, 30, 40, 50]
    assert np.all((result.trace["P_S"] >= 0) & (result.trace["P_S"] <= 1))
    assert torch.is_tensor(torch.as_tensor(result.trace["E_DCA"]))
