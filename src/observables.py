"""Per-sequence observables: energy, Hamming distance, P(S), context entropy.

All of them are batched over a set of sequences and work on CPU or GPU, so a
thousand chains are scored in one call.

The *context-dependent entropy* (CDE) of a sequence is the mean over sites of
the Shannon entropy of the model's single-site conditional at that site, given
that the rest of the sequence is kept fixed:

.. math::
    \\mathrm{CDE}(\\sigma) = \\frac1L \\sum_i H\\big[P(\\cdot \\mid \\sigma_{\\setminus i})\\big],
    \\qquad H[p] = -\\sum_a p_a \\log p_a .

A low CDE means the model considers the sequence's own context highly
constraining. The *sample quality* combines it with the energy,

.. math::
    Q_\\mu(\\sigma) = E_{\\mathrm{DCA}}(\\sigma) - \\mu\\, \\mathrm{CDE}(\\sigma),

and ``mu`` is calibrated on a natural alignment (see ``scripts/calibrate_mu.py``):
the value that makes ``Q`` as nearly constant as possible across natural
sequences is the one for which energy and context entropy trade off the way the
natural family does.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from adabmDCA.statmech import compute_energy

from potts import PottsModel

#: Sequences scored per chunk, to bound the memory of the all-site logits.
DEFAULT_CHUNK = 256


def one_hot_batch(model: PottsModel, sequences: torch.Tensor | np.ndarray) -> torch.Tensor:
    """``(N, L)`` encoded sequences -> ``(N, L, q)`` one-hot on the model's device."""
    encoded = torch.as_tensor(np.asarray(sequences) if not isinstance(sequences, torch.Tensor) else sequences)
    encoded = encoded.to(device=model.bias.device, dtype=torch.long)
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
    return torch.nn.functional.one_hot(encoded, num_classes=model.q).to(model.bias.dtype)


def energies(model: PottsModel, sequences: torch.Tensor | np.ndarray, chunk: int = DEFAULT_CHUNK) -> np.ndarray:
    """DCA energies (``adabmDCA.statmech.compute_energy``) of a batch."""
    out = []
    encoded = _as_batch(model, sequences)
    for start in range(0, encoded.shape[0], chunk):
        batch = one_hot_batch(model, encoded[start : start + chunk])
        out.append(compute_energy(batch, model.params).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def hamming_distance(sequences: torch.Tensor | np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Number of positions at which each sequence differs from ``reference``."""
    encoded = np.asarray(sequences.cpu() if isinstance(sequences, torch.Tensor) else sequences)
    if encoded.ndim == 1:
        encoded = encoded[None, :]
    return (encoded != np.asarray(reference)[None, :]).sum(axis=1).astype(np.int64)


def min_hamming_to_set(
    sequences: torch.Tensor | np.ndarray,
    reference_set: torch.Tensor | np.ndarray,
    device: str | torch.device = "cpu",
    exclude: np.ndarray | None = None,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each sequence to its nearest neighbour in a set of sequences.

    Returns ``(distance, index)``: for every sequence, the smallest Hamming
    distance to any member of ``reference_set`` and which member attains it.
    Agreements are counted with one matrix product per chunk -- the one-hot forms
    satisfy ``matches = x . y``, so ``distance = L - matches`` -- which is what
    makes a few thousand queries against a few thousand references immediate on a
    GPU.

    ``exclude`` gives, per query, one index into ``reference_set`` to ignore
    (``-1`` for none). It is what keeps a natural sequence from reporting a
    distance of zero to its own copy in the reference set.
    """
    queries = torch.as_tensor(np.asarray(sequences), dtype=torch.long, device=device)
    references = torch.as_tensor(np.asarray(reference_set), dtype=torch.long, device=device)
    if queries.ndim == 1:
        queries = queries.unsqueeze(0)
    if queries.shape[1] != references.shape[1]:
        raise ValueError(
            f"Sequences of length {queries.shape[1]} against references of length {references.shape[1]}"
        )
    if references.shape[0] == 0:
        raise ValueError("The reference set is empty")

    length = queries.shape[1]
    states = int(max(int(queries.max()), int(references.max())) + 1)
    reference_flat = (
        torch.nn.functional.one_hot(references, num_classes=states).to(torch.float32).reshape(references.shape[0], -1)
    )

    distances, indices = [], []
    for start in range(0, queries.shape[0], chunk):
        block = queries[start : start + chunk]
        flat = torch.nn.functional.one_hot(block, num_classes=states).to(torch.float32).reshape(block.shape[0], -1)
        matches = flat @ reference_flat.T  # (chunk, n_references)
        block_distance = length - matches
        if exclude is not None:
            skip = torch.as_tensor(np.asarray(exclude[start : start + chunk]), dtype=torch.long, device=device)
            rows = torch.arange(block.shape[0], device=device)
            valid = skip >= 0
            block_distance[rows[valid], skip[valid]] = float(length + 1)
        best = block_distance.min(dim=1)
        distances.append(best.values.round().to(torch.int64).cpu().numpy())
        indices.append(best.indices.cpu().numpy())
    return np.concatenate(distances), np.concatenate(indices)


def site_probabilities(
    model: PottsModel,
    sequences: torch.Tensor | np.ndarray,
    site: int,
    beta: float = 1.0,
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    """``P(sigma_site = . | rest)`` for every sequence, shape ``(N, q)``."""
    encoded = _as_batch(model, sequences)
    flat_couplings = model.couplings[site].reshape(model.q, model.L * model.q)
    out = []
    for start in range(0, encoded.shape[0], chunk):
        flat = one_hot_batch(model, encoded[start : start + chunk]).reshape(-1, model.L * model.q)
        logits = model.bias[site] + flat @ flat_couplings.T
        out.append(torch.softmax(beta * logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def critic_probability(
    model: PottsModel,
    sequences: torch.Tensor | np.ndarray,
    site: int,
    state: int,
    beta: float = 1.0,
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    """``P(sigma_site = state | rest)`` for every sequence."""
    return site_probabilities(model, sequences, site, beta=beta, chunk=chunk)[:, state]


def context_dependent_entropy(
    model: PottsModel,
    sequences: torch.Tensor | np.ndarray,
    beta: float = 1.0,
    chunk: int = DEFAULT_CHUNK,
    per_site: bool = False,
) -> np.ndarray:
    """Mean over sites of the conditional Shannon entropy (in nats).

    With ``per_site=True`` the full ``(N, L)`` matrix of site entropies is
    returned instead of its mean over sites.
    """
    encoded = _as_batch(model, sequences)
    L, q = model.L, model.q
    # J[i, :, i, :] is zero for an adabmDCA model, so the flat product already
    # excludes each site's own contribution to its logits.
    flat_couplings = model.couplings.reshape(L * q, L * q)
    out = []
    for start in range(0, encoded.shape[0], chunk):
        flat = one_hot_batch(model, encoded[start : start + chunk]).reshape(-1, L * q)
        logits = (model.bias.reshape(1, L * q) + flat @ flat_couplings.T).reshape(-1, L, q)
        log_p = torch.log_softmax(beta * logits, dim=2)
        entropy = -(log_p.exp() * log_p).sum(dim=2)  # (chunk, L)
        out.append((entropy if per_site else entropy.mean(dim=1)).cpu().numpy())
    return np.concatenate(out, axis=0)


def sample_quality(energy: np.ndarray, cde: np.ndarray, mu: float) -> np.ndarray:
    """``E_DCA - mu * CDE``."""
    return np.asarray(energy) - mu * np.asarray(cde)


def optimal_mu(energy: np.ndarray, cde: np.ndarray) -> float:
    """The ``mu`` minimising ``Var(E - mu * CDE)``: ``Cov(E, CDE) / Var(CDE)``."""
    energy = np.asarray(energy, dtype=float)
    cde = np.asarray(cde, dtype=float)
    variance = cde.var()
    if variance == 0.0:
        raise ValueError("The context-dependent entropy is constant; mu is undetermined")
    return float(np.cov(energy, cde, bias=True)[0, 1] / variance)


def describe(
    model: PottsModel,
    sequences: torch.Tensor | np.ndarray,
    reference: np.ndarray,
    critic_col: int,
    target_state: int,
    beta: float = 1.0,
    chunk: int = DEFAULT_CHUNK,
) -> Dict[str, np.ndarray]:
    """All four observables of a set of sequences, as a dict of 1-D arrays."""
    encoded = _as_batch(model, sequences)
    return {
        "E_DCA": energies(model, encoded, chunk=chunk),
        "hamming_to_wt": hamming_distance(encoded, reference),
        "P_S": critic_probability(model, encoded, critic_col, target_state, beta=beta, chunk=chunk),
        "CDE": context_dependent_entropy(model, encoded, beta=beta, chunk=chunk),
    }


def _as_batch(model: PottsModel, sequences: torch.Tensor | np.ndarray) -> torch.Tensor:
    encoded = torch.as_tensor(
        np.asarray(sequences) if not isinstance(sequences, torch.Tensor) else sequences
    ).to(device=model.bias.device, dtype=torch.long)
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
    if encoded.shape[1] != model.L:
        raise ValueError(f"Sequences have length {encoded.shape[1]}, the model expects {model.L}")
    return encoded
