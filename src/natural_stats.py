"""Reweighted statistics of the natural alignment, and a batch's agreement with them.

The generative check of a Potts model is whether sequences resampled from it
reproduce the statistics it was fitted on. Those statistics are the *reweighted*
one- and two-point frequencies of the training alignment -- each sequence carries
the weight adabmDCA assigned it, which downweights the dense clusters of close
homologues -- and the connected correlations

.. math::
    C_{ij}(a, b) = f_{ij}(a, b) - f_i(a) f_j(b), \\qquad i \\neq j .

Everything here delegates to adabmDCA: :func:`adabmDCA.stats.get_freq_single_point`
and :func:`~adabmDCA.stats.get_freq_two_points` for the frequencies,
:func:`~adabmDCA.stats.get_correlation_two_points` for the Pearson coefficient of
the connected correlations (which extracts the off-diagonal blocks
:math:`i \\neq j` itself, so the diagonal part never enters), and
:func:`~adabmDCA.stats.extract_Cij_from_freq` behind it.

Because the sampler never proposes a gap and never mutates a gap column, a chain
keeps the gap pattern of the sequence it started from. The gap state is therefore
one the chains cannot sample, and each quantity is reported twice: over all ``q``
states, and over the ``q - 1`` non-gap states alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from adabmDCA.stats import (
    extract_Cij_from_freq,
    get_correlation_two_points,
    get_freq_single_point,
    get_freq_two_points,
)

#: Sequences one-hot encoded per chunk when the alignment is too large to hold at once.
DEFAULT_CHUNK = 8192


def load_weights(path: str | Path, n_sequences: int) -> np.ndarray:
    """Read an adabmDCA weights file: one positive float per alignment row."""
    weights = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if weights.shape[0] != n_sequences:
        raise ValueError(
            f"{path} holds {weights.shape[0]} weights but the alignment has {n_sequences} sequences"
        )
    if not np.all(weights > 0):
        raise ValueError(f"{path} holds non-positive weights")
    return weights


def one_hot(encoded: torch.Tensor, q: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.nn.functional.one_hot(encoded.long(), num_classes=q).to(dtype)


@dataclass
class NaturalStatistics:
    """Reweighted ``f_i`` and ``f_ij`` of the natural alignment, and the masks."""

    fi: torch.Tensor  # (L, q)
    fij: torch.Tensor  # (L, q, L, q)
    gap_index: int
    n_sequences: int
    effective_sequences: float  # sum of the weights, the DCA M_eff
    kish_ess: float  # (sum w)^2 / sum w^2, a different and larger number
    pseudo_count: float

    @property
    def device(self) -> torch.device:
        return self.fi.device

    @classmethod
    def from_alignment(
        cls,
        encoded: np.ndarray,
        q: int,
        gap_index: int,
        weights: Optional[np.ndarray] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        pseudo_count: float = 0.0,
        chunk: int = DEFAULT_CHUNK,
    ) -> "NaturalStatistics":
        """Weighted frequencies of an encoded alignment, accumulated in chunks.

        The chunks are combined as the weighted average they are, which is exact
        for ``pseudo_count = 0``; a non-zero pseudo count is applied once at the
        end instead, so that it means what it means in adabmDCA.
        """
        encoded_t = torch.as_tensor(np.asarray(encoded), dtype=torch.long)
        n, length = encoded_t.shape
        w = torch.ones(n, dtype=torch.float64) if weights is None else torch.as_tensor(weights)
        total = float(w.sum())

        fi = torch.zeros((length, q), device=device, dtype=dtype)
        fij = torch.zeros((length, q, length, q), device=device, dtype=dtype)
        for start in range(0, n, chunk):
            block = one_hot(encoded_t[start : start + chunk].to(device), q, dtype)
            block_w = w[start : start + chunk].to(device=device, dtype=dtype)
            share = float(block_w.sum()) / total
            fi += share * get_freq_single_point(block, weights=block_w, pseudo_count=0.0)
            fij += share * get_freq_two_points(block, weights=block_w, pseudo_count=0.0)
            del block

        if pseudo_count:
            fi = (1.0 - pseudo_count) * fi + pseudo_count / q
            fij = (1.0 - pseudo_count) * fij + pseudo_count / (q**2)

        return cls(
            fi=fi,
            fij=fij,
            gap_index=gap_index,
            n_sequences=n,
            # M_eff in the DCA sense is the sum of the weights, not Kish's
            # effective sample size; with 1/n_neighbours weights the two differ by
            # a third here, so both are kept and named.
            effective_sequences=float(w.sum()),
            kish_ess=float(w.sum() ** 2 / (w**2).sum()),
            pseudo_count=pseudo_count,
        )

    def non_gap_mask(self) -> torch.Tensor:
        """``(L, q, L, q)`` mask excluding every entry that involves the gap state."""
        length, q = self.fi.shape
        keep = torch.ones(q, dtype=torch.bool, device=self.device)
        keep[self.gap_index] = False
        return keep.view(1, q, 1, 1) & keep.view(1, 1, 1, q) & torch.ones(
            (length, 1, length, 1), dtype=torch.bool, device=self.device
        )

    def compare(self, chains: torch.Tensor, chain_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Pearson coefficients of a batch's statistics against the natural ones.

        ``chains`` is one-hot, ``(M, L, q)``. The keys are ``pearson_f1``,
        ``pearson_f2``, ``pearson_cij`` and ``slope_cij``, each also in a
        ``_nogap`` variant computed over the non-gap states alone.
        """
        if chains.dim() != 3:
            raise ValueError(f"Expected a one-hot (M, L, q) batch, got {tuple(chains.shape)}")
        pi = get_freq_single_point(chains, weights=chain_weights, pseudo_count=0.0)
        pij = get_freq_two_points(chains, weights=chain_weights, pseudo_count=0.0)

        out: Dict[str, float] = {"pearson_f1": _pearson(self.fi.reshape(-1), pi.reshape(-1))}
        pearson, slope = get_correlation_two_points(fij=self.fij, pij=pij, fi=self.fi, pi=pi)
        out["pearson_cij"] = pearson
        out["slope_cij"] = slope
        out["pearson_f2"] = _pearson(*_off_diagonal(self.fij, pij))

        keep = torch.ones(self.fi.shape[1], dtype=torch.bool, device=self.device)
        keep[self.gap_index] = False
        out["pearson_f1_nogap"] = _pearson(self.fi[:, keep].reshape(-1), pi[:, keep].reshape(-1))
        mask = self.non_gap_mask()
        pearson, slope = get_correlation_two_points(fij=self.fij, pij=pij, fi=self.fi, pi=pi, mask=mask)
        out["pearson_cij_nogap"] = pearson
        out["slope_cij_nogap"] = slope
        out["pearson_f2_nogap"] = _pearson(*_off_diagonal(self.fij, pij, mask=mask))
        return out


def _off_diagonal(fij: torch.Tensor, pij: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """The ``i != j`` blocks of two frequency tensors, flattened and matched.

    This is what :func:`adabmDCA.stats.extract_Cij_from_freq` does to the
    *connected* correlations, applied to the raw two-point frequencies, so that
    the trivial diagonal blocks ``f_ii(a, b) = f_i(a) delta_ab`` -- which every
    sampler reproduces by construction -- stay out of the comparison.
    """
    length = fij.shape[0]
    rows, columns = torch.tril_indices(length, length, offset=-1)
    first, second = fij[rows, :, columns, :], pij[rows, :, columns, :]
    if mask is not None:
        keep = mask[rows, :, columns, :]
        return first[keep].reshape(-1), second[keep].reshape(-1)
    return first.reshape(-1), second.reshape(-1)


def _pearson(first: torch.Tensor, second: torch.Tensor) -> float:
    stacked = torch.stack([first.float().reshape(-1), second.float().reshape(-1)])
    return float(torch.corrcoef(stacked)[0, 1])


def weighted_mean(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Mean of a per-sequence observable under the alignment's weights."""
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(values.mean())
    weights = np.asarray(weights, dtype=np.float64)
    return float((values * weights).sum() / weights.sum())


def weighted_std(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(values.std())
    weights = np.asarray(weights, dtype=np.float64)
    mean = weighted_mean(values, weights)
    return float(np.sqrt((weights * (values - mean) ** 2).sum() / weights.sum()))


__all__ = [
    "NaturalStatistics",
    "load_weights",
    "one_hot",
    "weighted_mean",
    "weighted_std",
]
