"""The bmDCA model: adabmDCA 2.1 loading, energies, conditionals, mutation dE.

The model is loaded with :func:`adabmDCA.load_model`, which returns a
:class:`adabmDCA.api.model.DCAModel` holding

* ``bias``            of shape ``(L, q)``
* ``coupling_matrix`` of shape ``(L, q, L, q)``

and energies are computed with :func:`adabmDCA.statmech.compute_energy`, i.e.

.. math::
    E(\\sigma) = -\\sum_i h_i(\\sigma_i) - \\tfrac12 \\sum_{i,j} J_{ij}(\\sigma_i, \\sigma_j).

The single-site term uses adabmDCA's own Gibbs expression (see
``adabmDCA.sampling.gibbs_step_uniform_sites``)::

    logits_i = bias[i] + couplings[i].reshape(q, L * q) @ x.flatten()

With the symmetric, zero-diagonal couplings adabmDCA writes, those logits are at
once the conditional's logits -- ``P(sigma_i | rest) = softmax(beta * logits_i)``
-- and the negative substitution energy, ``E(... -> b) - E(... -> a) =
-(logits_i[b] - logits_i[a])``. Both identities are checked in
``tests/test_model.py`` against recomputed ``compute_energy`` values, and the
zero diagonal is verified when the model is loaded.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from adabmDCA import load_model
from adabmDCA.api.model import DCAModel
from adabmDCA.functional import one_hot
from adabmDCA.statmech import compute_energy

from encoding import Alphabet

#: File names tried, in order, when ``--model`` points at a directory.
MODEL_FILE_PATTERNS = ("*params*.dat", "params.dat", "*params*", "*.dat", "*.txt", "*.par")

#: Suffix of the optional cache written next to a parameter file.
CACHE_SUFFIX = ".cache.pt"


def resolve_model_path(path: str | Path) -> Path:
    """Return the parameter file for ``path``, which may be a file or a directory."""
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"No such model file or directory: {path}")
    for pattern in MODEL_FILE_PATTERNS:
        candidates = sorted(p for p in path.glob(pattern) if p.is_file() and not p.name.endswith(CACHE_SUFFIX))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"{path} holds several candidate parameter files {[p.name for p in candidates]}; "
                "pass the file itself to --model"
            )
    raise FileNotFoundError(f"No adabmDCA parameter file found in {path}")


class PottsModel:
    """An adabmDCA :class:`DCAModel` plus the per-site quantities the sampler needs."""

    def __init__(self, dca: DCAModel, alphabet: Alphabet | None = None) -> None:
        self.dca = dca
        self.params: Dict[str, torch.Tensor] = dca.params
        self.bias = self.params["bias"]
        self.couplings = self.params["coupling_matrix"]
        self.alphabet = alphabet or Alphabet(tokens=dca.tokens)
        if self.bias.shape[1] != self.alphabet.q:
            raise ValueError(
                f"The model has q={self.bias.shape[1]} states but the alphabet "
                f"{self.alphabet.tokens!r} has {self.alphabet.q}"
            )
        self.L, self.q = int(self.bias.shape[0]), int(self.bias.shape[1])
        self.source = dca.source
        self._flat_couplings = self.couplings.reshape(self.L, self.q, self.L * self.q)
        self._check_zero_diagonal()

    # ------------------------------------------------------------------ load

    @classmethod
    def load(
        cls,
        path: str | Path,
        alphabet: Alphabet | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        cache: bool = False,
        verbose: bool = False,
    ) -> "PottsModel":
        """Load adabmDCA 2.1 parameters from a file or a model directory.

        Parsing the text parameter file of a real model takes about half a
        minute; ``cache=True`` keeps a ``<file>.cache.pt`` tensor dump beside it
        and reuses it while it is newer than the text file.
        """
        alphabet = alphabet or Alphabet()
        model_file = resolve_model_path(path)
        cache_file = model_file.with_name(model_file.name + CACHE_SUFFIX)
        started = time.time()

        params = _read_cache(cache_file, model_file, alphabet, device, dtype) if cache else None
        if params is not None:
            dca = DCAModel(params, alphabet=alphabet.tokens, source=model_file)
        else:
            dca = load_model(model_file, alphabet=alphabet.tokens, device=device, dtype=dtype)
            if cache:
                _write_cache(cache_file, dca.params, alphabet, dtype)
        if verbose:
            print(f"model loaded in {time.time() - started:.1f}s from {model_file}")
        return cls(dca, alphabet=alphabet)

    def _check_zero_diagonal(self) -> None:
        diagonal = max(
            float(self.couplings[i, :, i, :].abs().max()) for i in range(self.L)
        )
        if diagonal > 1e-6:
            raise ValueError(
                "The model has non-zero self-couplings J[i, :, i, :] "
                f"(max |J| = {diagonal:.3g}). The conditional and the substitution "
                "energy of this code assume the zero diagonal adabmDCA writes."
            )

    # ---------------------------------------------------------------- energy

    def _as_tensor(self, sequence: Sequence[int] | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(sequence, torch.Tensor):
            tensor = sequence.to(dtype=torch.long, device=self.bias.device)
        else:
            tensor = torch.as_tensor(np.asarray(sequence), dtype=torch.long, device=self.bias.device)
        if tensor.shape[-1] != self.L:
            raise ValueError(f"Sequence has length {tensor.shape[-1]}, the model expects {self.L}")
        return tensor

    def one_hot(self, sequence: Sequence[int] | np.ndarray | torch.Tensor) -> torch.Tensor:
        """One-hot encode an encoded sequence (``adabmDCA.functional.one_hot``)."""
        return one_hot(self._as_tensor(sequence), num_classes=self.q, dtype=self.bias.dtype)

    def energy(self, sequence: Sequence[int] | np.ndarray | torch.Tensor) -> float:
        """DCA statistical energy of one encoded sequence."""
        return float(compute_energy(self.one_hot(sequence).unsqueeze(0), self.params)[0])

    def energies(self, sequences: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        """Energies of a batch of encoded sequences."""
        batch = torch.stack([self.one_hot(sequence) for sequence in sequences])
        return compute_energy(batch, self.params).cpu().numpy()

    def site_logits(
        self,
        site: int,
        sequence: Sequence[int] | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """``bias[site] + couplings[site] . x`` -- adabmDCA's Gibbs logits at ``site``."""
        flat = self.one_hot(sequence).reshape(-1)
        return self.bias[site] + self._flat_couplings[site] @ flat

    def conditional(
        self,
        site: int,
        sequence: Sequence[int] | np.ndarray | torch.Tensor,
        beta: float = 1.0,
    ) -> np.ndarray:
        """``P(sigma_site | sigma_{\\setminus site})`` over the whole alphabet."""
        return torch.softmax(beta * self.site_logits(site, sequence), dim=0).cpu().numpy()

    def log_conditional(
        self,
        site: int,
        sequence: Sequence[int] | np.ndarray | torch.Tensor,
        state: int,
        beta: float = 1.0,
    ) -> float:
        """``log P(sigma_site = state | sigma_{\\setminus site})``, computed stably."""
        return log_conditional_from_logits(self.site_logits(site, sequence), state, beta)

    def delta_energy(
        self,
        sequence: Sequence[int] | np.ndarray | torch.Tensor,
        site: int,
        new_state: int,
    ) -> float:
        """Energy change of the single substitution ``site: s_site -> new_state``."""
        encoded = self._as_tensor(sequence)
        old_state = int(encoded[site])
        if old_state == new_state:
            return 0.0
        logits = self.site_logits(site, encoded)
        return float(-(logits[new_state] - logits[old_state]))

    def coupling_column(self, site: int, other: int, state: int) -> torch.Tensor:
        """``J[site, :, other, state]`` -- used to update ``site_logits`` incrementally."""
        return self.couplings[site, :, other, state]


def log_conditional_from_logits(logits: torch.Tensor, state: int, beta: float = 1.0) -> float:
    """``log softmax(beta * logits)[state]``."""
    scaled = beta * logits
    return float(scaled[state] - torch.logsumexp(scaled, dim=0))


def _write_cache(cache_file: Path, params: Dict[str, torch.Tensor], alphabet: Alphabet, dtype: str) -> None:
    try:
        torch.save(
            {"params": {k: v.cpu() for k, v in params.items()}, "tokens": alphabet.tokens, "dtype": dtype},
            cache_file,
        )
    except OSError as error:  # a read-only model directory is not a failure
        print(f"WARNING: could not write the model cache {cache_file}: {error}")


def _read_cache(
    cache_file: Path,
    model_file: Path,
    alphabet: Alphabet,
    device: str,
    dtype: str,
) -> Dict[str, torch.Tensor] | None:
    if not cache_file.is_file() or cache_file.stat().st_mtime < model_file.stat().st_mtime:
        return None
    payload = torch.load(cache_file, map_location=device, weights_only=False)
    if payload.get("tokens") != alphabet.tokens or payload.get("dtype") != dtype:
        return None
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    return {key: value.to(device=device, dtype=torch_dtype) for key, value in payload["params"].items()}
