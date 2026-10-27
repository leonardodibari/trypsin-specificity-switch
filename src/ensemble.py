"""Many independent biased chains at once, vectorised over chains (CPU or GPU).

Same Hamiltonian and same move set as :mod:`mcmc` -- ``H = beta * E_DCA -
lambda * log P(S | critic)``, single-residue substitutions at the allowed,
non-gap sites, Metropolis acceptance ``min(1, exp(-dH))`` -- but a thousand
chains advance together as one tensor, which is what makes a sweep over lambda
affordable and what a GPU is for (pass ``device="cuda"``).

As in ``adabmDCA.sampling.gibbs_step_uniform_sites``, one step picks a single
site for the whole batch and scores every chain's substitution there with one
matrix product. The chains stay statistically independent: they share only the
order in which sites are visited, never a residue or an acceptance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import torch

from potts import PottsModel


@dataclass(frozen=True)
class EnsembleConfig:
    """Parameters of one batch of chains."""

    lam: float = 1.0
    beta: float = 1.0
    n_chains: int = 1000
    n_steps: int = 100_000
    seed: int = 0
    target_residue: str = "S"
    trace_every: int = 0  # 0 = no trace


@dataclass
class EnsembleResult:
    """Final state of every chain plus a coarse trace of the batch."""

    sequences: np.ndarray  # (n_chains, L), encoded
    log_p: np.ndarray  # (n_chains,) log P(S | critic) of the final sequences
    acceptance_rate: float
    config: EnsembleConfig
    critic_col: int
    allowed_cols: List[int]
    trace: Dict[str, np.ndarray] = field(default_factory=dict)


def run_ensemble(
    model: PottsModel,
    start_sequence: np.ndarray,
    critic_col: int,
    allowed_cols: Sequence[int],
    config: EnsembleConfig = EnsembleConfig(),
    progress_every: int = 0,
) -> EnsembleResult:
    """Run ``config.n_chains`` independent chains from the same starting sequence.

    Every chain performs ``config.n_steps`` proposed moves; the final sequences
    are returned. There is no stopping rule here -- unlike :func:`mcmc.run_mcmc`,
    this samples the biased ensemble rather than hunting for a first crossing.
    """
    device = model.bias.device
    dtype = model.bias.dtype
    alphabet = model.alphabet
    L, q = model.L, model.q
    n = int(config.n_chains)

    sites = np.asarray(sorted(set(int(c) for c in allowed_cols)), dtype=np.int64)
    if sites.size == 0:
        raise ValueError("No allowed site to mutate")
    start = np.asarray(start_sequence, dtype=np.int64)
    if np.any(start[sites] == alphabet.gap_index):
        raise ValueError("Allowed sites include a gap position of the starting sequence")

    target_state = alphabet.index(config.target_residue)
    states = torch.as_tensor(alphabet.residue_indices, device=device)  # never a gap
    n_states = int(states.numel())

    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.seed))

    chains = torch.as_tensor(start, device=device, dtype=torch.long).repeat(n, 1)  # (n, L)
    x = torch.nn.functional.one_hot(chains, num_classes=q).to(dtype)  # (n, L, q)
    flat = x.reshape(n, L * q)

    critic_block = model.couplings[critic_col]  # (q, L, q)
    critic_flat = critic_block.reshape(q, L * q)
    critic_logits = model.bias[critic_col] + flat @ critic_flat.T  # (n, q)
    log_p = _log_softmax_at(critic_logits, target_state, config.beta)

    site_tensor = torch.as_tensor(sites, device=device)
    rows = torch.arange(n, device=device)
    accepted_total = 0
    trace: Dict[str, List[np.ndarray]] = {"step": [], "P_S": [], "E_DCA": []}

    for step in range(1, int(config.n_steps) + 1):
        site = int(site_tensor[torch.randint(sites.size, (1,), generator=generator, device=device)])
        old_state = chains[:, site]

        # A uniform state different from the current one, gaps excluded.
        draw = torch.randint(n_states - 1, (n,), generator=generator, device=device)
        candidates = states.unsqueeze(0).expand(n, n_states)
        current_rank = (candidates == old_state.unsqueeze(1)).float().argmax(dim=1)
        new_state = states[draw + (draw >= current_rank).long()]

        site_logits = model.bias[site] + flat @ model.couplings[site].reshape(q, L * q).T  # (n, q)
        delta_energy = -(site_logits[rows, new_state] - site_logits[rows, old_state])

        if site == critic_col:
            new_critic_logits = critic_logits
            new_log_p = log_p
        else:
            block = critic_block[:, site, :]  # (q, q): J[critic, :, site, :]
            new_critic_logits = critic_logits + block[:, new_state].T - block[:, old_state].T
            new_log_p = _log_softmax_at(new_critic_logits, target_state, config.beta)

        delta_h = config.beta * delta_energy - config.lam * (new_log_p - log_p)
        uniform = torch.rand(n, generator=generator, device=device)
        accept = (delta_h <= 0) | (uniform < torch.exp(-delta_h.clamp(min=-80.0, max=80.0)))

        if bool(accept.any()):
            index = rows[accept]
            x[index, site, old_state[accept]] = 0.0
            x[index, site, new_state[accept]] = 1.0
            chains[index, site] = new_state[accept]
            critic_logits = torch.where(accept.unsqueeze(1), new_critic_logits, critic_logits)
            log_p = torch.where(accept, new_log_p, log_p)
            accepted_total += int(accept.sum())

        if config.trace_every and step % config.trace_every == 0:
            trace["step"].append(step)
            trace["P_S"].append(float(log_p.exp().mean()))
            trace["E_DCA"].append(float(_energy(model, x).mean()))
        if progress_every and step % progress_every == 0:
            print(
                f"  [lam={config.lam:g}] step {step}/{config.n_steps} "
                f"mean P(S)={float(log_p.exp().mean()):.4f} "
                f"acceptance={accepted_total / (step * n):.3f}",
                flush=True,
            )

    return EnsembleResult(
        sequences=chains.cpu().numpy(),
        log_p=log_p.cpu().numpy(),
        acceptance_rate=accepted_total / (config.n_steps * n),
        config=config,
        critic_col=int(critic_col),
        allowed_cols=[int(c) for c in sites],
        trace={key: np.asarray(value) for key, value in trace.items() if value},
    )


def _log_softmax_at(logits: torch.Tensor, state: int, beta: float) -> torch.Tensor:
    return torch.log_softmax(beta * logits, dim=1)[:, state]


def _energy(model: PottsModel, x: torch.Tensor) -> torch.Tensor:
    from adabmDCA.statmech import compute_energy

    return compute_energy(x, model.params)


def lam_dirname(lam: float) -> str:
    """``1.0 -> 'lam1'`` -- the same naming the single-chain runs use."""
    return f"lam{lam:g}"


__all__ = ["EnsembleConfig", "EnsembleResult", "run_ensemble", "lam_dirname"]
