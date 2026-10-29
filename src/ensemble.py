"""Many independent biased chains at once, vectorised over chains (CPU or GPU).

Same Hamiltonian and same move set as :mod:`mcmc` -- ``H = beta * E_DCA -
lambda * log P(S | critic)``, with the critic conditional always at beta = 1
(see ``CRITIC_BETA``), single-residue substitutions at the allowed,
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
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from potts import PottsModel

#: The critic's conditional ``P(S | critic, sigma)`` is always evaluated at
#: inverse temperature 1, whatever ``beta`` the DCA energy is sampled at. ``beta``
#: scales the energy term alone, so that it tunes how cold the Potts sampling is
#: without also sharpening the bias; the bias stays the model's own probability of
#: S at that site. At ``beta = 1`` nothing changes, so every earlier run stands.
CRITIC_BETA = 1.0


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
    trace_cde: bool = True  # the context entropy costs a full all-site pass
    trace_schedule: str = "linear"  # "linear" every trace_every steps, or "log"-spaced
    allow_gap_moves: bool = False  # treat the gap as an ordinary state (see run_ensemble)


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
    trace_hook: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]] = None,
    reference_sequence: Optional[np.ndarray] = None,
) -> EnsembleResult:
    """Run ``config.n_chains`` independent chains from the same starting sequence.

    Every chain performs ``config.n_steps`` proposed moves; the final sequences
    are returned. There is no stopping rule here -- unlike :func:`mcmc.run_mcmc`,
    this samples the biased ensemble rather than hunting for a first crossing.

    ``start_sequence`` is one sequence ``(L,)`` for every chain, or one per chain
    ``(n, L)`` -- starting the chains spread over the family rather than all at
    the same point is how ``run_generative_check.py`` tells a model that cannot
    reproduce its statistics from chains that simply have not mixed.
    ``reference_sequence`` is what the Hamming distance is measured against, the
    first starting sequence by default.

    With ``config.allow_gap_moves`` the gap becomes an ordinary state: it may be
    proposed, and a column that starts as a gap may be mutated. This changes the
    state space, and with it the equilibrium distribution. It is off by default,
    because a design run is meant to keep the wild type's gap pattern, and on for
    the generative check of ``run_generative_check.py``, where the chains must be
    free to move over the same space the model was fitted on -- a sampler that
    cannot produce a gap cannot reproduce the statistics of an alignment that is
    one-tenth gaps.

    ``trace_hook``, if given, is called at every traced step with the encoded
    chains ``(n, L)`` and their one-hot form ``(n, L, q)``; the scalars it returns
    are appended to the trace next to the built-in observables. It is what
    ``scripts/run_generative_check.py`` uses to follow the batch's agreement with
    the statistics of the natural alignment along the run.
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
    if start.ndim == 1:
        start = np.broadcast_to(start, (n, start.shape[0]))
    elif start.shape[0] != n:
        raise ValueError(f"{start.shape[0]} starting sequences for {n} chains")
    start = np.ascontiguousarray(start)
    if not config.allow_gap_moves and np.any(start[:, sites] == alphabet.gap_index):
        raise ValueError("Allowed sites include a gap position of a starting sequence")
    if reference_sequence is None:
        reference_sequence = start[0]

    target_state = alphabet.index(config.target_residue)
    states = torch.arange(q, device=device) if config.allow_gap_moves else torch.as_tensor(
        alphabet.residue_indices, device=device
    )
    n_states = int(states.numel())

    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.seed))

    chains = torch.as_tensor(start, device=device, dtype=torch.long).clone()  # (n, L)
    x = torch.nn.functional.one_hot(chains, num_classes=q).to(dtype)  # (n, L, q)
    flat = x.reshape(n, L * q)

    critic_block = model.couplings[critic_col]  # (q, L, q)
    critic_flat = critic_block.reshape(q, L * q)
    critic_logits = model.bias[critic_col] + flat @ critic_flat.T  # (n, q)
    log_p = _log_softmax_at(critic_logits, target_state, CRITIC_BETA)

    site_tensor = torch.as_tensor(sites, device=device)
    rows = torch.arange(n, device=device)
    accepted_total = 0
    trace: Dict[str, List[float]] = {}
    reference = torch.as_tensor(np.asarray(reference_sequence), device=device, dtype=torch.long)
    trace_steps = _trace_steps(config)

    for step in range(1, int(config.n_steps) + 1):
        site = int(site_tensor[torch.randint(sites.size, (1,), generator=generator, device=device)])
        old_state = chains[:, site]

        # A uniform state different from the current one.
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
            new_log_p = _log_softmax_at(new_critic_logits, target_state, CRITIC_BETA)

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

        if step in trace_steps:
            _record(
                trace,
                model=model,
                chains=chains,
                x=x,
                log_p=log_p,
                reference=reference,
                step=step,
                accepted=accepted_total / n,
                with_cde=config.trace_cde,
                beta=CRITIC_BETA,
                extra=trace_hook(chains, x) if trace_hook is not None else None,
            )
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


def run_ensemble_gibbs(
    model: PottsModel,
    start_sequence: np.ndarray,
    critic_col: int,
    config: EnsembleConfig = EnsembleConfig(),
    progress_every: int = 0,
    trace_hook: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]] = None,
    reference_sequence: Optional[np.ndarray] = None,
) -> EnsembleResult:
    """The same ensemble, advanced by adabmDCA's own Gibbs sampler.

    Each step resamples one site of every chain from its exact conditional
    ``P(a | rest)`` with :func:`adabmDCA.sampling.gibbs_step_uniform_sites` -- the
    step adabmDCA trains with -- so a "step" here is one site update per chain,
    directly comparable to one proposed Metropolis move. Every site is free and
    every state including the gap can be drawn, so there is no allowed-site list
    and no acceptance rate: a Gibbs move is always taken.

    Only ``lam = 0`` is supported. The bias ``-lambda log P(S | critic)`` is not a
    single-site term, so the conditional at a site would have to include its
    effect on the critic site's conditional, which is not what the Gibbs step
    computes. For the generative check, which is unbiased, that is no restriction.
    """
    from adabmDCA.sampling import gibbs_step_uniform_sites

    if config.lam != 0.0:
        raise ValueError("The Gibbs sampler is only correct for the unbiased ensemble (lam = 0)")

    device = model.bias.device
    dtype = model.bias.dtype
    L, q = model.L, model.q
    n = int(config.n_chains)

    start = np.asarray(start_sequence, dtype=np.int64)
    if start.ndim == 1:
        start = np.broadcast_to(start, (n, start.shape[0]))
    elif start.shape[0] != n:
        raise ValueError(f"{start.shape[0]} starting sequences for {n} chains")
    start = np.ascontiguousarray(start)
    if reference_sequence is None:
        reference_sequence = start[0]

    torch.manual_seed(int(config.seed))
    chains = torch.as_tensor(start, device=device, dtype=torch.long)
    x = torch.nn.functional.one_hot(chains, num_classes=q).to(dtype)

    target_state = model.alphabet.index(config.target_residue)
    critic_flat = model.couplings[critic_col].reshape(q, L * q)
    reference = torch.as_tensor(np.asarray(reference_sequence), device=device, dtype=torch.long)
    trace: Dict[str, List[float]] = {}
    trace_steps = _trace_steps(config)

    for step in range(1, int(config.n_steps) + 1):
        x = gibbs_step_uniform_sites(x, model.params, config.beta)
        if step in trace_steps or step == config.n_steps:
            encoded = x.argmax(dim=2)
            log_p = _log_softmax_at(model.bias[critic_col] + x.reshape(n, L * q) @ critic_flat.T,
                                    target_state, CRITIC_BETA)
            _record(
                trace, model=model, chains=encoded, x=x, log_p=log_p, reference=reference,
                step=step, accepted=float(step), with_cde=config.trace_cde, beta=CRITIC_BETA,
                extra=trace_hook(encoded, x) if trace_hook is not None else None,
            )
        if progress_every and step % progress_every == 0:
            print(f"  [gibbs] step {step}/{config.n_steps}", flush=True)

    encoded = x.argmax(dim=2)
    log_p = _log_softmax_at(
        model.bias[critic_col] + x.reshape(n, L * q) @ critic_flat.T, target_state, CRITIC_BETA
    )
    return EnsembleResult(
        sequences=encoded.cpu().numpy(),
        log_p=log_p.cpu().numpy(),
        acceptance_rate=float("nan"),  # a Gibbs move is always taken
        config=config,
        critic_col=int(critic_col),
        allowed_cols=list(range(L)),
        trace={key: np.asarray(value) for key, value in trace.items() if value},
    )


def _trace_steps(config: EnsembleConfig) -> set:
    """The steps at which the batch is scored.

    ``linear`` records every ``trace_every`` steps; ``log`` records about
    ``n_steps / trace_every`` log-spaced steps instead, which is what resolves
    the approach to equilibrium -- everything happens in the first few hundred
    moves and the rest is a flat line.
    """
    if not config.trace_every:
        return set()
    if config.trace_schedule == "log":
        count = max(2, config.n_steps // config.trace_every)
        steps = np.unique(np.round(np.logspace(0, np.log10(config.n_steps), count)).astype(int))
        return set(int(s) for s in steps) | {1, config.n_steps}
    return set(range(config.trace_every, config.n_steps + 1, config.trace_every)) | {1}


def _record(
    trace: Dict[str, List[float]],
    model: PottsModel,
    chains: torch.Tensor,
    x: torch.Tensor,
    log_p: torch.Tensor,
    reference: torch.Tensor,
    step: int,
    accepted: float,
    with_cde: bool,
    beta: float,
    extra: Optional[Dict[str, float]] = None,
) -> None:
    """Append the batch's observables at this step: mean and quartiles of each.

    ``extra`` holds the scalars of a ``trace_hook``, recorded as they come.
    """
    from observables import context_dependent_entropy

    values: Dict[str, torch.Tensor] = {
        "E_DCA": _energy(model, x),
        "hamming_to_wt": (chains != reference.unsqueeze(0)).sum(dim=1).to(torch.float32),
        "P_S": log_p.exp(),
    }
    if with_cde:
        values["CDE"] = torch.as_tensor(
            context_dependent_entropy(model, chains, beta=beta), device=chains.device
        )
    trace.setdefault("step", []).append(float(step))
    trace.setdefault("accepted_per_chain", []).append(float(accepted))
    for name, tensor in values.items():
        tensor = tensor.to(torch.float32)
        quartiles = torch.quantile(tensor, torch.tensor([0.25, 0.75], device=tensor.device))
        trace.setdefault(f"{name}_mean", []).append(float(tensor.mean()))
        trace.setdefault(f"{name}_q25", []).append(float(quartiles[0]))
        trace.setdefault(f"{name}_q75", []).append(float(quartiles[1]))
    for name, value in (extra or {}).items():
        trace.setdefault(name, []).append(float(value))


def _log_softmax_at(logits: torch.Tensor, state: int, beta: float) -> torch.Tensor:
    return torch.log_softmax(beta * logits, dim=1)[:, state]


def _energy(model: PottsModel, x: torch.Tensor) -> torch.Tensor:
    from adabmDCA.statmech import compute_energy

    return compute_energy(x, model.params)


def lam_dirname(lam: float) -> str:
    """``1.0 -> 'lam1'`` -- the same naming the single-chain runs use."""
    return f"lam{lam:g}"


__all__ = ["EnsembleConfig", "EnsembleResult", "run_ensemble", "run_ensemble_gibbs", "lam_dirname"]
