"""Metropolis-Hastings sampling under a specificity-biased Potts Hamiltonian.

The sampled Hamiltonian is

.. math::
    H(\\sigma) = \\beta\\, E_{\\mathrm{DCA}}(\\sigma)
                 - \\lambda \\log P(S \\mid i_{\\mathrm{critic}}, \\sigma)

where :math:`P(S \\mid i_{\\mathrm{critic}}, \\sigma)` is the single-site
conditional of the model at the specificity-switching site (3TGI 189) evaluated
on the target residue (Serine by default). A **larger lambda pulls harder toward
S**, at the cost of the DCA energy; lambda = 0 is plain Potts sampling at
inverse temperature beta.

``beta`` multiplies the DCA energy alone. The conditional is always evaluated at
inverse temperature 1 (:data:`CRITIC_BETA`), so that cooling the Potts term does
not also sharpen the bias: the bias stays the model's own probability of S at
that site. At ``beta = 1`` the two conventions coincide.

A move substitutes one residue at one allowed site by another non-gap residue,
drawn uniformly. The proposal is therefore symmetric and the acceptance is
``min(1, exp(-dH))``.

Note that :math:`P(S \\mid i_{\\mathrm{critic}}, \\sigma)` is a conditional *given
the rest of the sequence*, so it does not depend on the residue sitting at the
critic site: a move at the critic site changes the DCA energy only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch

from encoding import Alphabet, ungap, write_sequences
from potts import PottsModel, log_conditional_from_logits

#: See the module docstring: the critic conditional is never scaled by ``beta``.
CRITIC_BETA = 1.0


@dataclass(frozen=True)
class MCMCConfig:
    """Everything that defines one chain, apart from the model and the WT."""

    lam: float = 1.0
    beta: float = 1.0
    threshold: float = 0.80
    max_steps: int = 1_000_000
    record_every: int = 100
    seed: int = 42
    target_residue: str = "S"


@dataclass
class Record:
    """One recorded state of the chain."""

    step: int
    msa_col_changed: int | None
    aa_from: str | None
    aa_to: str | None
    E_DCA: float
    logP_S: float
    H_eff: float
    sequence: str = field(repr=False, default="")


@dataclass
class MCMCResult:
    """Outcome of a single chain."""

    records: List[Record]
    final_sequence: str
    final_energy: float
    final_logP: float
    proposed_steps: int
    accepted_steps: int
    converged: bool
    first_crossing_step: int | None
    config: MCMCConfig
    critic_col: int
    allowed_cols: List[int]

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_steps / self.proposed_steps if self.proposed_steps else 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "step": record.step,
                    "msa_col_changed": "" if record.msa_col_changed is None else record.msa_col_changed,
                    "aa_from": record.aa_from or "",
                    "aa_to": record.aa_to or "",
                    "E_DCA": record.E_DCA,
                    "logP_S": record.logP_S,
                    "H_eff": record.H_eff,
                }
                for record in self.records
            ],
            columns=["step", "msa_col_changed", "aa_from", "aa_to", "E_DCA", "logP_S", "H_eff"],
        )


def run_mcmc(
    model: PottsModel,
    start_sequence: np.ndarray,
    critic_col: int,
    allowed_cols: Sequence[int],
    config: MCMCConfig = MCMCConfig(),
    progress_every: int = 0,
) -> MCMCResult:
    """Run one biased Metropolis-Hastings chain from ``start_sequence``.

    Args:
        model: the loaded bmDCA model.
        start_sequence: encoded, aligned starting sequence (the WT).
        critic_col: MSA column of the specificity-switching site (3TGI 189).
        allowed_cols: MSA columns a move may touch (already gap-filtered).
        config: chain parameters.
        progress_every: if > 0, print a line every that many proposed steps.

    Returns:
        An :class:`MCMCResult` holding the recorded trajectory.
    """
    alphabet = model.alphabet
    if not allowed_cols:
        raise ValueError("No allowed site to mutate")
    sequence = np.array(start_sequence, dtype=np.int64, copy=True)
    if sequence.shape[0] != model.L:
        raise ValueError(f"Start sequence has length {sequence.shape[0]}, model expects {model.L}")

    target_state = alphabet.index(config.target_residue)
    sites = np.asarray(sorted(set(int(c) for c in allowed_cols)), dtype=np.int64)
    if np.any(sequence[sites] == alphabet.gap_index):
        raise ValueError("Allowed sites include a gap position of the starting sequence")
    states = alphabet.residue_indices  # never propose a gap
    rng = np.random.default_rng(config.seed)

    energy = model.energy(sequence)
    logits = model.site_logits(critic_col, sequence).clone()
    log_p = log_conditional_from_logits(logits, target_state, CRITIC_BETA)
    hamiltonian = config.beta * energy - config.lam * log_p

    records = [
        Record(
            step=0,
            msa_col_changed=None,
            aa_from=None,
            aa_to=None,
            E_DCA=energy,
            logP_S=log_p,
            H_eff=hamiltonian,
            sequence=alphabet.decode(sequence),
        )
    ]

    accepted = 0
    proposed = 0
    first_crossing: int | None = 0 if math.exp(log_p) >= config.threshold else None
    converged = first_crossing is not None

    while not converged and proposed < config.max_steps:
        proposed += 1
        site = int(rng.choice(sites))
        old_state = int(sequence[site])
        candidates = states[states != old_state]
        new_state = int(rng.choice(candidates))

        delta_energy = model.delta_energy(sequence, site, new_state)
        if site == critic_col:
            # The conditional at the critic site does not depend on its own residue.
            new_logits = logits
            new_log_p = log_p
        else:
            new_logits = (
                logits
                + model.coupling_column(critic_col, site, new_state)
                - model.coupling_column(critic_col, site, old_state)
            )
            new_log_p = log_conditional_from_logits(new_logits, target_state, CRITIC_BETA)

        delta_h = config.beta * delta_energy - config.lam * (new_log_p - log_p)
        if delta_h <= 0.0 or rng.random() < math.exp(-delta_h):
            sequence[site] = new_state
            energy += delta_energy
            logits = new_logits
            log_p = new_log_p
            hamiltonian += delta_h
            accepted += 1

            if math.exp(log_p) >= config.threshold:
                converged = True
                first_crossing = accepted

            if accepted % config.record_every == 0 or converged:
                records.append(
                    Record(
                        step=accepted,
                        msa_col_changed=site,
                        aa_from=alphabet.tokens[old_state],
                        aa_to=alphabet.tokens[new_state],
                        E_DCA=energy,
                        logP_S=log_p,
                        H_eff=config.beta * energy - config.lam * log_p,
                        sequence=alphabet.decode(sequence),
                    )
                )

        if progress_every and proposed % progress_every == 0:
            print(
                f"  [lam={config.lam:g} seed={config.seed}] proposed={proposed} "
                f"accepted={accepted} E={energy:.2f} P(S)={math.exp(log_p):.4f}",
                flush=True,
            )

    if records[-1].step != accepted:
        records.append(
            Record(
                step=accepted,
                msa_col_changed=None,
                aa_from=None,
                aa_to=None,
                E_DCA=energy,
                logP_S=log_p,
                H_eff=config.beta * energy - config.lam * log_p,
                sequence=alphabet.decode(sequence),
            )
        )

    # Guard against incremental drift over very long chains.
    energy = model.energy(sequence)
    log_p = model.log_conditional(critic_col, sequence, target_state, beta=CRITIC_BETA)

    return MCMCResult(
        records=records,
        final_sequence=alphabet.decode(sequence),
        final_energy=energy,
        final_logP=log_p,
        proposed_steps=proposed,
        accepted_steps=accepted,
        converged=converged,
        first_crossing_step=first_crossing,
        config=config,
        critic_col=int(critic_col),
        allowed_cols=[int(c) for c in sites],
    )


def save_result(result: MCMCResult, output_dir: str | Path, ungapped: bool = False) -> Path:
    """Write ``trajectory.csv``, ``sequences.fasta`` and ``summary.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.to_frame().to_csv(output_dir / "trajectory.csv", index=False)

    records = (
        (f"step_{record.step}", ungap(record.sequence) if ungapped else record.sequence)
        for record in result.records
    )
    write_sequences(output_dir / "sequences.fasta", records)

    summary = {
        **asdict(result.config),
        "critic_col": result.critic_col,
        "n_allowed_sites": len(result.allowed_cols),
        "allowed_cols": result.allowed_cols,
        "proposed_steps": result.proposed_steps,
        "accepted_steps": result.accepted_steps,
        "acceptance_rate": result.acceptance_rate,
        "converged": result.converged,
        "first_crossing_step": result.first_crossing_step,
        "final_E_DCA": result.final_energy,
        "final_logP_S": result.final_logP,
        "final_P_S": math.exp(result.final_logP),
        "final_sequence": result.final_sequence,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return output_dir


def lam_dirname(lam: float) -> str:
    """Directory name for a lambda value: ``1.0 -> 'lam1'``, ``0.5 -> 'lam0.5'``."""
    return f"lam{lam:g}"


__all__ = [
    "MCMCConfig",
    "MCMCResult",
    "Record",
    "run_mcmc",
    "save_result",
    "lam_dirname",
]
