"""Metropolis-Hastings sampling under a specificity-biased Potts Hamiltonian.

The sampled Hamiltonian is

.. math::
    H(\\sigma) = \\beta\\, E_{\\mathrm{DCA}}(\\sigma)
                 - \\lambda \\log P(S \\mid i_{\\mathrm{critic}}, \\sigma)
                 + t\\, d_H(\\sigma, \\sigma^{\\mathrm{WT}})

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
from typing import List, Optional, Sequence

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
    allow_gap_moves: bool = False  # treat the gap as an ordinary state (see run_mcmc)
    #: Hamming restraint toward the starting sequence: ``+ t * d_H(sigma, start)``
    #: in the Hamiltonian. t > 0 holds the chain near the wild type, t < 0 pushes
    #: it away, t = 0 is the original. See :mod:`ensemble` for the full note.
    t: float = 0.0
    #: The restraint split by move type -- ``t_sub`` prices substitutions,
    #: ``t_indel`` prices gained or lost residues, and ``None`` falls back to
    #: ``t``. Needed because with ``allow_gap_moves`` a single ``t`` charges an
    #: indel at the substitution price without saying so. See :mod:`ensemble`.
    t_sub: Optional[float] = None
    t_indel: Optional[float] = None

    @property
    def t_sub_effective(self) -> float:
        return self.t if self.t_sub is None else self.t_sub

    @property
    def t_indel_effective(self) -> float:
        return self.t if self.t_indel is None else self.t_indel


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
    if not config.allow_gap_moves and np.any(sequence[sites] == alphabet.gap_index):
        raise ValueError("Allowed sites include a gap position of the starting sequence")
    # With ``allow_gap_moves`` the gap is an ordinary state: it may be proposed and
    # a column that starts as a gap may mutate, so insertions and deletions are part
    # of the move set. Off by default, which freezes the wild type's gap pattern.
    states = (
        np.arange(model.q, dtype=np.int64) if config.allow_gap_moves else alphabet.residue_indices
    )
    rng = np.random.default_rng(config.seed)

    energy = model.energy(sequence)
    logits = model.site_logits(critic_col, sequence).clone()
    log_p = log_conditional_from_logits(logits, target_state, CRITIC_BETA)
    # The Hamming restraint is measured from the starting sequence (the wild
    # type), over the allowed columns only, split into its substitution and
    # indel parts, and carried incrementally: a move changes each by -1, 0 or +1.
    # Over a run whose allowed set is all of L the restriction is a no-op; where
    # it is not, it keeps a frozen difference out of a distance no move can
    # change. See :mod:`ensemble` for the full note.
    gap_index = alphabet.gap_index
    t_sub = float(config.t_sub_effective)
    t_indel = float(config.t_indel_effective)
    reference = np.array(start_sequence, dtype=np.int64, copy=True)

    def _split_distance(current: np.ndarray) -> tuple[int, int]:
        """``(d_sub, d_indel)`` over the allowed columns, disjoint, summing to d_H."""
        differs = current[sites] != reference[sites]
        indel = differs & ((current[sites] == gap_index) | (reference[sites] == gap_index))
        return int((differs & ~indel).sum()), int(indel.sum())

    d_sub, d_indel = _split_distance(sequence)

    def _restraint(sub: int, indel: int) -> float:
        return t_sub * sub + t_indel * indel

    hamiltonian = config.beta * energy - config.lam * log_p + _restraint(d_sub, d_indel)

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

        # Both parts are needed even when their sum is unchanged: a residue -> gap
        # move at a site whose reference residue is present converts a
        # substitution into an indel, repricing it from t_sub to t_indel.
        reference_state = int(reference[site])
        reference_is_gap = reference_state == gap_index
        old_differs = int(old_state) != reference_state
        new_differs = int(new_state) != reference_state
        old_indel = old_differs and (int(old_state) == gap_index or reference_is_gap)
        new_indel = new_differs and (int(new_state) == gap_index or reference_is_gap)
        delta_sub = int(new_differs and not new_indel) - int(old_differs and not old_indel)
        delta_indel = int(new_indel) - int(old_indel)
        delta_h = (
            config.beta * delta_energy
            - config.lam * (new_log_p - log_p)
            + _restraint(delta_sub, delta_indel)
        )
        if delta_h <= 0.0 or rng.random() < math.exp(-delta_h):
            sequence[site] = new_state
            energy += delta_energy
            logits = new_logits
            log_p = new_log_p
            d_sub += delta_sub
            d_indel += delta_indel
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
                        H_eff=config.beta * energy - config.lam * log_p + _restraint(d_sub, d_indel),
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
                H_eff=config.beta * energy - config.lam * log_p + _restraint(d_sub, d_indel),
                sequence=alphabet.decode(sequence),
            )
        )

    # Guard against incremental drift over very long chains.
    energy = model.energy(sequence)
    log_p = model.log_conditional(critic_col, sequence, target_state, beta=CRITIC_BETA)
    # The same guard on the restraint's two counters. They are checked separately:
    # their sum can be right while weight has leaked between them, which is the
    # error a split restraint makes possible and a single total would hide.
    exact_sub, exact_indel = _split_distance(sequence)
    if (exact_sub, exact_indel) != (d_sub, d_indel):
        raise RuntimeError(
            "Incremental Hamming distance drifted: carried "
            f"(sub={d_sub}, indel={d_indel}), recomputed "
            f"(sub={exact_sub}, indel={exact_indel})"
        )

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
