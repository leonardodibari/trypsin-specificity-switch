#!/usr/bin/env python
"""Sample a large ensemble of independent chains per lambda and score the result.

Every chain starts from the wild type, runs ``--n-steps`` proposed moves under
``H = beta * E_DCA - lambda * log P(S | critic) + t * d_H(sigma, WT)``, and
contributes its **final**
sequence. The chains advance together as one tensor, so a thousand of them cost
barely more than one; ``--device cuda`` puts the whole batch on a GPU.

Example::

    python scripts/run_ensemble.py \\
        --wt data/wt_trypsin.fasta \\
        --model ../bmDCA_params.dat \\
        --output ensembles/ \\
        --critic-col 201 --allowed-cols 165-243 \\
        --lambdas 0 5 10 20 50 \\
        --n-chains 1000 --n-steps 20000 \\
        --device cuda --cache-model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import add_common_arguments, prepare_run
from encoding import write_sequences
from ensemble import EnsembleConfig, run_dirname, run_ensemble
from observables import describe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--lambdas", type=float, nargs="+", required=True, help="Bias strengths to sample")
    parser.add_argument(
        "--t-values",
        type=float,
        nargs="+",
        default=[0.0],
        help="Hamming-restraint strengths. The Hamiltonian carries + t * d_H(sigma, WT), "
        "so t > 0 holds the chains near the wild type, t < 0 pays them to leave it and "
        "t = 0 (the default) is the original ladder, bit for bit. Every (lambda, t) pair "
        "is sampled; t is the chemical potential conjugate to the Hamming distance, so a "
        "scan over it traces <d>(t) and its susceptibility Var(d) = -d<d>/dt. With "
        "--t-indel-values unset this prices substitutions and indels alike, which is "
        "what t alone has always meant",
    )
    parser.add_argument(
        "--t-indel-values",
        type=float,
        nargs="+",
        default=None,
        help="Price of an indel (a gained or lost residue), separately from the price of a "
        "substitution in --t-values. Unset (the default) an indel costs the same t as a "
        "substitution; with --allow-gap-moves that means asking for a d-substitution design "
        "silently also charges t per indel, which is why this knob exists. Given, every "
        "(lambda, t_sub, t_indel) triple is sampled and the runs carrying a distinct indel "
        "price get a _ti suffix in their directory name. --t-indel-values 0 prices indels "
        "free; a large value suppresses them without touching the substitution budget",
    )
    parser.add_argument(
        "--mu",
        default=None,
        help="mu of the quality score Q = E_DCA - mu * CDE, as a number or as the path of "
        "a calibration mu.json. Given, Q goes into the trace, so saturation can be "
        "checked on the score the designs are ranked by and not only on its two parts",
    )
    parser.add_argument("--n-chains", type=int, default=1000, help="Chains per lambda (default: 1000)")
    parser.add_argument(
        "--n-steps", type=int, default=20_000, help="Proposed moves per chain (default: 20000)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed; run i uses seed + i")
    parser.add_argument(
        "--trace-every",
        type=int,
        default=0,
        help="Record the batch's observables every N proposed steps (0 = off)",
    )
    parser.add_argument(
        "--trace-lambdas",
        type=float,
        nargs="*",
        default=None,
        help="Trace only these lambdas (default: all of them). The trace costs a full "
        "all-site pass per recorded point, so two or three are usually enough",
    )
    parser.add_argument(
        "--trace-schedule",
        default="linear",
        choices=("linear", "log"),
        help="'linear' records every --trace-every steps; 'log' spaces the same number of "
        "points logarithmically, which resolves the approach to equilibrium",
    )
    parser.add_argument(
        "--no-trace-cde",
        action="store_true",
        help="Leave the context entropy out of the trace (it is its most expensive part)",
    )
    parser.add_argument(
        "--progress-every", type=int, default=0, help="Print a progress line every N steps (0 = quiet)"
    )
    parser.add_argument(
        "--ungapped-fasta", action="store_true", help="Write the final sequences without alignment gaps"
    )
    parser.add_argument(
        "--clamp-critic",
        action="store_true",
        help="CLAMP-AND-RELAX. Put --target-residue at the critic column of the starting "
        "sequence and freeze it there, then sample the rest. Normally the critic column "
        "is an ordinary mutable site and the bias is what pulls it toward S; under the "
        "clamp the residue is imposed and the chains answer a different question -- what "
        "the model thinks the REST of the protein should be, given a serine at i*. "
        "Use with --lambdas 0: it is the zero-parameter null the lambda ladder is "
        "measured against, and there is nothing in it for a bias to game",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # --threshold and --record-every belong to the single-chain runs; an
    # ensemble has no stopping rule and keeps only its final state.
    setup = prepare_run(args)
    alphabet = setup.alphabet
    target_state = alphabet.index(args.target_residue)

    start_sequence = setup.wt_sequence
    allowed_cols = setup.allowed_cols
    if args.clamp_critic:
        # Both halves are needed. Imposing S without freezing the column would
        # let the chains walk straight back to D -- at lambda = 0 nothing holds
        # them there. Freezing without imposing would clamp the WILD TYPE's D.
        start_sequence = start_sequence.copy()
        start_sequence[setup.critic_col] = target_state
        allowed_cols = [c for c in allowed_cols if c != setup.critic_col]
        if args.lambdas != [0.0] and any(lam != 0.0 for lam in args.lambdas):
            print("NOTE: --clamp-critic with lambda != 0. The bias still acts on "
                  "P(S | sigma_\\i*), which does not depend on the clamped residue, "
                  "so it now shapes the CONTEXT around an imposed serine.")
        print(f"clamped: critic column {setup.critic_col} fixed at "
              f"{args.target_residue!r}; {len(allowed_cols)} mutable columns remain")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    mu = _resolve_mu(args.mu)
    against_state = alphabet.index(args.against_residue)

    rows = []
    # Every (lambda, t_sub, t_indel) triple. With --t-indel-values unset the
    # indel price follows t_sub, so the grid, the seeds (seed + index, index
    # running over the grid in this order) and the directory names are exactly
    # the old ones; with the default --t-values 0 on top of that it is exactly
    # the original loop over lambdas.
    indel_values = [None] if args.t_indel_values is None else list(args.t_indel_values)
    grid = [
        (lam, t, t_indel)
        for lam in args.lambdas
        for t in args.t_values
        for t_indel in indel_values
    ]
    for index, (lam, t, t_indel) in enumerate(grid):
        traced = args.trace_lambdas is None or any(
            abs(lam - value) < 1e-12 for value in args.trace_lambdas
        )
        config = EnsembleConfig(
            lam=lam,
            t=t,
            t_indel=t_indel,
            mu=mu,
            beta=args.beta,
            n_chains=args.n_chains,
            n_steps=args.n_steps,
            seed=args.seed + index,
            target_residue=args.target_residue,
            trace_every=args.trace_every if traced else 0,
            trace_cde=not args.no_trace_cde,
            trace_schedule=args.trace_schedule,
            allow_gap_moves=args.allow_gap_moves,
            bias=args.bias,
            against_residue=args.against_residue,
        )
        print(
            f"\nlambda = {lam:g}, t_sub = {t:g}"
            + ("" if t_indel is None else f", t_indel = {t_indel:g}")
            + f": {args.n_chains} chains x {args.n_steps} steps "
            f"on {args.device}" + ("  (traced)" if traced and args.trace_every else "")
        )
        result = run_ensemble(
            model=setup.model,
            start_sequence=start_sequence,
            critic_col=setup.critic_col,
            allowed_cols=allowed_cols,
            config=config,
            progress_every=args.progress_every,
        )

        observables = describe(
            setup.model,
            result.sequences,
            reference=setup.wt_sequence,
            critic_col=setup.critic_col,
            target_state=target_state,
            beta=args.beta,
            against_state=against_state,
        )
        frame = pd.DataFrame({"chain": np.arange(result.sequences.shape[0]), **observables})
        frame.insert(1, "lam", lam)
        frame.insert(2, "t", t)
        # The indel price as sampled, so a table of results never has to infer it
        # from a directory name: it equals t wherever it was not set apart.
        frame.insert(3, "t_indel", t if t_indel is None else t_indel)

        directory = output / run_dirname(lam, t, t_indel)
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / "observables.csv", index=False)
        write_sequences(
            directory / "final_sequences.fasta",
            (
                (f"{run_dirname(lam, t, t_indel)}_chain{chain}", _maybe_ungap(alphabet.decode(sequence), args.ungapped_fasta))
                for chain, sequence in enumerate(result.sequences)
            ),
        )
        if result.trace:
            pd.DataFrame(result.trace).to_csv(directory / "trace.csv", index=False)
        summary = {
            "lam": lam,
            "t": t,
            "mu": mu,
            "beta": args.beta,
            "n_chains": args.n_chains,
            "n_steps": args.n_steps,
            "seed": config.seed,
            "acceptance_rate": result.acceptance_rate,
            "critic_col": result.critic_col,
            "n_allowed_sites": len(result.allowed_cols),
            "allowed_cols": result.allowed_cols,
            "allow_gap_moves": bool(args.allow_gap_moves),
            "clamp_critic": bool(args.clamp_critic),
            "bias": args.bias,
            "against_residue": args.against_residue,
            "mean_E_DCA": float(frame["E_DCA"].mean()),
            "mean_n_gaps": float(frame["n_gaps"].mean()),
            "mean_P_S": float(frame["P_S"].mean()),
            "mean_hamming_to_wt": float(frame["hamming_to_wt"].mean()),
            "mean_CDE": float(frame["CDE"].mean()),
            "mean_LOR": float(frame["LOR"].mean()),
            # Var(d) is the susceptibility conjugate to t: -d<d>/dt. A peak in it
            # along a t scan is a transition between two basins in d, which is the
            # whole point of scanning t.
            "var_hamming_to_wt": float(frame["hamming_to_wt"].var(ddof=1)),
        }
        if mu is not None:
            summary["mean_Q"] = float((frame["E_DCA"] - mu * frame["CDE"]).mean())
        with open(directory / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        rows.append(summary)
        print(
            f"  acceptance {result.acceptance_rate:.3f} | "
            f"<E> {summary['mean_E_DCA']:.1f} | <P(S)> {summary['mean_P_S']:.3f} | "
            f"<hamming> {summary['mean_hamming_to_wt']:.1f} "
            f"(var {summary['var_hamming_to_wt']:.1f}) | <CDE> {summary['mean_CDE']:.3f}"
        )

    pd.DataFrame(rows).to_csv(output / "ensemble_summary.csv", index=False)
    print(f"\nwritten to {output}")


def _maybe_ungap(sequence: str, ungapped: bool) -> str:
    return sequence.replace("-", "") if ungapped else sequence


def _resolve_mu(value: str | None) -> float | None:
    """``--mu`` as a number, or as the path of a calibration ``mu.json``."""
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        with open(path, encoding="utf-8") as handle:
            return float(json.load(handle)["mu"])
    try:
        return float(value)
    except ValueError:
        raise SystemExit(f"--mu is neither a number nor an existing file: {value!r}")


if __name__ == "__main__":
    main()
