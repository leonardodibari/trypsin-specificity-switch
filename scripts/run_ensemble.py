#!/usr/bin/env python
"""Sample a large ensemble of independent chains per lambda and score the result.

Every chain starts from the wild type, runs ``--n-steps`` proposed moves under
``H = beta * E_DCA - lambda * log P(S | critic)``, and contributes its **final**
sequence. The chains advance together as one tensor, so a thousand of them cost
barely more than one; ``--device cuda`` puts the whole batch on a GPU.

Example::

    python scripts/run_ensemble.py \\
        --wt data/wt_trypsin.fasta \\
        --model ../bmDCA_params.dat \\
        --output ensembles/ \\
        --critic-col 197 --allowed-cols 180-220 \\
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
from ensemble import EnsembleConfig, lam_dirname, run_ensemble
from observables import describe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--lambdas", type=float, nargs="+", required=True, help="Bias strengths to sample")
    parser.add_argument("--n-chains", type=int, default=1000, help="Chains per lambda (default: 1000)")
    parser.add_argument(
        "--n-steps", type=int, default=20_000, help="Proposed moves per chain (default: 20000)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed; lambda i uses seed + i")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # --threshold and --record-every belong to the single-chain runs; an
    # ensemble has no stopping rule and keeps only its final state.
    setup = prepare_run(args)
    alphabet = setup.alphabet
    target_state = alphabet.index(args.target_residue)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, lam in enumerate(args.lambdas):
        traced = args.trace_lambdas is None or any(
            abs(lam - value) < 1e-12 for value in args.trace_lambdas
        )
        config = EnsembleConfig(
            lam=lam,
            beta=args.beta,
            n_chains=args.n_chains,
            n_steps=args.n_steps,
            seed=args.seed + index,
            target_residue=args.target_residue,
            trace_every=args.trace_every if traced else 0,
            trace_cde=not args.no_trace_cde,
            trace_schedule=args.trace_schedule,
        )
        print(
            f"\nlambda = {lam:g}: {args.n_chains} chains x {args.n_steps} steps on {args.device}"
            + ("  (traced)" if traced and args.trace_every else "")
        )
        result = run_ensemble(
            model=setup.model,
            start_sequence=setup.wt_sequence,
            critic_col=setup.critic_col,
            allowed_cols=setup.allowed_cols,
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
        )
        frame = pd.DataFrame({"chain": np.arange(result.sequences.shape[0]), **observables})
        frame.insert(1, "lam", lam)

        directory = output / lam_dirname(lam)
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / "observables.csv", index=False)
        write_sequences(
            directory / "final_sequences.fasta",
            (
                (f"lam{lam:g}_chain{chain}", _maybe_ungap(alphabet.decode(sequence), args.ungapped_fasta))
                for chain, sequence in enumerate(result.sequences)
            ),
        )
        if result.trace:
            pd.DataFrame(result.trace).to_csv(directory / "trace.csv", index=False)
        summary = {
            "lam": lam,
            "beta": args.beta,
            "n_chains": args.n_chains,
            "n_steps": args.n_steps,
            "seed": config.seed,
            "acceptance_rate": result.acceptance_rate,
            "critic_col": result.critic_col,
            "n_allowed_sites": len(result.allowed_cols),
            "mean_E_DCA": float(frame["E_DCA"].mean()),
            "mean_P_S": float(frame["P_S"].mean()),
            "mean_hamming_to_wt": float(frame["hamming_to_wt"].mean()),
            "mean_CDE": float(frame["CDE"].mean()),
        }
        with open(directory / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        rows.append(summary)
        print(
            f"  acceptance {result.acceptance_rate:.3f} | "
            f"<E> {summary['mean_E_DCA']:.1f} | <P(S)> {summary['mean_P_S']:.3f} | "
            f"<hamming> {summary['mean_hamming_to_wt']:.1f} | <CDE> {summary['mean_CDE']:.3f}"
        )

    pd.DataFrame(rows).to_csv(output / "ensemble_summary.csv", index=False)
    print(f"\nwritten to {output}")


def _maybe_ungap(sequence: str, ungapped: bool) -> str:
    return sequence.replace("-", "") if ungapped else sequence


if __name__ == "__main__":
    main()
