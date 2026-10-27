#!/usr/bin/env python
"""Run a single biased-MCMC chain from wild-type trypsin toward a Ser-189 pocket.

Example::

    python scripts/run_mcmc.py \\
        --wt data/wt_trypsin.fasta \\
        --model data/adabmdca_model \\
        --mapping data/mapping_ShoMSA_pdb3TGI.csv \\
        --output results/ \\
        --lam 1.0 --beta 1.0 --threshold 0.80 \\
        --allowed-sites 160-228 \\
        --max-steps 1000000 --record-every 100 --seed 42
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from _common import add_common_arguments, prepare_run
from mcmc import MCMCConfig, lam_dirname, run_mcmc, save_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--lam", type=float, default=1.0, help="Bias strength lambda (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--ungapped-fasta",
        action="store_true",
        help="Write sequences.fasta without the alignment gaps",
    )
    parser.add_argument(
        "--list-sites",
        action="store_true",
        help="Print the resolved critic site and mutable columns, then exit",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print a progress line every N proposed steps (0 = quiet)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup = prepare_run(args)
    if args.list_sites:
        return

    config = MCMCConfig(
        lam=args.lam,
        beta=args.beta,
        threshold=args.threshold,
        max_steps=args.max_steps,
        record_every=args.record_every,
        seed=args.seed,
        target_residue=args.target_residue,
    )

    result = run_mcmc(
        model=setup.model,
        start_sequence=setup.wt_sequence,
        critic_col=setup.critic_col,
        allowed_cols=setup.allowed_cols,
        config=config,
        progress_every=args.progress_every,
    )

    output_dir = Path(args.output) / lam_dirname(args.lam)
    save_result(result, output_dir, ungapped=args.ungapped_fasta)

    status = "reached the threshold" if result.converged else "hit the step limit"
    print(
        f"\n{status} after {result.proposed_steps} proposed / {result.accepted_steps} accepted steps "
        f"(acceptance {result.acceptance_rate:.3f})"
    )
    print(f"final E_DCA = {result.final_energy:.3f}   P(S|critic) = {math.exp(result.final_logP):.4f}")
    print(f"written to {output_dir}")


if __name__ == "__main__":
    main()
