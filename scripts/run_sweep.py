#!/usr/bin/env python
"""Sweep the bias strength lambda, with several independent chains per lambda.

Each ``(lambda, run)`` pair is written to ``output/lam{value}/run{i}/``.

Example::

    python scripts/run_sweep.py \\
        --wt data/wt_trypsin.fasta \\
        --model data/adabmdca_model \\
        --mapping data/mapping_ShoMSA_pdb3TGI.csv \\
        --output results/ \\
        --lambdas 0.1 0.5 1.0 2.0 5.0 10.0 \\
        --beta 1.0 --threshold 0.80 \\
        --allowed-sites 160-228 \\
        --max-steps 1000000 --record-every 100 \\
        --n-runs 5 --seed 42 --n-jobs -1
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from joblib import Parallel, delayed

from _common import add_common_arguments, prepare_run
from mcmc import MCMCConfig, lam_dirname, run_mcmc, save_result

#: Per-process cache, so a joblib worker loads the model only once.
_SETUP_CACHE: Dict[Tuple, object] = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument(
        "--lambdas", type=float, nargs="+", required=True, help="Bias strengths to sweep over"
    )
    parser.add_argument("--n-runs", type=int, default=5, help="Independent chains per lambda (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; chain i uses seed + i")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers (joblib; default: -1)")
    parser.add_argument(
        "--ungapped-fasta", action="store_true", help="Write sequences.fasta without alignment gaps"
    )
    return parser


def _cached_setup(args):
    key = (args.model, args.wt, args.mapping, args.alphabet, args.allowed_sites, args.allowed_cols,
           args.critic_site, args.critic_col, args.mapping_base, args.device, args.dtype)
    if key not in _SETUP_CACHE:
        _SETUP_CACHE[key] = prepare_run(args, verbose=False)
    return _SETUP_CACHE[key]


def _run_one(args, lam: float, run_index: int) -> dict:
    setup = _cached_setup(args)
    config = MCMCConfig(
        lam=lam,
        beta=args.beta,
        threshold=args.threshold,
        max_steps=args.max_steps,
        record_every=args.record_every,
        seed=args.seed + run_index,
        target_residue=args.target_residue,
    )
    result = run_mcmc(
        model=setup.model,
        start_sequence=setup.wt_sequence,
        critic_col=setup.critic_col,
        allowed_cols=setup.allowed_cols,
        config=config,
    )
    output_dir = Path(args.output) / lam_dirname(lam) / f"run{run_index}"
    save_result(result, output_dir, ungapped=args.ungapped_fasta)
    print(
        f"lam={lam:g} run{run_index}: {'converged' if result.converged else 'max steps'} "
        f"after {result.accepted_steps} accepted steps, "
        f"P(S)={math.exp(result.final_logP):.4f}, E={result.final_energy:.2f}",
        flush=True,
    )
    return {
        "lam": lam,
        "run": run_index,
        "seed": config.seed,
        "converged": result.converged,
        "proposed_steps": result.proposed_steps,
        "accepted_steps": result.accepted_steps,
        "acceptance_rate": result.acceptance_rate,
        "first_crossing_step": result.first_crossing_step,
        "final_E_DCA": result.final_energy,
        "final_P_S": math.exp(result.final_logP),
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = build_parser().parse_args()

    # Validate the inputs once, in the parent, and report the setup.
    prepare_run(args)
    jobs = [(lam, run_index) for lam in args.lambdas for run_index in range(args.n_runs)]
    print(f"\nRunning {len(jobs)} chains ({len(args.lambdas)} lambdas x {args.n_runs} runs) "
          f"on {args.n_jobs} worker(s)\n")

    rows = Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(_run_one)(args, lam, run_index) for lam, run_index in jobs
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows).sort_values(["lam", "run"]).reset_index(drop=True)
    summary.to_csv(output / "sweep_summary.csv", index=False)

    print("\nconverged chains per lambda:")
    print(summary.groupby("lam")["converged"].agg(["sum", "count"]).to_string())
    print(f"\nsummary written to {output / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
