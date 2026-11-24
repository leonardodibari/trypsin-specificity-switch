#!/usr/bin/env python
"""Sample a t-ladder by ANNEALING, not from the wild type every time.

WHY THIS EXISTS. ``run_ensemble.py`` starts every (lambda, t) rung at the wild
type. That is fine on the 79-column grafting window -- all 20 of its runs
saturate -- but on the 265-column free runs it fails: a pilot at t = -0.25 run
for 400,000 steps (3.3x the first pass) still did not saturate in
``hamming_to_wt``. The trace shows why: the energetic observables (E_DCA, P_S,
CDE) settle in tens of thousands of steps, but the Hamming distance climbs
MONOTONICALLY and the per-step increment decays roughly logarithmically, not
exponentially -- d is diffusing outward from 0, not relaxing to it. See
``results/EQUILIBRATION_NOTE.md``.

THE FIX. t is the chemical potential conjugate to d, so ``-d<d>/dt = Var(d) >=
0``: the equilibrium <d> is a MONOTONE function of t. Sort the t values from
most restrained to least (large t -> small t -> negative t) and chain the runs:
each rung starts from the PREVIOUS rung's final chains instead of from the wild
type. Every rung then only has to cover the INCREMENT in <d> to its neighbour,
not the whole journey from d = 0 -- which is the only reason the journey was
slow in the first place.

The Hamming distance is still measured to the WILD TYPE throughout, never to a
rung's own starting point: every call passes ``reference_sequence=wt_sequence``
explicitly, because ``run_ensemble`` otherwise defaults the reference to
whatever sequence a chain happens to start this rung from -- which would make
``d`` reset to 0 at every rung and silently break the entire restraint.

Everything downstream (observables.csv, final_sequences.fasta, summary.json,
trace.csv, directory names) is written in exactly the shape run_ensemble.py
writes, so every existing reader (data.py, substitute_critic.py, saturation.py)
works on an annealed run without modification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _common import add_common_arguments, prepare_run  # noqa: E402
from encoding import read_single_sequence, write_sequences  # noqa: E402
from ensemble import EnsembleConfig, run_dirname, run_ensemble  # noqa: E402
from observables import describe  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--lam", type=float, required=True, help="The one lambda of this chain")
    parser.add_argument(
        "--t-values", type=float, nargs="+", required=True,
        help="Every t to sample, IN ANY ORDER -- they are re-sorted descending "
        "(most restrained first) before chaining, since that is the direction "
        "of increasing <d> and the direction a warm start helps in",
    )
    parser.add_argument(
        "--t-indel", type=float, default=None,
        help="Price of an indel, held FIXED across the rungs while --t-values anneals the "
        "price of a substitution. Unset, an indel costs the same as a substitution at "
        "every rung, which is what the ladder meant before the two were split. Annealing "
        "the substitution budget alone is the point of separating them: the gap pattern "
        "then stays under one fixed price while <d_sub> is walked",
    )
    parser.add_argument("--mu", default=None, help="Q = E_DCA - mu*CDE calibration, number or mu.json path")
    parser.add_argument("--n-chains", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=60_000, help="Steps PER RUNG")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; rung i uses seed + i")
    parser.add_argument("--trace-every", type=int, default=1000)
    parser.add_argument("--trace-schedule", default="linear", choices=("linear", "log"))
    parser.add_argument("--no-trace-cde", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--ungapped-fasta", action="store_true")
    parser.add_argument(
        "--start-fasta", default=None,
        help="FASTA with ONE aligned sequence to start the first rung from, instead of the "
        "wild type -- the far arm of the two-start bracket (e.g. data/chymotrypsin_bovine.fasta, "
        "which sits at d ~ 150 from the wild type). The Hamming distance is STILL measured to "
        "the wild type: only where the chains begin changes, never what d means.",
    )
    parser.add_argument(
        "--anneal-order", default="auto", choices=("auto", "descending", "ascending"),
        help="Which end of the t ladder to sample first. 'auto' (default) picks the rung whose "
        "equilibrium <d> is nearest the start: descending (most restrained first) from a "
        "wild-type start at d = 0, ascending (least restrained first) from a far start. Annealing "
        "only helps when the first rung is the one the start is already close to.",
    )
    return parser


def _resolve_mu(value: str | None) -> float | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        with open(path, encoding="utf-8") as handle:
            return float(json.load(handle)["mu"])
    return float(value)


def main() -> None:
    args = build_parser().parse_args()
    setup = prepare_run(args)
    alphabet = setup.alphabet
    target_state = alphabet.index(args.target_residue)
    against_state = alphabet.index(args.against_residue)
    mu = _resolve_mu(args.mu)

    wt_sequence = setup.wt_sequence  # the fixed reference for d_H, every rung
    allowed_cols = setup.allowed_cols

    # Where the chains begin. The wild type (d = 0) is the near arm of the
    # two-start bracket; --start-fasta is the far arm, a natural chymotrypsin at
    # d ~ 150. The two approach the d axis from opposite sides, so AGREEMENT
    # between them is a sharper equilibrium test than any single-run drift
    # criterion, and disagreement bounds the residual error honestly
    # (results/EQUILIBRATION_NOTE.md, fix 2).
    if args.start_fasta is not None:
        start_name, start_string = read_single_sequence(args.start_fasta)
        start_sequence = alphabet.encode(start_string)
        if start_sequence.shape[0] != setup.model.L:
            raise SystemExit(
                f"--start-fasta is {start_sequence.shape[0]} columns, the model is {setup.model.L}: "
                "it must be the ALIGNED sequence, not the raw one"
            )
        start_d = int((start_sequence != wt_sequence).sum())
        print(f"start: {start_name} from {args.start_fasta}, d_H to the wild type = {start_d}")
    else:
        start_name, start_sequence = setup.wt_name, wt_sequence
        start_d = 0
        print(f"start: the wild type ({start_name}), d_H = 0")

    # <d>(t) is monotone DECREASING in t, since -d<d>/dt = Var(d) >= 0. So the
    # rung nearest a d = 0 start is the most positive t, and the rung nearest a
    # far start is the most negative one. Sampling in that order is the whole
    # point: each rung then only has to cover the increment in <d> to its
    # neighbour, never the journey from the start.
    order = args.anneal_order
    if order == "auto":
        order = "descending" if start_d <= setup.model.L // 4 else "ascending"
    t_sorted = sorted(set(args.t_values), reverse=(order == "descending"))
    print(f"annealed chain ({order}): lambda={args.lam:g}, t = "
          + " -> ".join(f"{t:g}" for t in t_sorted))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, t in enumerate(t_sorted):
        config = EnsembleConfig(
            lam=args.lam, t=t, t_indel=args.t_indel, mu=mu, beta=args.beta,
            n_chains=args.n_chains, n_steps=args.n_steps, seed=args.seed + index,
            target_residue=args.target_residue,
            trace_every=args.trace_every, trace_cde=not args.no_trace_cde,
            trace_schedule=args.trace_schedule, allow_gap_moves=args.allow_gap_moves,
            bias=args.bias, against_residue=args.against_residue,
        )
        print(f"\nrung {index + 1}/{len(t_sorted)}: lambda={args.lam:g}, t_sub={t:g}"
              + ("" if args.t_indel is None else f", t_indel={args.t_indel:g}") + "  "
              f"({args.n_chains} chains x {args.n_steps} steps, {args.device})")
        result = run_ensemble(
            model=setup.model,
            start_sequence=start_sequence,
            critic_col=setup.critic_col,
            allowed_cols=allowed_cols,
            config=config,
            progress_every=args.progress_every,
            # ALWAYS the wild type. Leaving this out would default the
            # reference to `start_sequence`, which after rung 1 is NOT the
            # wild type, and d_H would silently stop meaning what every other
            # script in this repository assumes it means.
            reference_sequence=wt_sequence,
        )
        start_sequence = result.sequences  # the warm start for the next, colder rung

        observables = describe(
            setup.model, result.sequences, reference=wt_sequence,
            critic_col=setup.critic_col, target_state=target_state,
            beta=args.beta, against_state=against_state,
        )
        frame = pd.DataFrame({"chain": np.arange(result.sequences.shape[0]), **observables})
        frame.insert(1, "lam", args.lam)
        frame.insert(2, "t", t)

        directory = output / run_dirname(args.lam, t, args.t_indel)
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / "observables.csv", index=False)
        write_sequences(
            directory / "final_sequences.fasta",
            ((f"{run_dirname(args.lam, t, args.t_indel)}_chain{c}",
              _maybe_ungap(alphabet.decode(seq), args.ungapped_fasta))
             for c, seq in enumerate(result.sequences)),
        )
        if result.trace:
            pd.DataFrame(result.trace).to_csv(directory / "trace.csv", index=False)
        summary = {
            "lam": args.lam, "t": t, "mu": mu, "beta": args.beta,
            "n_chains": args.n_chains, "n_steps": args.n_steps, "seed": config.seed,
            "acceptance_rate": result.acceptance_rate, "critic_col": result.critic_col,
            "n_allowed_sites": len(result.allowed_cols), "allowed_cols": result.allowed_cols,
            "allow_gap_moves": bool(args.allow_gap_moves), "bias": args.bias,
            "against_residue": args.against_residue,
            "annealed": True, "annealed_rung_index": index, "annealed_order": t_sorted,
            "annealed_direction": order,
            # The arm of the two-start bracket this run belongs to, and how far
            # from the wild type its chains began. A figure that compares the two
            # arms must be able to tell them apart from the run alone.
            "start_name": start_name, "start_fasta": args.start_fasta,
            "start_hamming_to_wt": start_d,
            "mean_E_DCA": float(frame["E_DCA"].mean()),
            "mean_n_gaps": float(frame["n_gaps"].mean()),
            "mean_P_S": float(frame["P_S"].mean()),
            "mean_hamming_to_wt": float(frame["hamming_to_wt"].mean()),
            "mean_CDE": float(frame["CDE"].mean()),
            "mean_LOR": float(frame["LOR"].mean()) if "LOR" in frame else None,
            "var_hamming_to_wt": float(frame["hamming_to_wt"].var(ddof=1)),
        }
        if mu is not None:
            summary["mean_Q"] = float((frame["E_DCA"] - mu * frame["CDE"]).mean())
        with open(directory / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        rows.append(summary)
        print(f"  acceptance {result.acceptance_rate:.3f} | <E> {summary['mean_E_DCA']:.1f} | "
              f"<P(S)> {summary['mean_P_S']:.3f} | <hamming> {summary['mean_hamming_to_wt']:.1f} "
              f"(var {summary['var_hamming_to_wt']:.1f}) | <CDE> {summary['mean_CDE']:.3f}")

    pd.DataFrame(rows).to_csv(output / "ensemble_summary.csv", index=False)
    print(f"\nwritten to {output}")


def _maybe_ungap(sequence: str, ungapped: bool) -> str:
    return sequence.replace("-", "") if ungapped else sequence


if __name__ == "__main__":
    main()
