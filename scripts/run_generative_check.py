#!/usr/bin/env python
"""Unbiased resampling of the model, scored against the statistics it was fitted on.

This is the generative sanity check of the Potts model itself, not a design run:
``lambda = 0`` (no bias), ``beta = 1`` (the temperature the model was fitted at),
every non-gap column mutable. A batch of chains is started from the wild type and
followed as it equilibrates, and at each checkpoint the batch's statistics are
compared with the **reweighted** statistics of the natural alignment:

* ``pearson_f1``  -- Pearson coefficient of the single-site frequencies;
* ``pearson_f2``  -- of the two-site frequencies, off-diagonal blocks only;
* ``pearson_cij`` -- of the connected correlations
  ``C_ij(a,b) = f_ij(a,b) - f_i(a) f_j(b)``, ``i != j``, with the slope of the fit.

A model that reproduces its training statistics drives all three toward 1 and the
slope toward 1, while the per-sequence observables (energy, context entropy)
converge on the weighted natural means. Everything is computed with adabmDCA's own
functions; see :mod:`natural_stats`.

The chains never mutate a gap column and never propose a gap, so they keep the
wild type's gap pattern throughout. The gap state is therefore one they cannot
sample, and every coefficient is also reported over the non-gap states alone
(``*_nogap``), which is the fair comparison.

Example::

    python scripts/run_generative_check.py \\
        --wt data/wt_trypsin.fasta --model ../bmDCA_params.dat \\
        --natural ../iter_aln_dedup.faa --weights ../bmDCA_weights.dat \\
        --output ../results_generative/ --critic-col 197 --allowed-cols 0-259 \\
        --n-chains 1000 --n-steps 520000 --trace-every 8000 --trace-schedule log \\
        --device cuda --cache-model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import add_common_arguments, prepare_run
from encoding import read_sequences, write_sequences
from ensemble import EnsembleConfig, run_ensemble, run_ensemble_gibbs
from natural_stats import NaturalStatistics, load_weights, one_hot, weighted_mean, weighted_std
from observables import context_dependent_entropy, describe, energies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--natural", required=True, help="The training alignment, aligned FASTA")
    parser.add_argument(
        "--weights",
        default=None,
        help="adabmDCA weights file, one float per alignment row (default: uniform weights)",
    )
    parser.add_argument("--lam", type=float, default=0.0, help="Bias strength; 0 is the unbiased check")
    parser.add_argument(
        "--sampler",
        default="metropolis",
        choices=("metropolis", "gibbs"),
        help="'metropolis' is this repository's single-site sampler; 'gibbs' is adabmDCA's own "
        "gibbs_step_uniform_sites, which resamples a site from its exact conditional and is the "
        "sampler the model was trained with. Gibbs requires --lam 0 and frees every site and state",
    )
    parser.add_argument(
        "--no-gap-moves",
        action="store_true",
        help="Keep the gap inert, as a design run does. The chains then keep the gap pattern they start "
        "from, which confines them to a subspace of the one the model was fitted on; the default is to "
        "let the gap be proposed and mutated like any other state",
    )
    parser.add_argument(
        "--start",
        default="wt",
        choices=("wt", "natural", "random"),
        help="Where the chains start: all at the wild type (default), one natural sequence each drawn "
        "under the alignment's weights, or a uniformly random sequence each. The natural start is the "
        "control that separates a model which cannot reproduce its statistics from chains that have not "
        "mixed: chains started inside the family either stay there or drift out, and that is the answer. "
        "The random start approaches the same equilibrium from the opposite side, from sequences the "
        "model finds absurd; wt, natural and random agreeing is the strongest mixing evidence available",
    )
    parser.add_argument("--n-chains", type=int, default=1000, help="Chains (default: 1000)")
    parser.add_argument("--n-steps", type=int, default=520_000, help="Proposed moves per chain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-every", type=int, default=8000, help="Checkpoint spacing (0 = off)")
    parser.add_argument("--trace-schedule", default="log", choices=("linear", "log"))
    parser.add_argument(
        "--pseudo-count", type=float, default=0.0, help="Pseudo count in the frequencies (default: 0, none)"
    )
    parser.add_argument(
        "--natural-max-sequences",
        type=int,
        default=0,
        help="Score at most this many natural sequences for the per-sequence means (0 = all)",
    )
    parser.add_argument("--chunk", type=int, default=256, help="Sequences scored per chunk")
    parser.add_argument(
        "--ceiling-repeats",
        type=int,
        default=5,
        help="Draws of --n-chains natural sequences used to measure the finite-sample ceiling "
        "of each coefficient (0 = skip). A batch of M chains estimates f_ij from M samples, so "
        "even a perfect model cannot reach a Pearson of 1 at finite M",
    )
    parser.add_argument(
        "--progress-every", type=int, default=20_000, help="Print a progress line every N steps (0 = quiet)"
    )
    return parser


def _sampling_ceiling(
    statistics: NaturalStatistics,
    natural: np.ndarray,
    weights: np.ndarray | None,
    n_chains: int,
    repeats: int,
    model,
    device,
    seed: int,
) -> dict:
    """How well ``n_chains`` *natural* sequences reproduce the natural statistics.

    A batch of ``M`` chains estimates ``f_ij`` from ``M`` samples, and that
    estimate is noisy: the Pearson coefficient a perfect sampler could reach at
    finite ``M`` is below 1, the more so for the connected correlations, whose
    entries are small. Drawing ``M`` sequences from the natural alignment itself
    -- under the same weights, so the draw follows the same distribution --
    measures that ceiling directly, and is the honest target for the curves.
    """
    if repeats <= 0:
        return {}
    probabilities = None if weights is None else np.asarray(weights, dtype=np.float64) / float(np.sum(weights))
    generator = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        index = generator.choice(natural.shape[0], size=n_chains, replace=False, p=probabilities)
        batch = one_hot(torch.as_tensor(natural[index], device=device), model.q, model.bias.dtype)
        draws.append(statistics.compare(batch))
    return {
        name: {
            "mean": float(np.mean([draw[name] for draw in draws])),
            "std": float(np.std([draw[name] for draw in draws])),
        }
        for name in draws[0]
    }


def _natural_start(
    natural: np.ndarray,
    weights: np.ndarray | None,
    setup,
    n_chains: int,
    seed: int,
    gap_index: int,
    allow_gap_moves: bool,
):
    """``n_chains`` natural sequences, made compatible with the mutable columns.

    The sampler never proposes a gap and never mutates a gap column, so a chain
    keeps the gap pattern it starts from and that pattern must leave every mutable
    column non-gap. Only a handful of natural sequences satisfy that exactly, so
    the gaps a drawn sequence has at mutable columns are filled with the wild
    type's residue there -- a median of about nine columns out of several hundred.
    The sequence is otherwise untouched, which is all this control needs.
    """
    columns = np.asarray(setup.allowed_cols, dtype=np.int64)
    probabilities = None if weights is None else np.asarray(weights, float) / float(np.sum(weights))
    index = np.random.default_rng(seed).choice(
        natural.shape[0], size=n_chains, replace=False, p=probabilities
    )
    start = natural[index].copy()
    if allow_gap_moves:
        print(f"Start             : {n_chains} natural sequences, unmodified")
        return start, {"start": "natural", "gap_positions_filled": 0}
    is_gap = start[:, columns] == gap_index
    filled = int(is_gap.sum())
    rows, offsets = np.nonzero(is_gap)
    start[rows, columns[offsets]] = setup.wt_sequence[columns[offsets]]
    print(
        f"Start             : {n_chains} natural sequences, "
        f"{filled} gap positions filled from the wild type "
        f"({filled / n_chains:.1f} per sequence, of {columns.size} mutable columns)"
    )
    return start, {
        "start": "natural",
        "gap_positions_filled": filled,
        "gap_positions_filled_per_chain": filled / n_chains,
    }


def _random_start(
    setup,
    n_chains: int,
    seed: int,
    q: int,
    gap_index: int,
    allow_gap_moves: bool,
    sampler: str,
):
    """``n_chains`` uniformly random sequences over the columns the sampler can move.

    The opposite control to the natural start: sequences the model finds absurd,
    carrying none of the family's structure. Columns the sampler cannot move keep
    the wild type's residue, since a chain would never leave whatever is put there.
    The gap is drawn like any other state when the chains are free to mutate it,
    and excluded when they are not -- a gap at a mutable column would otherwise be
    frozen there for the whole run.
    """
    generator = np.random.default_rng(seed)
    columns = (
        np.arange(setup.model.L, dtype=np.int64)
        if sampler == "gibbs"
        else np.asarray(setup.allowed_cols, dtype=np.int64)
    )
    states = np.arange(q) if allow_gap_moves else np.array([s for s in range(q) if s != gap_index])
    start = np.broadcast_to(setup.wt_sequence, (n_chains, setup.model.L)).copy()
    start[:, columns] = generator.choice(states, size=(n_chains, columns.size))
    print(
        f"Start             : {n_chains} uniformly random sequences over {columns.size} columns, "
        f"{states.size} states each" + ("" if allow_gap_moves else " (the gap excluded)")
    )
    return start, {"start": "random", "random_start_columns": int(columns.size),
                   "random_start_states": int(states.size)}


def main() -> None:
    args = build_parser().parse_args()
    allow_gap_moves = not args.no_gap_moves or args.sampler == "gibbs"
    if args.sampler == "gibbs" and args.lam != 0.0:
        raise SystemExit("--sampler gibbs is only correct for the unbiased ensemble (--lam 0)")
    setup = prepare_run(args, drop_gap_columns=not allow_gap_moves)
    model = setup.model
    alphabet = setup.alphabet
    device = model.bias.device
    target_state = alphabet.index(args.target_residue)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- natural
    records = read_sequences(args.natural)
    natural = np.stack([alphabet.encode(sequence) for _, sequence in records])
    if natural.shape[1] != model.L:
        raise SystemExit(f"The alignment has L={natural.shape[1]}, the model has L={model.L}")
    weights = load_weights(args.weights, natural.shape[0]) if args.weights else None
    print(f"\nNatural alignment : {natural.shape[0]} sequences")

    statistics = NaturalStatistics.from_alignment(
        natural,
        q=model.q,
        gap_index=alphabet.gap_index,
        weights=weights,
        device=device,
        dtype=model.bias.dtype,
        pseudo_count=args.pseudo_count,
    )
    print(
        f"Weights           : {'uniform' if weights is None else args.weights}  "
        f"(M_eff = sum w = {statistics.effective_sequences:.1f}, "
        f"{100 * statistics.effective_sequences / natural.shape[0]:.1f} % of M; "
        f"Kish ESS {statistics.kish_ess:.1f})"
    )

    scored, scored_weights = natural, weights
    if args.natural_max_sequences and natural.shape[0] > args.natural_max_sequences:
        index = np.random.default_rng(args.seed).choice(
            natural.shape[0], args.natural_max_sequences, replace=False
        )
        index.sort()
        scored = natural[index]
        scored_weights = None if weights is None else weights[index]
    natural_observables = describe(
        model, scored, reference=setup.wt_sequence, critic_col=setup.critic_col,
        target_state=target_state, beta=args.beta, chunk=args.chunk,
    )
    reference = {
        name: {
            "mean": weighted_mean(values, scored_weights),
            "std": weighted_std(values, scored_weights),
        }
        for name, values in natural_observables.items()
    }
    print(f"Weighted means    : " + "  ".join(
        f"{name} {entry['mean']:.3f}" for name, entry in reference.items()
    ))

    ceiling = _sampling_ceiling(
        statistics, natural, weights, args.n_chains, args.ceiling_repeats, model, device, args.seed
    )
    if ceiling:
        print("Sampling ceiling  : " + "  ".join(
            f"{name.replace('pearson_', '')} {entry['mean']:.4f}"
            for name, entry in ceiling.items()
            if name.startswith("pearson") and not name.endswith("_nogap")
        ))

    # ------------------------------------------------------------------ chains
    start_sequence = setup.wt_sequence
    start_info = {"start": "wt"}
    if args.start == "natural":
        start_sequence, start_info = _natural_start(
            natural, weights, setup, args.n_chains, args.seed, alphabet.gap_index, allow_gap_moves
        )
    elif args.start == "random":
        start_sequence, start_info = _random_start(
            setup, args.n_chains, args.seed, model.q, alphabet.gap_index, allow_gap_moves, args.sampler
        )

    def trace_hook(chains: torch.Tensor, x: torch.Tensor) -> dict:
        return statistics.compare(x)

    config = EnsembleConfig(
        lam=args.lam,
        beta=args.beta,
        n_chains=args.n_chains,
        n_steps=args.n_steps,
        seed=args.seed,
        target_residue=args.target_residue,
        trace_every=args.trace_every,
        trace_cde=True,
        trace_schedule=args.trace_schedule,
        allow_gap_moves=allow_gap_moves,
    )
    columns = model.L if args.sampler == "gibbs" else len(setup.allowed_cols)
    print(
        f"\nlambda = {args.lam:g}, beta = {args.beta:g}: {args.n_chains} chains x {args.n_steps} steps "
        f"over {columns} mutable columns on {args.device}, {args.sampler} sampler"
        + ("  (the gap is an ordinary state)" if allow_gap_moves else "  (the gap is inert)")
    )
    if args.sampler == "gibbs":
        result = run_ensemble_gibbs(
            model=model,
            start_sequence=start_sequence,
            critic_col=setup.critic_col,
            config=config,
            progress_every=args.progress_every,
            trace_hook=trace_hook if args.trace_every else None,
            reference_sequence=setup.wt_sequence,
        )
    else:
        result = run_ensemble(
            model=model,
            start_sequence=start_sequence,
            critic_col=setup.critic_col,
            allowed_cols=setup.allowed_cols,
            config=config,
            progress_every=args.progress_every,
            trace_hook=trace_hook if args.trace_every else None,
            reference_sequence=setup.wt_sequence,
        )

    # ----------------------------------------------------------------- outputs
    observables = describe(
        model, result.sequences, reference=setup.wt_sequence, critic_col=setup.critic_col,
        target_state=target_state, beta=args.beta, chunk=args.chunk,
    )
    frame = pd.DataFrame({"chain": np.arange(result.sequences.shape[0]), **observables})
    frame.to_csv(output / "observables.csv", index=False)
    write_sequences(
        output / "final_sequences.fasta",
        ((f"chain{chain}", alphabet.decode(sequence)) for chain, sequence in enumerate(result.sequences)),
    )
    if result.trace:
        pd.DataFrame(result.trace).to_csv(output / "trace.csv", index=False)

    final = statistics.compare(one_hot(torch.as_tensor(result.sequences, device=device), model.q, model.bias.dtype))
    summary = {
        "lam": args.lam,
        "beta": args.beta,
        "n_chains": args.n_chains,
        "n_steps": args.n_steps,
        "seed": args.seed,
        "acceptance_rate": None if args.sampler == "gibbs" else result.acceptance_rate,
        "n_mutable_columns": columns,
        "sampler": args.sampler,
        "critic_col": setup.critic_col,
        "pseudo_count": args.pseudo_count,
        "allow_gap_moves": allow_gap_moves,
        **start_info,
        "natural": str(args.natural),
        "weights": str(args.weights) if args.weights else None,
        "n_natural": int(natural.shape[0]),
        "n_natural_scored": int(scored.shape[0]),
        "effective_sequences": statistics.effective_sequences,
        "kish_ess": statistics.kish_ess,
        "final": {name: float(values.mean()) for name, values in observables.items()},
        "natural_weighted": reference,
        "pearson_final": final,
        "sampling_ceiling": ceiling,
    }
    with open(output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "\nacceptance n/a (a Gibbs move is always taken)"
        if args.sampler == "gibbs"
        else f"\nacceptance {result.acceptance_rate:.3f}"
    )
    print("final batch vs natural (sampling ceiling in brackets):")
    for name, value in final.items():
        limit = f"   [{ceiling[name]['mean']:+.4f}]" if ceiling and name in ceiling else ""
        print(f"  {name:22s} {value:+.4f}{limit}")
    print("final batch means (natural weighted mean in brackets):")
    for name, values in observables.items():
        print(f"  {name:22s} {values.mean():10.3f}   [{reference[name]['mean']:.3f}]")
    print(f"\nwritten to {output}")


if __name__ == "__main__":
    main()
