#!/usr/bin/env python
"""Distance from each sequence to the nearest natural sequence that carries the target residue.

A design's energy says how plausible the model finds it; it does not say whether
anything like it exists. This adds that second question as a per-sequence column:
the Hamming distance to the closest member of the natural alignment **restricted
to the sequences that already have the target residue at the critic site** -- the
family's own chymotrypsin-like pocket, as it were. A design sitting close to that
subset is a variation on something nature has; one sitting far from it is not.

The column is written into the ``observables.csv`` of every lambda, so that
``plot_ensemble.py --quantities ... min_hamming_natural_S_pct`` picks it up like
any other quantity.

``--allowed-cols`` restricts the comparison to the columns the sampler was allowed
to move, and ``--as-percentage`` reports it as a share of them. That is the
meaningful version: outside those columns a design *is* the wild type, so counting
the rest adds the same large constant -- the wild type's own distance from the
family -- to every design, and hides what distinguishes them.

With ``--natural-observables`` the same column is added to the natural
observables as well, to give the figure a reference cloud. A natural sequence that
is itself in the subset would otherwise be at distance zero from itself, so its own
row is excluded from its minimum; the subsample is reconstructed from the seed and
size that ``calibrate_mu.py`` used, and the names are verified row by row.

Example::

    python scripts/add_natural_neighbour.py \\
        --ensemble-dir ../results_serine/ensembles \\
        --natural ../iter_aln_dedup.faa --critic-col 197 \\
        --natural-observables ../results/calibration/natural_observables.csv \\
        --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import _SRC  # noqa: F401  (puts src/ on sys.path)

from encoding import Alphabet, read_sequences, read_single_sequence  # noqa: E402
from observables import min_hamming_to_set  # noqa: E402
from sites import mutable_columns, parse_column_spec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-dir", required=True, help="Directory of lam*/final_sequences.fasta")
    parser.add_argument("--natural", required=True, help="The natural alignment, aligned FASTA")
    parser.add_argument("--critic-col", type=int, required=True, help="MSA column of the critic site (0-based)")
    parser.add_argument("--target-residue", default="S", help="Residue the natural subset must carry (default: S)")
    parser.add_argument(
        "--natural-observables",
        default=None,
        help="natural_observables.csv to receive the same column (its own row excluded from each minimum)",
    )
    parser.add_argument(
        "--natural-subsample-seed",
        type=int,
        default=0,
        help="The --seed calibrate_mu.py used for its subsample (default: 0)",
    )
    parser.add_argument(
        "--natural-subsample-size",
        type=int,
        default=20000,
        help="The --max-sequences calibrate_mu.py used (default: 20000)",
    )
    parser.add_argument(
        "--allowed-cols",
        default=None,
        help="Restrict the distance to these MSA columns, 0-based (e.g. '180-220'): the columns the "
        "sampler was allowed to mutate. Everywhere else a design is the wild type by construction, so "
        "counting those positions only measures how far the wild type is from the family",
    )
    parser.add_argument(
        "--wt", default=None, help="Aligned wild-type FASTA; its gap columns are dropped from --allowed-cols"
    )
    parser.add_argument(
        "--as-percentage",
        action="store_true",
        help="Report the distance as a percentage of the columns compared, rather than a count",
    )
    parser.add_argument("--column", default=None, help="Column name (default: min_hamming_natural_<residue>)")
    parser.add_argument("--alphabet", default="protein", help="adabmDCA alphabet (default: protein)")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    alphabet = Alphabet.from_name(args.alphabet)
    column = args.column or (
        f"min_hamming_natural_{args.target_residue}_pct"
        if args.as_percentage
        else f"min_hamming_natural_{args.target_residue}"
    )
    critic = args.critic_col

    records = read_sequences(args.natural)
    names = np.array([name for name, _ in records])
    natural = np.stack([alphabet.encode(sequence) for _, sequence in records])
    if not 0 <= critic < natural.shape[1]:
        raise SystemExit(f"The critic column {critic} is outside the alignment (L={natural.shape[1]})")

    compared = _compared_columns(args, alphabet, natural.shape[1])
    target_state = alphabet.index(args.target_residue)
    in_subset = np.flatnonzero(natural[:, critic] == target_state)
    if in_subset.size == 0:
        raise SystemExit(f"No natural sequence carries {args.target_residue} at column {critic}")
    subset = natural[in_subset]
    print(
        f"Natural alignment : {natural.shape[0]} sequences of length {natural.shape[1]}\n"
        f"Reference subset  : {subset.shape[0]} with {args.target_residue} at column {critic} "
        f"({100 * subset.shape[0] / natural.shape[0]:.1f} %)\n"
        f"Compared over     : {compared.size} columns"
        + (", as a percentage of them" if args.as_percentage else "")
    )
    unit = "%" if args.as_percentage else ""

    for directory in sorted(Path(args.ensemble_dir).glob("lam*")):
        fasta = directory / "final_sequences.fasta"
        observables = directory / "observables.csv"
        if not directory.is_dir() or not fasta.is_file() or not observables.is_file():
            continue
        encoded = np.stack([alphabet.encode(sequence) for _, sequence in read_sequences(fasta)])
        distance, _ = min_hamming_to_set(encoded[:, compared], subset[:, compared], device=args.device)
        distance = _scale(distance, compared.size, args.as_percentage)
        frame = pd.read_csv(observables)
        if len(frame) != len(distance):
            raise SystemExit(f"{observables} has {len(frame)} rows for {len(distance)} sequences")
        frame[column] = distance
        frame.to_csv(observables, index=False)
        print(
            f"  {directory.name}: median {np.median(distance):.1f}{unit}  "
            f"min {distance.min():.1f}{unit}  max {distance.max():.1f}{unit}  -> {observables}"
        )

    if args.natural_observables:
        _annotate_natural(args, names, natural, subset, in_subset, column, compared)


def _compared_columns(args, alphabet, length: int) -> np.ndarray:
    """The columns the distance is measured over: all of them, or the mutable ones.

    A design differs from the wild type only where the sampler was allowed to
    move, so counting the other columns adds the wild type's own distance from the
    family to every design equally -- a large constant that hides the differences
    between them.
    """
    if args.allowed_cols is None:
        return np.arange(length, dtype=np.int64)
    columns = sorted(set(parse_column_spec(args.allowed_cols, length=length)))
    if args.wt:
        _, wt_string = read_single_sequence(args.wt)
        wt_sequence = alphabet.encode(wt_string)
        kept = mutable_columns(columns, [wt_sequence], alphabet)
        dropped = [c for c in columns if c not in kept]
        if dropped:
            print(f"Skipping gap positions of the wild type: {dropped}")
        columns = kept
    if not columns:
        raise SystemExit(f"No column left from --allowed-cols {args.allowed_cols!r}")
    return np.asarray(columns, dtype=np.int64)


def _scale(distance: np.ndarray, n_columns: int, as_percentage: bool) -> np.ndarray:
    return 100.0 * distance / n_columns if as_percentage else distance


def _annotate_natural(args, names, natural, subset, in_subset, column: str, compared: np.ndarray) -> None:
    """Add the column to the natural observables, excluding each sequence's own row."""
    path = Path(args.natural_observables)
    frame = pd.read_csv(path)
    index = np.arange(natural.shape[0])
    if args.natural_subsample_size and natural.shape[0] > args.natural_subsample_size:
        index = np.random.default_rng(args.natural_subsample_seed).choice(
            natural.shape[0], args.natural_subsample_size, replace=False
        )
        index.sort()
    if len(frame) != len(index) or not np.array_equal(frame["name"].to_numpy(), names[index]):
        raise SystemExit(
            f"{path} does not match the subsample reconstructed with seed "
            f"{args.natural_subsample_seed} and size {args.natural_subsample_size}; "
            "pass the values calibrate_mu.py was run with"
        )

    # For a sequence that is itself in the reference subset, its own row is the
    # trivial nearest neighbour and is excluded from the minimum.
    position_in_subset = -np.ones(natural.shape[0], dtype=np.int64)
    position_in_subset[in_subset] = np.arange(in_subset.size)
    distance, _ = min_hamming_to_set(
        natural[index][:, compared], subset[:, compared], device=args.device,
        exclude=position_in_subset[index],
    )
    distance = _scale(distance, compared.size, args.as_percentage)
    frame[column] = distance
    frame.to_csv(path, index=False)
    carriers = int((position_in_subset[index] >= 0).sum())
    print(
        f"  natural: median {np.median(distance):.1f}  min {distance.min():.1f}  max {distance.max():.1f}  "
        f"({carriers} of {len(index)} are themselves in the subset) -> {path}"
    )


if __name__ == "__main__":
    main()
