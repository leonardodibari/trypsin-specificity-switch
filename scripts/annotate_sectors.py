"""Where did the mutations land -- the red sector, the grafting segment, or neither?

Two things are annotated, from the same sector map:

**Endpoints.** For every ``lam*/final_sequences.fasta`` of an ensemble run, each
chain's final sequence is compared with the wild type and its substitutions are
split by sector. The counts (``n_mut_red``, ``n_mut_grafting``, ``n_mut_other``,
``n_mut_total``, and ``gaps_opened`` / ``gaps_closed``) are written into that
lambda's ``observables.csv``, so every plotting script can cross them with the
energy and with ``P(S)`` like any other quantity.

**Trajectories.** For every ``trajectory.csv`` of a sweep, the column each move
changed is labelled with its sector and its PDB-style residue name, written
beside the original as ``trajectory_sectors.csv``.

Both also get a summary: ``sector_enrichment.csv`` compares the mutations a
sector received with its share of the *mutable* columns, which is the only
comparison that means anything when the run was restricted to a window; and
``column_frequency.csv`` gives the per-column mutation rate, which is what shows
whether an unrestricted run rediscovers the red sector on its own.

Usage:

    python scripts/annotate_sectors.py \\
        --sectors ../work/data/sectors_iter_aln_v2.csv \\
        --wt ../work/data/wt_trypsin.fasta \\
        --ensemble-dir ../results_v2/ensembles \\
        --sweep-dir ../results_v2/sweep \\
        --output ../results_v2/sectors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from encoding import Alphabet, read_sequences, read_single_sequence  # noqa: E402
from sectors import (  # noqa: E402
    SECTORS,
    Sectors,
    annotate_trajectory,
    enrichment,
    mutation_counts,
    per_column_frequency,
)

#: Columns this script owns in an ``observables.csv``; rewritten on every run.
OWNED = [f"n_mut_{name}" for name in SECTORS] + [
    "n_mut_total",
    "gaps_opened",
    "gaps_closed",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sectors", required=True, help="CSV with msa_col, sector, pdb_residue, red_rank")
    parser.add_argument("--wt", required=True, help="FASTA with the aligned wild-type sequence")
    parser.add_argument("--ensemble-dir", default=None, help="Directory of lam*/final_sequences.fasta")
    parser.add_argument("--sweep-dir", default=None, help="Directory of lam*/run*/trajectory.csv")
    parser.add_argument("--output", default=None, help="Where the summaries go (default: alongside the inputs)")
    parser.add_argument(
        "--alphabet",
        default="protein",
        help="adabmDCA alphabet: 'protein' (default), 'dna', 'rna' or explicit tokens",
    )
    parser.add_argument(
        "--mutable-cols",
        default=None,
        help="MSA columns the sampler could move, 0-based (e.g. '165-243'); by default "
        "they are read from each run's summary.json, and failing that every column is "
        "assumed mutable",
    )
    return parser


def _lam_dirs(root: Path) -> List[Path]:
    return sorted((d for d in root.glob("lam*") if d.is_dir()), key=lambda d: _lam_value(d))


def _lam_value(directory: Path) -> float:
    try:
        return float(directory.name[3:])
    except ValueError:
        return float("inf")


def _mutable_from_summary(directory: Path) -> Optional[List[int]]:
    path = directory / "summary.json"
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    columns = payload.get("allowed_cols")
    return [int(c) for c in columns] if columns else None


def annotate_ensembles(
    root: Path,
    sectors: Sectors,
    wt: np.ndarray,
    alphabet: Alphabet,
    output: Path,
    override: Optional[List[int]],
) -> pd.DataFrame:
    """Split each chain's final substitutions by sector; summarise per lambda."""
    summaries, columns_frames = [], []
    directories = _lam_dirs(root)
    if not directories:
        raise SystemExit(f"No lam* directory under {root}")

    for directory in directories:
        fasta = directory / "final_sequences.fasta"
        observables = directory / "observables.csv"
        if not fasta.is_file() or not observables.is_file():
            print(f"  skipping {directory.name}: no final_sequences.fasta / observables.csv")
            continue
        encoded = np.stack([alphabet.encode(sequence) for _, sequence in read_sequences(fasta)])
        counts = mutation_counts(encoded, wt, sectors, gap_index=alphabet.gap_index)

        frame = pd.read_csv(observables)
        if len(frame) != len(counts):
            raise SystemExit(
                f"{observables} has {len(frame)} rows but {fasta} has {len(encoded)} sequences"
            )
        frame = frame.drop(columns=[c for c in OWNED if c in frame.columns])
        frame = pd.concat([frame, counts], axis=1)
        frame.to_csv(observables, index=False)

        mutable = override or _mutable_from_summary(directory)
        if mutable is None:
            # Falling back to "every column is mutable" normalises the sector
            # shares against the whole alignment. For a restricted run that
            # silently inflates every ratio by L / n_allowed -- a depleted
            # sector can come out looking enriched. Refuse rather than guess.
            raise SystemExit(
                f"{directory}/summary.json carries no 'allowed_cols', so the mutable set "
                f"is unknown and sector enrichment would be normalised against all "
                f"{sectors.L} columns. Pass --mutable-cols explicitly (e.g. --mutable-cols "
                f"165-243), or re-run the step that wrote this directory."
            )
        table = enrichment(counts, sectors, mutable=mutable)
        table.insert(0, "lam", _lam_value(directory))
        summaries.append(table)

        frequency = per_column_frequency(encoded, wt, sectors, gap_index=alphabet.gap_index)
        frequency.insert(0, "lam", _lam_value(directory))
        columns_frames.append(frequency)

        shares = ", ".join(
            f"{name} {counts[f'n_mut_{name}'].mean():.2f}" for name in SECTORS
        )
        gaps = (
            f", gaps +{counts['gaps_opened'].mean():.2f}/-{counts['gaps_closed'].mean():.2f}"
            if "gaps_opened" in counts
            else ""
        )
        print(f"  {directory.name}: mean mutations per chain -- {shares}{gaps}")

    if not summaries:
        raise SystemExit(f"Nothing annotated under {root}")
    combined = pd.concat(summaries, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "sector_enrichment.csv", index=False)
    pd.concat(columns_frames, ignore_index=True).to_csv(output / "column_frequency.csv", index=False)
    return combined


def annotate_sweeps(root: Path, sectors: Sectors, output: Path) -> pd.DataFrame:
    """Label every accepted move of every trajectory with the sector it touched."""
    rows = []
    paths = sorted(root.glob("lam*/**/trajectory.csv"))
    if not paths:
        raise SystemExit(f"No lam*/**/trajectory.csv under {root}")

    for path in paths:
        frame = pd.read_csv(path)
        annotated = annotate_trajectory(frame, sectors)
        annotated.to_csv(path.with_name("trajectory_sectors.csv"), index=False)

        moves = annotated[annotated["sector"] != ""]
        record = {
            "lam": _lam_value(path.parents[1] if path.parent.name.startswith("run") else path.parent),
            "run": path.parent.name,
            "path": str(path.parent),
            "recorded_moves": int(len(moves)),
        }
        for name in SECTORS:
            record[f"n_mut_{name}"] = int((moves["sector"] == name).sum())
        rows.append(record)

    combined = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "trajectory_sector_counts.csv", index=False)

    by_lam = combined.groupby("lam")[[f"n_mut_{name}" for name in SECTORS]].mean().reset_index()
    by_lam.to_csv(output / "trajectory_sector_means.csv", index=False)
    print("\nRecorded moves per chain, by sector (mean over runs):")
    print(by_lam.to_string(index=False))
    return combined


def main() -> None:
    args = build_parser().parse_args()
    if not args.ensemble_dir and not args.sweep_dir:
        raise SystemExit("Give at least one of --ensemble-dir / --sweep-dir")

    alphabet = Alphabet.from_name(args.alphabet)
    _, wt_string = read_single_sequence(args.wt)
    wt = alphabet.encode(wt_string)
    sectors = Sectors.load(args.sectors, length=wt.shape[0])
    if sectors.L != wt.shape[0]:
        raise SystemExit(
            f"The sector map covers {sectors.L} columns but the wild type has {wt.shape[0]}"
        )

    override = None
    if args.mutable_cols:
        from sites import parse_column_spec

        override = sorted(set(parse_column_spec(args.mutable_cols, length=sectors.L)))

    print(f"Sector map: {sectors.source}  sizes {sectors.sizes()}")
    print(
        "Red sector, most to least important: "
        + ", ".join(f"{sectors.label_of[c]}(col {c})" for c in sectors.red_order())
    )

    if args.ensemble_dir:
        root = Path(args.ensemble_dir)
        output = Path(args.output) if args.output else root
        print(f"\nEndpoints under {root}:")
        table = annotate_ensembles(root, sectors, wt, alphabet, output, override)
        print("\nSector enrichment (share of mutations / share of mutable sites):")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.sweep_dir:
        root = Path(args.sweep_dir)
        output = Path(args.output) if args.output else root
        print(f"\nTrajectories under {root}:")
        annotate_sweeps(root, sectors, output)


if __name__ == "__main__":
    main()
