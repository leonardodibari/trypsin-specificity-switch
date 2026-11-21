"""Derive the consensus site set from THIS model's cold run.

The set was originally defined as: the columns the cold run rewrites in at least
95 % of its chains. That definition is a method, so it carries over; the columns
it picks do not, because they are a property of the model that produced them.
Running this against the serine-protease model gives that model's own consensus.

The SIZE is whatever the threshold yields. It is deliberately not forced to 19 --
calling the result "consensus-19" when the threshold gives a different number
would be inheriting the old model's answer through the back door.

Usage:
    python analysis/derive_consensus.py --results results/ladder \\
        --wt work/data/wt_trypsin.fasta --threshold 0.95 \\
        --output results/ladder/consensus_columns.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from encoding import Alphabet, read_sequences, read_single_sequence  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="Results root of this model")
    parser.add_argument("--wt", required=True, help="Aligned wild-type FASTA")
    parser.add_argument("--cold-run", default="cooling/b5/ensembles/lam500",
                        help="Run to read the consensus from, relative to --results "
                             "(default: the cold beta=5, lam_tilde=100 rung, as in "
                             "the original definition)")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Column is in the set if this fraction of chains rewrote it")
    parser.add_argument("--critic-col", type=int, default=201)
    parser.add_argument("--alphabet", default="protein")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    alphabet = Alphabet.from_name(args.alphabet)
    results = Path(args.results)

    run_dir = results / args.cold_run
    fasta = run_dir / "final_sequences.fasta"
    if not fasta.is_file():
        raise SystemExit(f"Cold run not found: {fasta}\n"
                         "Run the grafting stage first (bash analysis/run_ladder.sh graft).")

    wt = alphabet.encode(read_single_sequence(args.wt)[1])
    sequences = np.asarray([alphabet.encode(s) for _, s in read_sequences(str(fasta))])
    n_chains = len(sequences)
    rewritten = (sequences != wt[None, :]).mean(axis=0)

    summary = json.loads((run_dir / "summary.json").read_text())
    allowed = list(summary.get("allowed_cols", range(len(wt))))

    chosen = sorted(c for c in allowed if rewritten[c] >= args.threshold)
    if args.critic_col not in chosen:
        chosen = sorted(chosen + [args.critic_col])

    Path(args.output).write_text(",".join(str(c) for c in chosen) + "\n")

    print(f"cold run       : {fasta}")
    print(f"chains         : {n_chains:,}")
    print(f"allowed columns: {len(allowed)}")
    print(f"threshold      : {args.threshold:.0%} of chains rewrote the column")
    print(f"consensus size : {len(chosen)}  -> call it consensus-{len(chosen)}")
    print(f"columns        : {','.join(str(c) for c in chosen)}")
    print()
    # How was each chosen column rewritten? --allow-gap-moves means a column can
    # qualify on INDEL moves alone. Column 231 of the wide set is exactly that:
    # the wild type has a gap there and the sampler fills it, so the column is
    # not a point-mutation target at all. Anyone reading the set as a list of
    # positions to mutate has to know which entries are like that.
    gap_state = alphabet.gap_index
    print("composition of each chosen column:")
    print(f"  {'col':>5}{'rewritten':>11}{'subst':>9}{'deletion':>10}{'insertion':>11}")
    flagged = []
    for column in chosen:
        changed = sequences[:, column] != wt[column]
        to_gap = float((changed & (sequences[:, column] == gap_state)).mean())
        from_gap = float((changed & (wt[column] == gap_state)
                          & (sequences[:, column] != gap_state)).mean())
        sub = float(changed.mean()) - to_gap - from_gap
        note = ""
        if to_gap + from_gap > 0.5 * float(changed.mean()):
            note = "  <-- INDEL-DRIVEN, not a substitution site"
            flagged.append(column)
        print(f"  {column:>5}{100 * changed.mean():>10.1f}%{100 * sub:>8.1f}%"
              f"{100 * to_gap:>9.1f}%{100 * from_gap:>10.1f}%{note}")
    if flagged:
        print(f"\n  WARNING: {len(flagged)} column(s) qualify mainly on indel moves: "
              f"{', '.join(str(c) for c in flagged)}")
        print("  They are NOT point-mutation targets; say so wherever the set is "
              "presented as a mutation list.")
    print()
    print("per-column rewrite frequency, the columns around the cut:")
    ranked = sorted(allowed, key=lambda c: -rewritten[c])
    for rank, column in enumerate(ranked[: len(chosen) + 8]):
        mark = "  <= in" if rewritten[column] >= args.threshold else ""
        star = " (critic)" if column == args.critic_col else ""
        print(f"  {rank + 1:>3}. col {column:>4}  {rewritten[column]:6.1%}{mark}{star}")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
