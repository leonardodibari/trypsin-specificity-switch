#!/usr/bin/env python
"""Stamp the target residue into the critic site of sampled sequences, and rescore.

The sampler biases the *conditional* ``P(S | critic site)``, which does not depend
on the residue sitting at that site; a chain's final sequence therefore still
carries whatever residue it last accepted there. This script takes the ensembles
written by ``run_ensemble.py``, substitutes the target residue (default ``S``) at
the critic column of every final sequence, and recomputes the four observables, so
that a figure drawn from them describes sequences that actually carry the serine.

Only the sampled sequences are touched. The natural alignment and the calibration
of ``mu`` are left exactly as they were.

Example::

    python scripts/substitute_critic.py \\
        --ensemble-dir ../results/ensembles \\
        --output ../results_serine/ensembles \\
        --wt data/wt_trypsin.fasta --model ../bmDCA_params.dat \\
        --critic-col 197 --cache-model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import _SRC  # noqa: F401  (puts src/ on sys.path)

from encoding import Alphabet, read_sequences, read_single_sequence  # noqa: E402
from observables import describe  # noqa: E402
from potts import PottsModel  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-dir", required=True, help="Directory of lam*/final_sequences.fasta")
    parser.add_argument("--output", required=True, help="Directory for the substituted ensembles")
    parser.add_argument("--wt", required=True, help="FASTA with the aligned wild-type sequence")
    parser.add_argument("--model", required=True, help="adabmDCA parameter file or model directory")
    parser.add_argument("--critic-col", type=int, required=True, help="MSA column of the critic site (0-based)")
    parser.add_argument("--target-residue", default="S", help="Residue to stamp in (default: S)")
    parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature (default: 1.0)")
    parser.add_argument("--alphabet", default="protein", help="adabmDCA alphabet (default: protein)")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--cache-model", action="store_true", help="Reuse a <params>.cache.pt tensor dump")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    alphabet = Alphabet.from_name(args.alphabet)
    model = PottsModel.load(
        args.model, alphabet=alphabet, device=args.device, dtype=args.dtype, cache=args.cache_model
    )

    wt_name, wt_string = read_single_sequence(args.wt)
    wt_sequence = alphabet.encode(wt_string)
    if wt_sequence.shape[0] != model.L:
        raise SystemExit(f"The wild type has length {wt_sequence.shape[0]}, the model expects L={model.L}")

    critic = args.critic_col
    if not 0 <= critic < model.L:
        raise SystemExit(f"The critic column {critic} is outside the model (L={model.L})")
    target_state = alphabet.index(args.target_residue)
    if target_state == alphabet.gap_index:
        raise SystemExit("The target residue may not be the gap")

    print(f"Model     : {model.source}  (L={model.L}, q={model.q})")
    print(
        f"Critic    : MSA column {critic} -> {args.target_residue} "
        f"(wild type has {alphabet.tokens[int(wt_sequence[critic])]})"
    )

    source = Path(args.ensemble_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for directory in sorted(source.glob("lam*")):
        fasta = directory / "final_sequences.fasta"
        if not directory.is_dir() or not fasta.is_file():
            continue
        try:
            lam = float(directory.name[3:])
        except ValueError:
            continue

        records = read_sequences(fasta)
        encoded = np.stack([alphabet.encode(sequence) for _, sequence in records])
        if encoded.shape[1] != model.L:
            raise SystemExit(f"{fasta} holds sequences of length {encoded.shape[1]}, the model expects {model.L}")

        before = encoded[:, critic].copy()
        already = int((before == target_state).sum())
        substituted = encoded.copy()
        substituted[:, critic] = target_state

        observables = describe(
            model,
            substituted,
            reference=wt_sequence,
            critic_col=critic,
            target_state=target_state,
            beta=args.beta,
        )
        frame = pd.DataFrame({"chain": np.arange(substituted.shape[0]), **observables})
        frame.insert(1, "lam", lam)
        frame["critic_residue_before"] = [alphabet.tokens[int(state)] for state in before]

        target = output / directory.name
        target.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target / "observables.csv", index=False)
        with open(target / "final_sequences.fasta", "w", encoding="utf-8") as handle:
            for (name, _), sequence in zip(records, substituted):
                handle.write(f">{name}\n{alphabet.decode(sequence)}\n")

        counts = pd.Series([alphabet.tokens[int(state)] for state in before]).value_counts()
        summary = {
            "lam": lam,
            "beta": args.beta,
            "n_chains": int(substituted.shape[0]),
            "critic_col": critic,
            "target_residue": args.target_residue,
            "source": str(fasta),
            "chains_already_target": already,
            "critic_residue_counts_before": {str(k): int(v) for k, v in counts.items()},
            "mean_E_DCA": float(frame["E_DCA"].mean()),
            "mean_P_S": float(frame["P_S"].mean()),
            "mean_hamming_to_wt": float(frame["hamming_to_wt"].mean()),
            "mean_CDE": float(frame["CDE"].mean()),
        }
        with open(target / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        rows.append(summary)

        top = ", ".join(f"{residue} {count}" for residue, count in counts.head(4).items())
        print(
            f"lambda = {lam:g}: {substituted.shape[0]} chains, {already} already {args.target_residue} "
            f"(before: {top}) | <E> {summary['mean_E_DCA']:.1f} | "
            f"<P(S)> {summary['mean_P_S']:.3f} | <CDE> {summary['mean_CDE']:.3f}"
        )

    if not rows:
        raise SystemExit(f"No lam*/final_sequences.fasta found under {source}")
    pd.DataFrame(rows).drop(columns=["critic_residue_counts_before"]).to_csv(
        output / "ensemble_summary.csv", index=False
    )
    print(f"\nwritten to {output}")


if __name__ == "__main__":
    main()
