"""Shared CLI plumbing for the scripts: imports, model/WT/site loading."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

# Allow running the scripts straight from a checkout (``python scripts/run_mcmc.py``)
# as well as from an installed package (``pip install -e .``).
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from encoding import Alphabet, read_single_sequence  # noqa: E402
from potts import PottsModel  # noqa: E402
from sites import (  # noqa: E402
    CRITIC_3TGI_POSITION,
    load_mapping,
    mutable_columns,
    parse_column_spec,
    resolve_sites,
)


@dataclass
class RunSetup:
    """Everything a chain needs that is shared across lambdas and replicates."""

    model: PottsModel
    alphabet: Alphabet
    wt_name: str
    wt_sequence: np.ndarray
    critic_col: int
    allowed_cols: List[int]
    allowed_labels: List[str]


def add_common_arguments(parser) -> None:
    """CLI options shared by ``run_mcmc.py`` and ``run_sweep.py``."""
    parser.add_argument("--wt", required=True, help="FASTA with the aligned wild-type sequence")
    parser.add_argument("--model", required=True, help="adabmDCA parameter file or model directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--mapping",
        default=None,
        help="CSV mapping 3TGI residues to MSA columns (columns pdb_residue, msa_col)",
    )
    parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature (default: 1.0)")
    parser.add_argument(
        "--threshold", type=float, default=0.80, help="Stop when P(S|critic) reaches this (default: 0.80)"
    )
    parser.add_argument(
        "--allowed-sites",
        default="160-228",
        help="3TGI positions that may mutate: range, comma list, or 'sho' (default: 160-228)",
    )
    parser.add_argument(
        "--allowed-cols",
        default=None,
        help="MSA columns that may mutate, 0-based (e.g. '40-60,72'); replaces --allowed-sites",
    )
    parser.add_argument("--max-steps", type=int, default=1_000_000, help="Maximum proposed steps")
    parser.add_argument("--record-every", type=int, default=100, help="Record every N accepted steps")
    parser.add_argument(
        "--critic-site",
        default=CRITIC_3TGI_POSITION,
        help=f"3TGI position of the specificity site (default: {CRITIC_3TGI_POSITION})",
    )
    parser.add_argument(
        "--critic-col", type=int, default=None, help="MSA column of the critic site; replaces --critic-site"
    )
    parser.add_argument(
        "--target-residue", default="S", help="Residue favoured at the critic site (default: S)"
    )
    parser.add_argument(
        "--alphabet",
        default="protein",
        help="adabmDCA alphabet: 'protein' (default, -ACDEFGHIKLMNPQRSTVWY), 'dna', 'rna' or explicit tokens",
    )
    parser.add_argument(
        "--mapping-base",
        type=int,
        default=0,
        choices=(0, 1),
        help="Index base of msa_col in the mapping CSV (0 = Python, 1 = Julia; default: 0)",
    )
    parser.add_argument("--device", default="cpu", help="torch device for the model (default: cpu)")
    parser.add_argument(
        "--dtype", default="float32", choices=("float32", "float64"), help="Model dtype"
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Intra-op torch threads per chain; 1 is fastest here and leaves the cores to joblib",
    )
    parser.add_argument(
        "--cache-model",
        action="store_true",
        help="Keep a <params>.cache.pt tensor dump beside the parameter file and reuse it",
    )


def prepare_run(args, verbose: bool = True) -> RunSetup:
    """Load the model, the wild type and the mutable MSA columns."""
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    alphabet = Alphabet.from_name(args.alphabet)
    model = PottsModel.load(
        args.model,
        alphabet=alphabet,
        device=args.device,
        dtype=args.dtype,
        cache=args.cache_model,
        verbose=verbose,
    )

    wt_name, wt_string = read_single_sequence(args.wt)
    wt_sequence = alphabet.encode(wt_string)
    if wt_sequence.shape[0] != model.L:
        raise SystemExit(
            f"The wild-type sequence has length {wt_sequence.shape[0]} but the model has L={model.L}. "
            "The WT FASTA must be the aligned (gapped) MSA row."
        )

    critic_col, columns, labels = resolve_columns(args, model.L)
    label_of = dict(zip(columns, labels))

    if not 0 <= critic_col < model.L:
        raise SystemExit(f"The critic column {critic_col} is outside the model (L={model.L})")
    keep = mutable_columns(columns, [wt_sequence], alphabet)
    dropped = [label_of[c] for c in columns if c not in keep]
    if dropped and verbose:
        print(f"Skipping gap positions of the wild type: {dropped}")
    if not keep:
        raise SystemExit("Every allowed site is a gap in the wild-type sequence")

    if verbose:
        print(f"Model     : {model.source}  (L={model.L}, q={model.q}, tokens {alphabet.tokens!r})")
        print(f"Wild type : {wt_name}")
        print(
            f"Critic    : {_critic_label(args, critic_col)} "
            f"(WT residue {alphabet.tokens[int(wt_sequence[critic_col])]})"
        )
        kind = "3TGI sites" if args.allowed_cols is None else "MSA columns"
        print(f"Mutable   : {len(keep)} {kind} -> {[label_of[c] for c in keep]}")
        if critic_col in keep:
            print("Note      : the critic site is itself mutable.")

    return RunSetup(
        model=model,
        alphabet=alphabet,
        wt_name=wt_name,
        wt_sequence=wt_sequence,
        critic_col=critic_col,
        allowed_cols=keep,
        allowed_labels=[label_of[c] for c in keep],
    )


def resolve_columns(args, length: int | None = None):
    """``(critic_col, columns, labels)`` from either 3TGI labels or MSA columns.

    Sites are given in 3TGI numbering and mapped through ``--mapping``; passing
    ``--allowed-cols``/``--critic-col`` instead skips the mapping entirely, which
    is what to do when no mapping CSV is at hand.
    """
    direct_sites = args.allowed_cols is not None
    direct_critic = args.critic_col is not None

    if not (direct_sites and direct_critic) and not args.mapping:
        raise SystemExit(
            "--mapping is required to translate 3TGI positions into MSA columns; "
            "alternatively give the columns directly with --critic-col and --allowed-cols"
        )

    mapping = load_mapping(args.mapping, base=args.mapping_base) if args.mapping else None
    critic_col = args.critic_col if direct_critic else mapping.column(args.critic_site)

    if direct_sites:
        columns = sorted(set(parse_column_spec(args.allowed_cols, length=length)))
        labels = [str(column) for column in columns]
        if mapping is not None:
            known = mapping.column_to_label()
            labels = [known.get(column, str(column)) for column in columns]
    else:
        labels, columns, missing = resolve_sites(args.allowed_sites, mapping)
        if missing:
            print(f"WARNING: 3TGI positions absent from the mapping file: {missing}", file=sys.stderr)
        if not columns:
            raise SystemExit(f"No MSA column matches --allowed-sites {args.allowed_sites!r}")
    return int(critic_col), [int(c) for c in columns], list(labels)


def _critic_label(args, critic_col: int) -> str:
    if args.critic_col is not None:
        return f"MSA column {critic_col}"
    return f"3TGI {args.critic_site} -> MSA column {critic_col}"
