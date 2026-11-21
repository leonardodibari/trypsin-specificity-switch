#!/usr/bin/env python
"""Pull one labelled row out of the alignment and write it as a FASTA file.

Every single-sequence reference under ``data/`` is produced by this script, and
that matters more than it looks.  A reference sequence is only meaningful in the
coordinate system of the alignment the model was fitted on: ``iter_aln_v2`` has
L = 265, and a row taken from an earlier alignment (L = 260) will load, will
look like a protein, and will then be silently wrong at every column after the
first indel.

That is not hypothetical.  The repository shipped a 260-column
``wt_trypsin.fasta`` left over from the previous alignment; it was caught only
because ``run_ensemble.py`` refuses a wild type whose length does not match the
model's L.  Regenerating the references with this script, rather than copying
them between directories, is what keeps that from happening again.

Usage
-----
    python scripts/extract_aligned_row.py --alignment $TRYP_DATA_ROOT/iter_aln_v2.faa \\
        --id P00763 --output data/wt_trypsin.fasta \\
        --description "TRY2_RAT anionic trypsin, aligned row of iter_aln_v2 (L=265)"

The accession is matched against the first whitespace-delimited token of the
header, which is how the rows of ``iter_aln_v2.faa`` are named.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def read_row(alignment: Path, accession: str) -> str:
    """Return the gapped sequence whose header starts with ``accession``."""
    chunks: list[str] = []
    grabbing = False
    for line in alignment.open():
        if line.startswith(">"):
            if grabbing:
                break
            grabbing = line[1:].strip().split()[0] == accession
        elif grabbing:
            chunks.append(line.strip())
    if not chunks:
        raise SystemExit(f"accession {accession!r} not found in {alignment}")
    return "".join(chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alignment", type=Path, required=True)
    ap.add_argument("--id", required=True, help="accession, e.g. P00763")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--description", default="", help="text after the accession")
    ap.add_argument("--expect-length", type=int, default=None,
                    help="refuse to write unless the row has this many columns")
    ap.add_argument("--wrap", type=int, default=60)
    args = ap.parse_args()

    seq = read_row(args.alignment, args.id)
    if args.expect_length is not None and len(seq) != args.expect_length:
        raise SystemExit(
            f"{args.id}: row is {len(seq)} columns, expected {args.expect_length}. "
            "Wrong alignment?")

    header = args.description or f"{args.id}, aligned row of {args.alignment.stem}"
    body = "\n".join(seq[i:i + args.wrap] for i in range(0, len(seq), args.wrap))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f">{header}\n{body}\n")
    print(f"{args.output}: {len(seq)} columns, {seq.count('-')} gaps")


if __name__ == "__main__":
    main()
