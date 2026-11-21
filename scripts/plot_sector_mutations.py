#!/usr/bin/env python
"""Where the mutations landed: red sector, grafting segment, or elsewhere.

Reads the two summaries written by ``annotate_sectors.py`` and draws three
panels:

1. **Mutations per chain by sector, against lambda.** The absolute load each
   sector carries as the bias is turned up.
2. **Enrichment against lambda.** The share of mutations a sector received
   divided by its share of the *mutable* columns. The dashed rule at 1 is what
   a sampler indifferent to the sectors would give, so a curve above it means
   the bias is concentrating mutations there of its own accord. For a run
   restricted to the grafting window this is only interesting between red and
   the rest of the window; for an unrestricted run it is the whole question.
3. **Per-column mutation frequency along the alignment.** One bar per column,
   coloured by sector, for a chosen lambda -- the resolved version of panel 2,
   which shows whether a handful of columns carry the sector's whole signal.

Example::

    python scripts/plot_sector_mutations.py \\
        --sector-dir ../results_v2/sectors \\
        --column-lambda 1000 \\
        --save docs/sector_mutations.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sectors import SECTORS  # noqa: E402

SURFACE = "#fcfcfb"
GRID_KWARGS = dict(color="#e2e1dc", linewidth=0.5, alpha=0.9)
#: One colour per sector, kept the same in every panel.
SECTOR_COLOR = {"red": "#c0392b", "grafting": "#e08a2e", "other": "#8d8c86"}
SECTOR_LABEL = {
    "red": "red sector",
    "grafting": "grafting segment (outside red)",
    "other": "rest of the alignment",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sector-dir", required=True, help="Directory written by annotate_sectors.py")
    parser.add_argument(
        "--column-lambda",
        type=float,
        default=None,
        help="Which lambda the per-column panel shows (default: the largest)",
    )
    parser.add_argument(
        "--title",
        default="Where the mutations land",
        help="Figure title",
    )
    parser.add_argument("--save", default="sector_mutations.png", help="Output image")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--figsize", type=float, nargs=2, default=(13.0, 8.4))
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument(
        "--annotate-top",
        type=int,
        default=8,
        help="Label this many of the most-mutated red-sector columns (0 = none)",
    )
    return parser


def _style(axis) -> None:
    axis.grid(True, **GRID_KWARGS)
    axis.set_axisbelow(True)
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#a9a8a2")
    axis.tick_params(colors="#52514e")


def _lambda_axis(axis, lambdas) -> None:
    """Lambda spans decades and usually includes 0, so use a symlog axis."""
    axis.set_xscale("symlog", linthresh=10)
    axis.set_xticks(sorted(lambdas))
    axis.set_xticklabels([f"{v:g}" for v in sorted(lambdas)])
    axis.set_xlabel(r"$\lambda$")


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.sector_dir)
    enrichment_path = root / "sector_enrichment.csv"
    columns_path = root / "column_frequency.csv"
    if not enrichment_path.is_file():
        raise SystemExit(f"{enrichment_path} not found -- run annotate_sectors.py first")

    scale = args.font_scale
    plt.rcParams.update(
        {
            "font.size": 10 * scale,
            "axes.titlesize": 12 * scale,
            "axes.labelsize": 11.5 * scale,
            "xtick.labelsize": 9.5 * scale,
            "ytick.labelsize": 9.5 * scale,
            "legend.fontsize": 10.5 * scale,
        }
    )

    table = pd.read_csv(enrichment_path)
    lambdas = sorted(table["lam"].unique())

    figure = plt.figure(figsize=tuple(args.figsize))
    figure.patch.set_facecolor(SURFACE)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.38, wspace=0.30)
    ax_load = figure.add_subplot(grid[0, 0])
    ax_ratio = figure.add_subplot(grid[0, 1])
    ax_columns = figure.add_subplot(grid[1, :])

    # --- 1. mutations per chain ------------------------------------------------
    for sector in SECTORS:
        rows = table[table["sector"] == sector].sort_values("lam")
        if rows.empty:
            continue
        ax_load.plot(
            rows["lam"], rows["mean_per_sequence"], marker="o", markersize=5,
            color=SECTOR_COLOR[sector], label=SECTOR_LABEL[sector], linewidth=1.8,
        )
    _style(ax_load)
    _lambda_axis(ax_load, lambdas)
    ax_load.set_ylabel("Mutations per chain")
    ax_load.set_title("How many mutations each sector carries", loc="left")
    ax_load.legend(frameon=False)

    # --- 2. enrichment ---------------------------------------------------------
    for sector in SECTORS:
        rows = table[table["sector"] == sector].sort_values("lam")
        if rows.empty or rows["mutable_columns"].sum() == 0:
            continue
        ax_ratio.plot(
            rows["lam"], rows["ratio"], marker="o", markersize=5,
            color=SECTOR_COLOR[sector], label=SECTOR_LABEL[sector], linewidth=1.8,
        )
    ax_ratio.axhline(1.0, linestyle="--", linewidth=1.0, color="#52514e")
    ax_ratio.text(
        0.99, 1.0, " indifferent", transform=ax_ratio.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=9 * scale, color="#52514e",
    )
    _style(ax_ratio)
    _lambda_axis(ax_ratio, lambdas)
    ax_ratio.set_yscale("log")
    ax_ratio.set_ylabel("Enrichment ratio")
    ax_ratio.set_title("Enrichment over an indifferent sampler", loc="left")

    # --- 3. per-column frequency ----------------------------------------------
    if columns_path.is_file():
        frequency = pd.read_csv(columns_path)
        chosen = args.column_lambda if args.column_lambda is not None else max(lambdas)
        rows = frequency[np.isclose(frequency["lam"], chosen)]
        if rows.empty:
            raise SystemExit(
                f"No per-column rows for lambda {chosen:g}; available: "
                + ", ".join(f"{v:g}" for v in sorted(frequency['lam'].unique()))
            )
        rows = rows.sort_values("msa_col")
        colours = [SECTOR_COLOR.get(s, SECTOR_COLOR["other"]) for s in rows["sector"]]
        ax_columns.bar(rows["msa_col"], rows["frac_mutated"], color=colours, width=1.0)
        _style(ax_columns)
        ax_columns.set_xlabel("MSA column (0-based)")
        ax_columns.set_ylabel("Fraction of chains mutated")
        ax_columns.set_xlim(-1, int(rows["msa_col"].max()) + 1)
        ax_columns.set_title(
            f"Per-column mutation frequency at $\\lambda$ = {chosen:g}", loc="left"
        )
        handles = [
            plt.Line2D([], [], color=SECTOR_COLOR[s], linewidth=6, label=SECTOR_LABEL[s])
            for s in SECTORS
        ]
        ax_columns.legend(
            handles=handles, frameon=False, ncol=3,
            loc="upper center", bbox_to_anchor=(0.5, -0.20),
        )

        if args.annotate_top:
            red = rows[rows["sector"] == "red"].nlargest(args.annotate_top, "frac_mutated")
            for _, row in red.iterrows():
                ax_columns.annotate(
                    str(row["pdb_residue"]),
                    xy=(row["msa_col"], row["frac_mutated"]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5 * scale, color=SECTOR_COLOR["red"],
                )
    else:
        ax_columns.set_visible(False)

    figure.suptitle(args.title, x=0.012, ha="left", fontsize=14 * scale)
    figure.savefig(args.save, dpi=args.dpi, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {args.save}")


if __name__ == "__main__":
    main()
