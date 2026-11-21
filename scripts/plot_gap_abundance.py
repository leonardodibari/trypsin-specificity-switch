#!/usr/bin/env python
"""How many gaps the designs carry, and where those gaps are.

Only meaningful for a run sampled with ``--allow-gap-moves``: without it every
sequence keeps the wild type's gap pattern and all three panels are flat lines.

1. **Gap count per chain, by lambda.** The distribution of ``n_gaps`` across the
   chains of each ensemble, as a violin, with the wild type's own gap count as a
   rule and the natural alignment's mean as a second one. This is the figure
   that says whether the bias drives insertions or deletions.
2. **Gaps opened and closed.** The same drift split into its two halves --
   columns that were a residue in the wild type and became a gap, and the
   reverse. A net count of zero can still hide a wholesale rearrangement.
3. **Per-column gap frequency.** Along the alignment, one bar per column,
   coloured by sector, for a chosen lambda. Columns that are a gap in the wild
   type are outlined, so a bar that is tall *without* an outline is a newly
   opened gap and a short bar *with* one is a gap the chains filled in.

Example::

    python scripts/plot_gap_abundance.py \\
        --ensemble-dir ../results_v2/grafting/ensembles_serine \\
        --sector-dir ../results_v2/grafting/sectors \\
        --natural-observables ../results_v2/calibration/natural_observables.csv \\
        --wt ../work/data/wt_trypsin.fasta \\
        --save docs/gap_abundance.png
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

from encoding import Alphabet, read_single_sequence  # noqa: E402
from sectors import SECTORS  # noqa: E402

SURFACE = "#fcfcfb"
GRID_KWARGS = dict(color="#e2e1dc", linewidth=0.5, alpha=0.9)
SECTOR_COLOR = {"red": "#c0392b", "grafting": "#e08a2e", "other": "#8d8c86"}
SECTOR_LABEL = {
    "red": "red sector",
    "grafting": "grafting segment (outside red)",
    "other": "rest of the alignment",
}
LAMBDA_CMAP = plt.get_cmap("Blues")
LAMBDA_RANGE = (0.38, 0.95)
OPEN_COLOR = "#2f6f4e"
CLOSE_COLOR = "#8c5bb0"
WT_COLOR = "#1f7a4d"
NATURAL_COLOR = "#b0afa8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ensemble-dir", required=True, help="Directory of lam*/observables.csv")
    parser.add_argument(
        "--sector-dir",
        default=None,
        help="Directory with column_frequency.csv from annotate_sectors.py (panel 3)",
    )
    parser.add_argument("--wt", default=None, help="Wild-type FASTA, for its own gap count")
    parser.add_argument(
        "--natural-observables",
        default=None,
        help="natural_observables.csv from calibrate_mu.py, for the family's mean gap count",
    )
    parser.add_argument(
        "--column-lambda",
        type=float,
        default=None,
        help="Which lambda panel 3 shows (default: the largest)",
    )
    parser.add_argument(
        "--alphabet", default="protein", help="adabmDCA alphabet (default: protein)"
    )
    parser.add_argument("--title", default="Gaps in the designed sequences", help="Figure title")
    parser.add_argument("--save", default="gap_abundance.png", help="Output image")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--figsize", type=float, nargs=2, default=(13.0, 8.4))
    parser.add_argument("--font-scale", type=float, default=1.0)
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


def _lam_value(directory: Path) -> float:
    try:
        return float(directory.name[3:])
    except ValueError:
        return float("inf")


def _load(directory: Path):
    frames = []
    for sub in sorted((d for d in directory.glob("lam*") if d.is_dir()), key=_lam_value):
        path = sub / "observables.csv"
        if path.is_file():
            frames.append((_lam_value(sub), pd.read_csv(path)))
    if not frames:
        raise SystemExit(f"No lam*/observables.csv under {directory}")
    return frames


def main() -> None:
    args = build_parser().parse_args()
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

    ensembles = _load(Path(args.ensemble_dir))
    lambdas = [lam for lam, _ in ensembles]
    if "n_gaps" not in ensembles[0][1].columns:
        raise SystemExit(
            "observables.csv has no n_gaps column -- rerun run_ensemble.py with this version"
        )

    wt_gaps = None
    if args.wt:
        alphabet = Alphabet.from_name(args.alphabet)
        _, wt_string = read_single_sequence(args.wt)
        wt_gaps = int((alphabet.encode(wt_string) == alphabet.gap_index).sum())

    natural_gaps = None
    if args.natural_observables and Path(args.natural_observables).is_file():
        natural = pd.read_csv(args.natural_observables)
        if "n_gaps" in natural.columns:
            natural_gaps = float(natural["n_gaps"].mean())

    figure = plt.figure(figsize=tuple(args.figsize))
    figure.patch.set_facecolor(SURFACE)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.40, wspace=0.26)
    ax_count = figure.add_subplot(grid[0, 0])
    ax_drift = figure.add_subplot(grid[0, 1])
    ax_columns = figure.add_subplot(grid[1, :])

    positions = np.arange(len(ensembles))
    shades = np.linspace(*LAMBDA_RANGE, max(len(ensembles), 2))

    # --- 1. gap count per chain -------------------------------------------------
    data = [frame["n_gaps"].to_numpy(dtype=float) for _, frame in ensembles]
    # A violin needs variation; a lambda whose chains all agree gets a bar instead.
    varying = [i for i, values in enumerate(data) if values.std() > 0]
    if varying:
        parts = ax_count.violinplot(
            [data[i] for i in varying], positions=positions[varying], widths=0.75,
            showextrema=False, showmedians=True,
        )
        for index, body in zip(varying, parts["bodies"]):
            body.set_facecolor(LAMBDA_CMAP(shades[index]))
            body.set_edgecolor("none")
            body.set_alpha(0.85)
        parts["cmedians"].set_color("#2f2e2b")
        parts["cmedians"].set_linewidth(1.2)
    for index, values in enumerate(data):
        if values.std() == 0:
            ax_count.plot(
                [index], [values[0]], marker="_", markersize=28,
                color=LAMBDA_CMAP(shades[index]), markeredgewidth=3,
            )
    if wt_gaps is not None:
        ax_count.axhline(wt_gaps, color=WT_COLOR, linestyle="--", linewidth=1.2)
        ax_count.text(
            0.995, wt_gaps, " wild type ", transform=ax_count.get_yaxis_transform(),
            ha="right", va="bottom", color=WT_COLOR, fontsize=9 * scale,
        )
    if natural_gaps is not None:
        ax_count.axhline(natural_gaps, color=NATURAL_COLOR, linestyle=":", linewidth=1.4)
        ax_count.text(
            0.995, natural_gaps, " natural mean ", transform=ax_count.get_yaxis_transform(),
            ha="right", va="bottom", color="#6f6e69", fontsize=9 * scale,
        )
    _style(ax_count)
    ax_count.set_xticks(positions)
    ax_count.set_xticklabels([f"{v:g}" for v in lambdas])
    ax_count.set_xlabel(r"$\lambda$")
    ax_count.set_ylabel("Gaps per sequence")
    ax_count.set_title("How many gaps each ensemble carries", loc="left")

    # --- 2. opened against closed ----------------------------------------------
    if "gaps_opened" in ensembles[0][1].columns:
        for name, colour, label in (
            ("gaps_opened", OPEN_COLOR, "opened (residue → gap)"),
            ("gaps_closed", CLOSE_COLOR, "closed (gap → residue)"),
        ):
            means = [frame[name].mean() for _, frame in ensembles]
            q25 = [frame[name].quantile(0.25) for _, frame in ensembles]
            q75 = [frame[name].quantile(0.75) for _, frame in ensembles]
            ax_drift.plot(positions, means, marker="o", markersize=5, color=colour,
                          linewidth=1.8, label=label)
            ax_drift.fill_between(positions, q25, q75, color=colour, alpha=0.16, linewidth=0)
        net = [frame["gaps_opened"].mean() - frame["gaps_closed"].mean() for _, frame in ensembles]
        ax_drift.plot(positions, net, marker="s", markersize=4, color="#2f2e2b",
                      linewidth=1.3, linestyle="--", label="net change")
        ax_drift.axhline(0.0, color="#a9a8a2", linewidth=1.0)
        ax_drift.legend(frameon=False)
    else:
        ax_drift.text(0.5, 0.5, "no gaps_opened / gaps_closed columns\n(run annotate_sectors.py)",
                      ha="center", va="center", transform=ax_drift.transAxes, color="#6f6e69")
    _style(ax_drift)
    ax_drift.set_xticks(positions)
    ax_drift.set_xticklabels([f"{v:g}" for v in lambdas])
    ax_drift.set_xlabel(r"$\lambda$")
    ax_drift.set_ylabel("Columns per sequence")
    ax_drift.set_title("Gaps opened and closed, against the wild type", loc="left")

    # --- 3. per-column gap frequency -------------------------------------------
    columns_path = Path(args.sector_dir) / "column_frequency.csv" if args.sector_dir else None
    if columns_path and columns_path.is_file():
        frequency = pd.read_csv(columns_path)
        if "frac_gap" not in frequency.columns:
            raise SystemExit(f"{columns_path} has no frac_gap column -- rerun annotate_sectors.py")
        chosen = args.column_lambda if args.column_lambda is not None else max(lambdas)
        rows = frequency[np.isclose(frequency["lam"], chosen)].sort_values("msa_col")
        if rows.empty:
            raise SystemExit(f"No per-column rows for lambda {chosen:g}")
        colours = [SECTOR_COLOR.get(s, SECTOR_COLOR["other"]) for s in rows["sector"]]
        edges = ["#2f2e2b" if bool(g) else "none" for g in rows.get("wt_gap", [False] * len(rows))]
        ax_columns.bar(
            rows["msa_col"], rows["frac_gap"], color=colours, width=1.0,
            edgecolor=edges, linewidth=0.7,
        )
        _style(ax_columns)
        ax_columns.set_xlabel("MSA column (0-based)")
        ax_columns.set_ylabel("Fraction of chains with a gap")
        ax_columns.set_xlim(-1, int(rows["msa_col"].max()) + 1)
        ax_columns.set_ylim(0, 1.05)
        ax_columns.set_title(
            f"Per-column gap frequency at $\\lambda$ = {chosen:g}"
            "  —  outlined bars are gaps in the wild type",
            loc="left",
        )
        handles = [
            plt.Line2D([], [], color=SECTOR_COLOR[s], linewidth=6, label=SECTOR_LABEL[s])
            for s in SECTORS
        ]
        ax_columns.legend(
            handles=handles, frameon=False, ncol=3,
            loc="upper center", bbox_to_anchor=(0.5, -0.20),
        )
    else:
        ax_columns.set_visible(False)

    figure.suptitle(args.title, x=0.012, ha="left", fontsize=14 * scale)
    figure.savefig(args.save, dpi=args.dpi, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {args.save}")


if __name__ == "__main__":
    main()
