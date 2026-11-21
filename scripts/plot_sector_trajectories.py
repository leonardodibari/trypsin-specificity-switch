#!/usr/bin/env python
"""Where the mutations happen *along* the chains, not just at the end.

Reads the ``trajectory_sectors.csv`` files written by ``annotate_sectors.py``
-- one per chain, every recorded move carrying the sector and residue label of
the column it changed -- and draws one column of panels per lambda:

1. **A raster of the moves.** Every recorded move is a dot at (accepted step,
   MSA column), coloured by sector, pooled over the replicate chains. This is
   the direct answer to *where and when*: a horizontal band means one column is
   being hit over and over, a dot early in a red-sector row means the chain
   went there first.
2. **Cumulative mutations by sector.** The running count of moves in each
   sector, median over chains with the inter-quartile band. The slope is the
   rate at which a sector is being worked on, so a red curve that steepens
   where the orange one flattens says the chain switched its attention.
3. **The running share of red-sector moves.** The same thing normalised, with
   the share of *mutable* columns that are red drawn as a dashed rule: above it
   the chain is favouring the red sector over an indifferent sampler.

Example::

    python scripts/plot_sector_trajectories.py \\
        --sweep-dir ../results_v2/grafting/sweep \\
        --sectors ../work/data/sectors_iter_aln_v2.csv \\
        --save docs/sector_trajectories.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sectors import SECTORS, Sectors  # noqa: E402

SURFACE = "#fcfcfb"
GRID_KWARGS = dict(color="#e2e1dc", linewidth=0.5, alpha=0.9)
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
    parser.add_argument("--sweep-dir", required=True, help="Directory of lam*/run*/trajectory_sectors.csv")
    parser.add_argument("--sectors", required=True, help="The sector CSV, for the red-sector baseline")
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=None, help="Which lambdas to show (default: all)"
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Truncate the x axis (0 = all)")
    parser.add_argument("--log-x", action="store_true", help="Logarithmic step axis")
    parser.add_argument("--title", default="Where the mutations happen along the chains")
    parser.add_argument("--save", default="sector_trajectories.png", help="Output image")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--figsize", type=float, nargs=2, default=None)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--point-size", type=float, default=7.0, help="Raster marker area")
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


def _lam_value(path: Path) -> float:
    for part in path.parts[::-1]:
        if part.startswith("lam"):
            try:
                return float(part[3:])
            except ValueError:
                continue
    return float("nan")


def _mutable(path: Path) -> List[int] | None:
    """The allowed columns of the run this trajectory belongs to."""
    for candidate in (path.parent / "summary.json",):
        if candidate.is_file():
            try:
                with open(candidate, encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                return None
            columns = payload.get("allowed_cols")
            return [int(c) for c in columns] if columns else None
    return None


def load(sweep_dir: Path) -> Dict[float, List[pd.DataFrame]]:
    """Every annotated trajectory, grouped by lambda."""
    grouped: Dict[float, List[pd.DataFrame]] = {}
    paths = sorted(sweep_dir.glob("lam*/**/trajectory_sectors.csv"))
    if not paths:
        raise SystemExit(
            f"No lam*/**/trajectory_sectors.csv under {sweep_dir} -- run annotate_sectors.py "
            "with --sweep-dir first"
        )
    for path in paths:
        frame = pd.read_csv(path)
        frame["msa_col_changed"] = pd.to_numeric(frame["msa_col_changed"], errors="coerce")
        frame["sector"] = frame["sector"].fillna("").astype(str)
        frame.attrs["mutable"] = _mutable(path)
        grouped.setdefault(_lam_value(path), []).append(frame)
    return grouped


def _accepted_axis(frame: pd.DataFrame) -> np.ndarray:
    """The recorded moves numbered 1, 2, 3 ... -- ``step`` counts proposals."""
    return np.arange(1, len(frame) + 1, dtype=float)


def main() -> None:
    args = build_parser().parse_args()
    sweep = Path(args.sweep_dir)
    grouped = load(sweep)
    lambdas = sorted(grouped) if args.lambdas is None else [
        lam for lam in sorted(grouped) if any(np.isclose(lam, v) for v in args.lambdas)
    ]
    if not lambdas:
        raise SystemExit("No lambda left to plot")

    sectors = Sectors.load(args.sectors)
    scale = args.font_scale
    plt.rcParams.update(
        {
            "font.size": 10 * scale,
            "axes.titlesize": 11.5 * scale,
            "axes.labelsize": 11 * scale,
            "xtick.labelsize": 9 * scale,
            "ytick.labelsize": 9 * scale,
            "legend.fontsize": 10 * scale,
        }
    )

    figsize = tuple(args.figsize) if args.figsize else (4.6 * len(lambdas) + 1.0, 10.4)
    figure, axes = plt.subplots(3, len(lambdas), figsize=figsize, squeeze=False)
    figure.patch.set_facecolor(SURFACE)

    for column, lam in enumerate(lambdas):
        frames = grouped[lam]
        ax_raster, ax_cumulative, ax_share = (axes[row][column] for row in range(3))

        mutable = next((f.attrs.get("mutable") for f in frames if f.attrs.get("mutable")), None)
        allowed = set(range(sectors.L)) if not mutable else {int(c) for c in mutable}
        red_share = len([c for c in sectors.columns("red") if c in allowed]) / max(len(allowed), 1)

        # --- 1. raster ---------------------------------------------------------
        for frame in frames:
            moves = frame[frame["sector"] != ""]
            if moves.empty:
                continue
            index = _accepted_axis(frame)[frame["sector"].to_numpy() != ""]
            if args.max_steps:
                keep = index <= args.max_steps
                moves, index = moves[keep], index[keep]
            ax_raster.scatter(
                index, moves["msa_col_changed"], s=args.point_size,
                c=[SECTOR_COLOR.get(s, SECTOR_COLOR["other"]) for s in moves["sector"]],
                alpha=0.55, linewidths=0,
            )
        _style(ax_raster)
        if args.log_x:
            ax_raster.set_xscale("log")
        ax_raster.set_title(rf"$\lambda = {lam:g}$", loc="left")
        if column == 0:
            ax_raster.set_ylabel("MSA column changed")
            ax_raster.legend(
                handles=[
                    plt.Line2D([], [], marker="o", linestyle="none", color=SECTOR_COLOR[s],
                               markersize=6, label=SECTOR_LABEL[s])
                    for s in SECTORS
                ],
                frameon=False, loc="upper left", fontsize=8.5 * scale,
            )

        # --- 2. cumulative counts ---------------------------------------------
        length = max(len(frame) for frame in frames)
        axis_steps = np.arange(1, length + 1, dtype=float)
        for name in SECTORS:
            curves = []
            for frame in frames:
                hit = (frame["sector"].to_numpy() == name).astype(float)
                curve = np.cumsum(hit)
                if len(curve) < length:  # a chain that stopped early holds its last value
                    curve = np.concatenate([curve, np.full(length - len(curve), curve[-1])])
                curves.append(curve)
            stack = np.vstack(curves)
            if stack.sum() == 0:
                continue
            median = np.median(stack, axis=0)
            ax_cumulative.plot(axis_steps, median, color=SECTOR_COLOR[name], linewidth=1.8,
                               label=SECTOR_LABEL[name])
            ax_cumulative.fill_between(
                axis_steps, np.quantile(stack, 0.25, axis=0), np.quantile(stack, 0.75, axis=0),
                color=SECTOR_COLOR[name], alpha=0.16, linewidth=0,
            )
        _style(ax_cumulative)
        if args.log_x:
            ax_cumulative.set_xscale("log")
        if args.max_steps:
            ax_cumulative.set_xlim(right=args.max_steps)
        if column == 0:
            ax_cumulative.set_ylabel("Cumulative mutations")
            ax_cumulative.legend(frameon=False, loc="upper left", fontsize=8.5 * scale)

        # --- 3. running share of red moves -------------------------------------
        shares = []
        for frame in frames:
            hit = (frame["sector"].to_numpy() == "red").astype(float)
            any_move = (frame["sector"].to_numpy() != "").astype(float)
            total = np.cumsum(any_move)
            with np.errstate(invalid="ignore", divide="ignore"):
                share = np.where(total > 0, np.cumsum(hit) / np.maximum(total, 1), np.nan)
            if len(share) < length:
                share = np.concatenate([share, np.full(length - len(share), share[-1])])
            shares.append(share)
        stack = np.vstack(shares)
        ax_share.plot(axis_steps, np.nanmedian(stack, axis=0), color=SECTOR_COLOR["red"],
                      linewidth=1.8)
        ax_share.fill_between(
            axis_steps, np.nanquantile(stack, 0.25, axis=0), np.nanquantile(stack, 0.75, axis=0),
            color=SECTOR_COLOR["red"], alpha=0.16, linewidth=0,
        )
        ax_share.axhline(red_share, linestyle="--", linewidth=1.1, color="#52514e")
        ax_share.text(
            0.99, red_share, " indifferent ", transform=ax_share.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8.5 * scale, color="#52514e",
        )
        _style(ax_share)
        if args.log_x:
            ax_share.set_xscale("log")
        if args.max_steps:
            ax_share.set_xlim(right=args.max_steps)
        ax_share.set_ylim(0, 1)
        ax_share.set_xlabel("Recorded (accepted) move")
        if column == 0:
            ax_share.set_ylabel("Share of moves in the red sector")

    figure.suptitle(args.title, x=0.012, ha="left", fontsize=14 * scale)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(args.save, dpi=args.dpi, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {args.save}")


if __name__ == "__main__":
    main()
