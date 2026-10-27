#!/usr/bin/env python
"""Pairwise scatter matrix of the ensembles, one colour per lambda.

The four quantities, one per row/column:

* ``E_DCA``          -- the Potts statistical energy;
* ``hamming_to_wt``  -- substitutions away from the wild type;
* ``P_S``            -- P(S | critic site);
* ``Q_mu``           -- the sample quality ``E_DCA - mu * CDE``, with ``mu``
  calibrated on the natural alignment by ``scripts/calibrate_mu.py``.

Lambda is an ordered magnitude, so the ensembles are coloured along one hue,
light to dark, with the natural alignment (when given) as a grey cloud behind
them and the wild type as a green circle.

Example::

    python scripts/plot_ensemble.py \\
        --ensemble-dir ensembles/ \\
        --mu calibration/mu.json \\
        --natural-observables calibration/natural_observables.csv \\
        --save docs/ensemble_pairs.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import _SRC  # noqa: F401,E402  (puts src/ on sys.path)

from observables import sample_quality  # noqa: E402

LAMBDA_CMAP = plt.get_cmap("Blues")
LAMBDA_RANGE = (0.38, 0.95)
NATURAL_COLOR = "#b0afa8"
SURFACE = "#fcfcfb"
GRID_KWARGS = dict(color="#e2e1dc", linewidth=0.5, alpha=0.9)

LABELS = {
    "E_DCA": r"$E_{\mathrm{DCA}}$",
    "hamming_to_wt": "Hamming to WT",
    "P_S": r"$P(S\,|\,\mathrm{critic})$",
    "Q": r"$Q_\mu = E - \mu\,\mathrm{CDE}$",
    "CDE": "CDE (nats)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-dir", required=True, help="Directory of lam*/observables.csv")
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=None, help="Which lambdas to show (default: all found)"
    )
    parser.add_argument(
        "--mu",
        default=None,
        help="Weight of the CDE in the sample quality: a number, or a mu.json from calibrate_mu.py",
    )
    parser.add_argument(
        "--natural-observables",
        default=None,
        help="natural_observables.csv from calibrate_mu.py, drawn as a grey cloud",
    )
    parser.add_argument(
        "--quantities",
        nargs="+",
        default=["E_DCA", "hamming_to_wt", "P_S", "Q"],
        help="Columns to cross (default: E_DCA hamming_to_wt P_S Q)",
    )
    parser.add_argument("--threshold", type=float, default=0.80, help="P(S) threshold, drawn as a rule")
    parser.add_argument("--max-points", type=int, default=0, help="Subsample per ensemble (0 = all)")
    parser.add_argument("--save", default="ensemble_pairs.png", help="Output image")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--point-size", type=float, default=22.0, help="Marker area of the ensembles")
    parser.add_argument(
        "--natural-point-size", type=float, default=14.0, help="Marker area of the natural cloud"
    )
    parser.add_argument("--font-scale", type=float, default=1.35, help="Multiplier on every font size")
    parser.add_argument("--figsize", type=float, nargs=2, default=None, help="Figure size in inches")
    return parser


# --------------------------------------------------------------------- inputs


def load_mu(value: str | None) -> float | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        with open(path, encoding="utf-8") as handle:
            return float(json.load(handle)["mu"])
    return float(value)


def load_ensembles(directory: Path, wanted: List[float] | None) -> List[Tuple[float, pd.DataFrame]]:
    """Every ``lam*/observables.csv`` under ``directory``, sorted by lambda."""
    found: List[Tuple[float, pd.DataFrame]] = []
    for candidate in sorted(directory.glob("lam*")):
        path = candidate / "observables.csv"
        if not candidate.is_dir() or not path.is_file():
            continue
        try:
            lam = float(candidate.name[3:])
        except ValueError:
            continue
        if wanted is not None and not any(math.isclose(lam, w) for w in wanted):
            continue
        found.append((lam, pd.read_csv(path)))
    return sorted(found, key=lambda item: item[0])


def prepare(frame: pd.DataFrame, mu: float | None, max_points: int, seed: int = 0) -> pd.DataFrame:
    frame = frame.copy()
    if mu is not None and {"E_DCA", "CDE"} <= set(frame.columns):
        frame["Q"] = sample_quality(frame["E_DCA"].to_numpy(), frame["CDE"].to_numpy(), mu)
    if max_points and len(frame) > max_points:
        frame = frame.sample(max_points, random_state=seed)
    return frame


# -------------------------------------------------------------------- drawing


def lambda_colors(n: int) -> List[tuple]:
    positions = np.linspace(*LAMBDA_RANGE, n) if n > 1 else np.array([LAMBDA_RANGE[1]])
    return [LAMBDA_CMAP(p) for p in positions]


def main() -> None:
    args = build_parser().parse_args()
    directory = Path(args.ensemble_dir)
    ensembles = load_ensembles(directory, args.lambdas)
    if not ensembles:
        raise SystemExit(f"No lam*/observables.csv found under {directory}")

    scale = args.font_scale
    plt.rcParams.update(
        {
            "font.size": 10 * scale,
            "axes.titlesize": 12 * scale,
            "axes.labelsize": 12 * scale,
            "xtick.labelsize": 9.5 * scale,
            "ytick.labelsize": 9.5 * scale,
            "legend.fontsize": 11 * scale,
        }
    )

    mu = load_mu(args.mu)
    quantities = list(args.quantities)
    if "Q" in quantities and mu is None:
        print("WARNING: no --mu given, dropping the sample quality from the figure", file=sys.stderr)
        quantities.remove("Q")

    frames = [(lam, prepare(frame, mu, args.max_points)) for lam, frame in ensembles]
    missing = [q for q in quantities if q not in frames[0][1].columns]
    if missing:
        raise SystemExit(f"The ensembles have no column {missing}; available: {list(frames[0][1].columns)}")

    natural = None
    if args.natural_observables:
        natural = prepare(pd.read_csv(args.natural_observables), mu, args.max_points)

    colors = lambda_colors(len(frames))
    n = len(quantities)
    figsize = tuple(args.figsize) if args.figsize else (3.5 * n + 1.4, 3.4 * n)
    figure, axes = plt.subplots(n, n, figsize=figsize, squeeze=False, constrained_layout=True)
    figure.patch.set_facecolor(SURFACE)

    for row, y_name in enumerate(quantities):
        for column, x_name in enumerate(quantities):
            axis = axes[row][column]
            _style(axis)
            if column > row:
                axis.set_visible(False)
                continue
            if row == column:
                _diagonal(axis, frames, natural, y_name, colors)
            else:
                _scatter(
                    axis, frames, natural, x_name, y_name, colors, args.threshold,
                    point_size=args.point_size, natural_point_size=args.natural_point_size,
                )
            if row == n - 1:
                axis.set_xlabel(LABELS.get(x_name, x_name))
            if column == 0 and row > 0:
                axis.set_ylabel(LABELS.get(y_name, y_name))
            if row == column == 0:
                axis.set_ylabel("density")

    _legend(figure, frames, colors, natural is not None, mu)
    title = "Final sequences of 1000 chains per lambda"
    if mu is not None:
        title += rf"   ($\mu = {mu:.3g}$)"
    figure.suptitle(title, fontsize=15 * scale)

    save = Path(args.save)
    save.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save, dpi=args.dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    print(f"figure written to {save}")


def _scatter(
    axis, frames, natural, x_name, y_name, colors, threshold,
    point_size: float = 22.0, natural_point_size: float = 14.0,
) -> None:
    if natural is not None and {x_name, y_name} <= set(natural.columns):
        axis.scatter(
            natural[x_name], natural[y_name], s=natural_point_size, color=NATURAL_COLOR,
            alpha=0.35, linewidths=0, zorder=1,
        )
    for (lam, frame), color in zip(frames, colors):
        axis.scatter(
            frame[x_name], frame[y_name], s=point_size, color=color, alpha=0.55,
            linewidths=0, zorder=2,
        )
    if x_name == "P_S":
        axis.axvline(threshold, color="#52514e", linestyle="--", linewidth=1.4, zorder=3)
    if y_name == "P_S":
        axis.axhline(threshold, color="#52514e", linestyle="--", linewidth=1.4, zorder=3)


def _diagonal(axis, frames, natural, name, colors) -> None:
    values = [frame[name].to_numpy() for _, frame in frames]
    if natural is not None and name in natural.columns:
        values.append(natural[name].to_numpy())
    low = min(float(np.min(v)) for v in values)
    high = max(float(np.max(v)) for v in values)
    bins = np.linspace(low, high, 46) if high > low else 10
    if natural is not None and name in natural.columns:
        axis.hist(natural[name], bins=bins, color=NATURAL_COLOR, alpha=0.45, density=True)
    for (lam, frame), color in zip(frames, colors):
        axis.hist(frame[name], bins=bins, color=color, alpha=0.55, density=True)
    axis.set_yticks([])


def _legend(figure, frames, colors, has_natural, mu) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=color, marker="o", markersize=10, linestyle="none", label=rf"$\lambda = {lam:g}$")
        for (lam, _), color in zip(frames, colors)
    ]
    if has_natural:
        handles.append(
            Line2D([], [], color=NATURAL_COLOR, marker="o", markersize=10, linestyle="none",
                   label="natural alignment")
        )
    figure.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.97),
        frameon=False,
        markerscale=1.3,
    )


def _style(axis) -> None:
    axis.grid(True, **GRID_KWARGS)
    axis.set_axisbelow(True)
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#a9a8a2")
    axis.tick_params(colors="#52514e")


if __name__ == "__main__":
    main()
