#!/usr/bin/env python
"""How the ensemble's observables move while the chains run.

One panel per quantity -- energy, Hamming distance to the wild type,
``P(S | critic site)`` and the sample quality ``Q_mu = E - mu * CDE`` -- each
showing the batch mean as a line and the inter-quartile range across chains as
a band, with one colour per lambda. The x axis is the mean number of *accepted*
moves per chain (``--x proposed`` switches to proposed steps).

Reads the ``trace.csv`` written by ``run_ensemble.py --trace-every``.

Example::

    python scripts/plot_ensemble_trace.py \\
        --ensemble-dir ensembles/ \\
        --mu calibration/mu.json \\
        --natural-observables calibration/natural_observables.csv \\
        --save docs/ensemble_trace.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import _SRC  # noqa: F401,E402  (puts src/ on sys.path)

LAMBDA_CMAP = plt.get_cmap("Blues")
LAMBDA_RANGE = (0.42, 0.97)
NATURAL_COLOR = "#8a8984"
RULE_COLOR = "#52514e"
SURFACE = "#fcfcfb"
GRID_KWARGS = dict(color="#e2e1dc", linewidth=0.6, alpha=0.9)

PANELS = [
    ("E_DCA", r"$E_{\mathrm{DCA}}$"),
    ("hamming_to_wt", "Hamming distance to WT"),
    ("P_S", r"$P(S\,|\,\mathrm{critic\ site})$"),
    ("Q", r"$Q_\mu = E_{\mathrm{DCA}} - \mu\,\mathrm{CDE}$"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-dir", required=True, help="Directory of lam*/trace.csv")
    parser.add_argument("--lambdas", type=float, nargs="+", default=None, help="Which lambdas to draw")
    parser.add_argument("--mu", default=None, help="A number or a mu.json, for the sample-quality panel")
    parser.add_argument(
        "--natural-observables", default=None, help="natural_observables.csv, drawn as a grey rule per panel"
    )
    parser.add_argument("--threshold", type=float, default=0.80, help="P(S) threshold rule")
    parser.add_argument(
        "--natural-rules",
        nargs="*",
        default=["E_DCA", "P_S", "Q", "CDE"],
        help="Panels that get the natural mean as a rule (the family's distance to one wild "
        "type is not a target, so Hamming is left out by default)",
    )
    parser.add_argument(
        "--x",
        default="accepted",
        choices=("accepted", "proposed"),
        help="x axis: mean accepted moves per chain (default) or proposed steps",
    )
    parser.add_argument("--log-x", action="store_true", help="Logarithmic x axis")
    parser.add_argument("--save", default="ensemble_trace.png", help="Output image")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--figsize", type=float, nargs=2, default=(13.5, 9.5))
    parser.add_argument("--font-scale", type=float, default=1.25, help="Multiplier on every font size")
    parser.add_argument("--line-width", type=float, default=2.4)
    return parser


def load_traces(directory: Path, wanted: List[float] | None) -> List[Tuple[float, pd.DataFrame]]:
    found: List[Tuple[float, pd.DataFrame]] = []
    for candidate in sorted(directory.glob("lam*")):
        path = candidate / "trace.csv"
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


def load_mu(value: str | None) -> float | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        with open(path, encoding="utf-8") as handle:
            return float(json.load(handle)["mu"])
    return float(value)


def add_quality(frame: pd.DataFrame, mu: float | None) -> pd.DataFrame:
    """``Q = E - mu * CDE`` for the mean and, approximately, for the quartiles.

    The quartiles of Q are not functions of the quartiles of E and CDE, but the
    two move together along a chain, so the band built this way is a fair
    picture of the spread; the mean line is exact.
    """
    if mu is None or "CDE_mean" not in frame.columns:
        return frame
    for suffix in ("mean", "q25", "q75"):
        if f"E_DCA_{suffix}" in frame.columns and f"CDE_{suffix}" in frame.columns:
            frame[f"Q_{suffix}"] = frame[f"E_DCA_{suffix}"] - mu * frame[f"CDE_{suffix}"]
    return frame


def main() -> None:
    args = build_parser().parse_args()
    scale = args.font_scale
    plt.rcParams.update(
        {
            "font.size": 10 * scale,
            "axes.titlesize": 12 * scale,
            "axes.labelsize": 12 * scale,
            "xtick.labelsize": 10 * scale,
            "ytick.labelsize": 10 * scale,
            "legend.fontsize": 11 * scale,
        }
    )

    directory = Path(args.ensemble_dir)
    traces = load_traces(directory, args.lambdas)
    if not traces:
        raise SystemExit(
            f"No lam*/trace.csv under {directory} -- run run_ensemble.py with --trace-every"
        )
    mu = load_mu(args.mu)
    traces = [(lam, add_quality(frame, mu)) for lam, frame in traces]

    natural = pd.read_csv(args.natural_observables) if args.natural_observables else None
    panels = [(name, label) for name, label in PANELS if f"{name}_mean" in traces[0][1].columns]

    colors = lambda_colors(len(traces))
    rows = 2
    columns = int(np.ceil(len(panels) / rows))
    figure, axes = plt.subplots(rows, columns, figsize=tuple(args.figsize), squeeze=False)
    figure.patch.set_facecolor(SURFACE)

    x_column = "accepted_per_chain" if args.x == "accepted" else "step"
    x_label = "accepted moves per chain" if args.x == "accepted" else "proposed steps"

    for index, (name, label) in enumerate(panels):
        axis = axes[index // columns][index % columns]
        _style(axis)
        for (lam, frame), color in zip(traces, colors):
            x = frame[x_column].to_numpy()
            axis.fill_between(
                x, frame[f"{name}_q25"], frame[f"{name}_q75"], color=color, alpha=0.18, linewidth=0
            )
            axis.plot(x, frame[f"{name}_mean"], color=color, linewidth=args.line_width,
                      label=rf"$\lambda = {lam:g}$")
        if natural is not None and name in natural.columns and name in args.natural_rules:
            axis.axhline(
                float(natural[name].mean()), color=NATURAL_COLOR, linestyle=":", linewidth=2.0,
            )
        if name == "P_S":
            axis.axhline(args.threshold, color=RULE_COLOR, linestyle="--", linewidth=1.6)
            axis.set_ylim(0, 1.02)
        if args.log_x:
            axis.set_xscale("symlog", linthresh=1)
            axis.set_xlim(left=0)  # a symlog axis otherwise draws its mirror half
        axis.set_xlabel(x_label)
        axis.set_ylabel(label)

    for index in range(len(panels), rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if natural is not None:
        from matplotlib.lines import Line2D

        handles.append(
            Line2D([], [], color=NATURAL_COLOR, linestyle=":", linewidth=2.0, label="natural mean")
        )
        labels.append("natural mean")
    figure.legend(
        handles, labels, loc="lower center", ncol=min(len(handles), 8), frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    title = "Ensemble observables along the run"
    if mu is not None:
        title += rf"   ($\mu = {mu:.4g}$)"
    figure.suptitle(title, fontsize=15 * scale)
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))

    save = Path(args.save)
    save.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save, dpi=args.dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    print(f"figure written to {save}")


def lambda_colors(n: int) -> List[tuple]:
    positions = np.linspace(*LAMBDA_RANGE, n) if n > 1 else np.array([LAMBDA_RANGE[1]])
    return [LAMBDA_CMAP(p) for p in positions]


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
