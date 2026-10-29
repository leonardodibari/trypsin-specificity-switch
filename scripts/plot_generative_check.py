#!/usr/bin/env python
"""Does unbiased resampling reach the statistics the model was trained on?

Two rows of panels against the number of proposed moves per chain:

* **the agreement with the natural statistics** -- the Pearson coefficients of
  the one-site frequencies, the two-site frequencies and the connected
  correlations, each against its *finite-sample ceiling*: the value that
  ``n_chains`` sequences drawn from the natural alignment itself reach, which is
  what a perfect sampler could achieve with this many chains;
* **the per-sequence observables** -- energy, Hamming distance to the wild type,
  ``P(S | critic)`` and context entropy, each against the weighted natural mean,
  with the inter-quartile band across chains.

A converged check has the first row flat against its ceilings and the second row
flat against its dotted rules.

Example::

    python scripts/plot_generative_check.py \\
        --run-dir ../results_generative --log-x \\
        --save docs/generative_check.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import _SRC  # noqa: F401,E402  (puts src/ on sys.path)

from plot_ensemble import GRID_KWARGS, SURFACE, _style  # noqa: E402

RUN_COLORS = ["#2f6ea8", "#c2703d", "#8a6bab", "#5a9e6f"]
NOGAP_ALPHA = 0.55
CEILING = "#52514e"
NATURAL = "#b0afa8"

PEARSONS = [
    ("pearson_f1", r"Pearson $f_i$"),
    ("pearson_f2", r"Pearson $f_{ij}$  ($i \neq j$)"),
    ("pearson_cij", r"Pearson $C_{ij}$  ($i \neq j$)"),
]
OBSERVABLES = [
    ("E_DCA", r"$E_{\mathrm{DCA}}$"),
    ("hamming_to_wt", "Hamming to WT"),
    ("P_S", r"$P(S\,|\,\mathrm{critic})$"),
    ("CDE", "CDE (nats)"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir", required=True, nargs="+", help="One or more output directories of run_generative_check.py"
    )
    parser.add_argument(
        "--labels", nargs="+", default=None, help="Legend label per run (default: its 'start' setting)"
    )
    parser.add_argument("--x", default="proposed", choices=("proposed", "accepted"), help="Horizontal axis")
    parser.add_argument("--log-x", action="store_true", help="Logarithmic horizontal axis")
    parser.add_argument("--no-nogap", action="store_true", help="Hide the non-gap variants")
    parser.add_argument("--save", default="generative_check.png", help="Output image")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--font-scale", type=float, default=1.25)
    parser.add_argument("--figsize", type=float, nargs=2, default=(16.0, 8.0))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = []
    for position, name in enumerate(args.run_dir):
        directory = Path(name)
        with open(directory / "summary.json", encoding="utf-8") as handle:
            summary = json.load(handle)
        label = args.labels[position] if args.labels and position < len(args.labels) else _label(summary)
        runs.append((label, pd.read_csv(directory / "trace.csv"), summary))

    summary = runs[0][2]
    ceiling = summary.get("sampling_ceiling") or {}
    natural = summary.get("natural_weighted") or {}
    x_label = "proposed moves per chain" if args.x == "proposed" else "accepted moves per chain"

    scale = args.font_scale
    plt.rcParams.update(
        {
            "font.size": 10 * scale,
            "axes.titlesize": 11.5 * scale,
            "axes.labelsize": 11.5 * scale,
            "xtick.labelsize": 9 * scale,
            "ytick.labelsize": 9 * scale,
            "legend.fontsize": 10 * scale,
        }
    )

    columns = max(len(PEARSONS), len(OBSERVABLES))
    figure, axes = plt.subplots(2, columns, figsize=tuple(args.figsize), constrained_layout=True)
    figure.patch.set_facecolor(SURFACE)

    for index in range(columns):
        axis = axes[0][index]
        if index >= len(PEARSONS):
            axis.set_visible(False)
            continue
        name, label = PEARSONS[index]
        _style(axis)
        low = 0.0
        for position, (run_label, trace, _) in enumerate(runs):
            color = RUN_COLORS[position % len(RUN_COLORS)]
            steps = trace["step"] if args.x == "proposed" else trace["accepted_per_chain"]
            axis.plot(steps, trace[name], color=color, linewidth=2.0, zorder=3)
            low = min(low, float(trace[name].min()))
            if not args.no_nogap and f"{name}_nogap" in trace:
                axis.plot(
                    steps, trace[f"{name}_nogap"], color=color, linewidth=1.5, linestyle="-.",
                    alpha=NOGAP_ALPHA, zorder=3,
                )
        if name in ceiling:
            axis.axhline(
                ceiling[name]["mean"], color=CEILING, linestyle="--", linewidth=1.4, zorder=2,
            )
        axis.set_ylim(low, 1.02)
        axis.set_title(label)
        axis.set_xlabel(x_label)
        if args.log_x:
            axis.set_xscale("log")

    for index in range(columns):
        axis = axes[1][index]
        if index >= len(OBSERVABLES):
            axis.set_visible(False)
            continue
        name, label = OBSERVABLES[index]
        _style(axis)
        for position, (run_label, trace, _) in enumerate(runs):
            color = RUN_COLORS[position % len(RUN_COLORS)]
            steps = trace["step"] if args.x == "proposed" else trace["accepted_per_chain"]
            if f"{name}_q25" in trace:
                axis.fill_between(
                    steps, trace[f"{name}_q25"], trace[f"{name}_q75"], color=color,
                    alpha=0.16, linewidth=0, zorder=1,
                )
            axis.plot(steps, trace[f"{name}_mean"], color=color, linewidth=2.0, zorder=3)
        if name in natural:
            axis.axhline(natural[name]["mean"], color=CEILING, linestyle=":", linewidth=1.6, zorder=2)
        axis.set_title(label)
        axis.set_xlabel(x_label)
        if args.log_x:
            axis.set_xscale("log")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=RUN_COLORS[position % len(RUN_COLORS)], linewidth=2.4, label=run_label)
        for position, (run_label, _, _) in enumerate(runs)
    ]
    if not args.no_nogap:
        handles.append(
            Line2D([], [], color=CEILING, linewidth=1.6, linestyle="-.", alpha=NOGAP_ALPHA,
                   label="same, non-gap states only")
        )
    handles += [
        Line2D([], [], color=CEILING, linewidth=1.4, linestyle="--",
               label=f"finite-sample ceiling at M = {summary['n_chains']}"),
        Line2D([], [], color=CEILING, linewidth=1.6, linestyle=":", label="natural alignment (weighted)"),
    ]
    figure.legend(
        handles=handles, loc="outside lower center", ncol=min(3, len(handles)),
        frameon=False, columnspacing=2.2,
    )

    title = (
        rf"Unbiased resampling against the training statistics  "
        rf"($\lambda = {summary['lam']:g}$, $\beta = {summary['beta']:g}$, "
        rf"{summary['n_chains']} chains, {summary['n_mutable_columns']} mutable columns)"
    )
    figure.suptitle(title, fontsize=14 * scale)

    save = Path(args.save)
    save.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save, dpi=args.dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    print(f"figure written to {save}")


def _label(summary: dict) -> str:
    """How a run describes itself: where its chains started."""
    return {"wt": "started at the wild type", "natural": "started at natural sequences"}.get(
        summary.get("start", "wt"), str(summary.get("start"))
    )


if __name__ == "__main__":
    main()
