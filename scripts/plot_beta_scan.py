#!/usr/bin/env python
"""Does cooling the Potts term buy anything the bias cannot?

Six panels over the ``(beta, lambda)`` grid, every ensemble rescored at
``beta = 1`` because that is where ``mu`` was calibrated and a ``beta != 1``
conditional is not the same observable.

With the critic pinned at ``beta = 1`` the sampled measure is
``exp(-beta [E - lambda_tilde log P(S)])`` with ``lambda_tilde = lambda / beta``,
so **lambda_tilde says what is being optimised and beta only says how cold**.
The top row is therefore read against ``lambda_tilde``: if cooling were doing
nothing, every marker would fall on the ``beta = 1`` curve. The bottom row is
read against the energy, which is the quantity the designs are meant to keep
comparable to the wild type's: at a given energy, what specificity, what
distance from the natural S-carriers, and what is left of the library.

Two marks, so the reference is never in doubt:

* **circles joined by a solid line** -- ``beta = 1``, the thermal reference;
* **diamonds** -- every cooled run, joined by a dashed line once ``beta >= 5``,
  where the sampler has reached its zero-temperature limit and ``beta = 5`` and
  ``beta = 10`` land on top of each other.

Example::

    python scripts/plot_beta_scan.py \\
        --summary ../results_pcd/beta_scan_summary.csv \\
        --save docs/beta_scan_pcd.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import _SRC  # noqa: F401,E402  (puts src/ on sys.path)

from plot_ensemble import NATURAL_COLOR, SURFACE, _style  # noqa: E402

INK = "#52514e"
CONTROL = "#c2703d"  # the lambda = 0 control, the one orange in the figure
RAMP = plt.get_cmap("Blues")
QUENCH_FROM = 5.0  # beta at or above which the sampler is at its zero-temperature limit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", required=True, help="beta_scan_summary.csv")
    parser.add_argument("--natural-q", type=float, nargs=2, default=(-79.1, 16.5),
                        help="Mean and sd of the natural sample quality")
    parser.add_argument("--natural-e", type=float, default=289.6, help="Weighted natural mean energy")
    parser.add_argument("--natural-ps", type=float, default=0.110, help="Weighted natural mean P(S)")
    parser.add_argument("--natural-nbr", type=float, default=38.9,
                        help="Natural median distance to the nearest natural S-carrier, %%")
    parser.add_argument("--natural-diversity", type=float, default=21.6,
                        help="Natural mean pairwise Hamming over the mutable columns")
    parser.add_argument("--wt-e", type=float, default=143.7, help="Energy of the wild type")
    parser.add_argument("--wt-q", type=float, default=-100.0, help="Sample quality of the wild type")
    parser.add_argument("--save", default="docs/beta_scan_pcd.png")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--font-scale", type=float, default=1.15)
    parser.add_argument("--figsize", type=float, nargs=2, default=(16.5, 9.6))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    table = pd.read_csv(args.summary)
    betas = sorted(table["beta"].unique())
    colors = {b: RAMP(p) for b, p in zip(betas, np.linspace(0.40, 1.0, len(betas)))}
    q_mean, q_sd = args.natural_q

    biased = table[table["lam_tilde"] > 0]
    warm = biased[biased["beta"] == 1].sort_values("lam_tilde")
    quench = biased[biased["beta"] == QUENCH_FROM].sort_values("lam_tilde")
    control = table[table["lam_tilde"] == 0].iloc[0]  # the lambda = 0, beta = 1 control

    scale = args.font_scale
    plt.rcParams.update({
        "font.size": 10 * scale, "axes.titlesize": 11.5 * scale, "axes.labelsize": 11 * scale,
        "xtick.labelsize": 9 * scale, "ytick.labelsize": 9 * scale, "legend.fontsize": 9.5 * scale,
    })
    figure, axes = plt.subplots(2, 3, figsize=tuple(args.figsize), constrained_layout=True)
    figure.patch.set_facecolor(SURFACE)

    def draw(axis, x_name: str, y_name: str, logx: bool) -> None:
        """The two curves and every marker, warm as circles and cooled as diamonds."""
        axis.plot(warm[x_name], warm[y_name], color=colors[1.0], linewidth=1.8, alpha=0.9, zorder=3)
        axis.plot(quench[x_name], quench[y_name], color=colors[QUENCH_FROM], linewidth=1.8,
                  linestyle="--", alpha=0.9, zorder=3)
        for beta in betas:
            rows = biased[biased["beta"] == beta].sort_values(x_name)
            warm_row = beta == 1
            axis.scatter(rows[x_name], rows[y_name], s=140 if warm_row else 95,
                         marker="o" if warm_row else "D", color=colors[beta],
                         edgecolor=SURFACE, linewidth=1.8, zorder=4)
        if logx:
            axis.set_xscale("log")

    def rule(axis, value, color, style, horizontal: bool, label=None, below: bool = False) -> None:
        line = axis.axhline if horizontal else axis.axvline
        line(value, color=color, linestyle=style, linewidth=1.6, zorder=1)
        if label:
            if horizontal:
                axis.annotate(label, xy=(0.015, value), xycoords=("axes fraction", "data"),
                              xytext=(0, -11 if below else 3), textcoords="offset points",
                              color=color, fontsize=8.6 * scale)
            else:
                axis.annotate(label, xy=(value, 0.985), xycoords=("data", "axes fraction"),
                              xytext=(3, -11), textcoords="offset points", color=color,
                              fontsize=8.6 * scale)

    TILDE = r"$\tilde\lambda = \lambda / \beta$"
    ENERGY = r"$\langle E_{\mathrm{DCA}}\rangle$"

    # ------------------------------------------------- top row: against the bias strength
    axis = axes[0][0]
    _style(axis)
    rule(axis, args.natural_e, NATURAL_COLOR, "-", True, "natural")
    rule(axis, args.wt_e, INK, ":", True, "wild type")
    rule(axis, control["E"], CONTROL, "-.", True, r"$\lambda = 0$ control")
    draw(axis, "lam_tilde", "E", logx=True)
    axis.set_ylim(top=args.natural_e + 14)
    axis.set_xlabel(TILDE); axis.set_ylabel(ENERGY)
    axis.set_title("Energy")

    axis = axes[0][1]
    _style(axis)
    axis.axhspan(q_mean - q_sd, q_mean + q_sd, color=NATURAL_COLOR, alpha=0.30, linewidth=0, zorder=0)
    rule(axis, q_mean, NATURAL_COLOR, "-", True, "natural band")
    rule(axis, args.wt_q, INK, ":", True, "wild type")
    rule(axis, control["Q"], CONTROL, "-.", True, r"$\lambda = 0$ control")
    draw(axis, "lam_tilde", "Q", logx=True)
    axis.set_xlabel(TILDE); axis.set_ylabel(r"$\langle Q_\mu\rangle = \langle E - \mu\,\mathrm{CDE}\rangle$")
    axis.set_title("Sample quality")

    axis = axes[0][2]
    _style(axis)
    rule(axis, args.natural_ps, NATURAL_COLOR, "-", True, "natural", below=True)
    draw(axis, "lam_tilde", "P_S", logx=True)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel(TILDE); axis.set_ylabel(r"$\langle P(S\,|\,\mathrm{critic})\rangle$")
    axis.set_title("Specificity")

    # ------------------------------------------- bottom row: what a given energy buys
    axis = axes[1][0]
    _style(axis)
    rule(axis, args.wt_e, INK, ":", False, "WT")
    rule(axis, control["E"], CONTROL, "-.", False, r"$\lambda = 0$")
    draw(axis, "E", "P_S", logx=False)
    axis.set_xlabel(ENERGY); axis.set_ylabel(r"$\langle P(S\,|\,\mathrm{critic})\rangle$")
    axis.set_title("What an energy buys in specificity")

    axis = axes[1][1]
    _style(axis)
    rule(axis, args.natural_nbr, NATURAL_COLOR, "-", True, "natural median")
    rule(axis, args.wt_e, INK, ":", False)
    rule(axis, control["E"], CONTROL, "-.", False)
    draw(axis, "E", "nbr_pct", logx=False)
    axis.set_xlabel(ENERGY); axis.set_ylabel("distance to nearest natural\nS-carrier (% of mutable sites)")
    axis.set_title("...and in distance from the family")

    axis = axes[1][2]
    _style(axis)
    rule(axis, args.natural_diversity, NATURAL_COLOR, "-", True, "natural alignment")
    rule(axis, control["mean_pair_hamming"], CONTROL, "-.", True, r"$\lambda = 0$ control")
    rule(axis, args.wt_e, INK, ":", False)
    draw(axis, "E", "mean_pair_hamming", logx=False)
    axis.set_ylim(0, args.natural_diversity * 1.15)
    axis.set_xlabel(ENERGY); axis.set_ylabel("mean pairwise Hamming inside the library")
    axis.set_title("...and what is left of the library")

    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=colors[1.0], marker="o", markersize=10, linewidth=1.8,
                      label=r"$\beta = 1$  (thermal reference)")]
    handles += [Line2D([], [], color=colors[b], marker="D", markersize=8, linestyle="none",
                       label=rf"$\beta = {b:g}$") for b in betas if b != 1]
    handles += [
        Line2D([], [], color=colors[QUENCH_FROM], linestyle="--", linewidth=1.8,
               label=r"$\beta \geq 5$: zero-temperature limit"),
        Line2D([], [], color=NATURAL_COLOR, linewidth=1.8, label="natural alignment"),
        Line2D([], [], color=CONTROL, linestyle="-.", linewidth=1.8, label=r"$\lambda = 0$ control"),
        Line2D([], [], color=INK, linestyle=":", linewidth=1.8, label="wild type"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False,
                  columnspacing=1.8)
    figure.suptitle(
        r"Cooling the Potts term: $H = \beta E_{\mathrm{DCA}} - \lambda \log P(S\,|\,\mathrm{critic})$, "
        r"the critic always at $\beta = 1$;  every ensemble rescored at $\beta = 1$",
        fontsize=13.5 * scale,
    )

    save = Path(args.save)
    save.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save, dpi=args.dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    print(f"figure written to {save}")


if __name__ == "__main__":
    main()
