#!/usr/bin/env python
"""What stamping the target residue into the critic site costs, chain by chain.

Two panels, each a histogram per lambda over the chains of an ensemble:

* ``Delta E_DCA`` -- the energy of the substituted sequence minus the energy of
  the sequence the chain actually ended on;
* ``Delta Q_mu``  -- the same difference in sample quality, ``E - mu * CDE``,
  which is ``Delta E - mu * Delta CDE`` and so also carries the change in context
  entropy.

Positive means the substitution made the sequence worse by that measure. The
colours are the ones ``plot_ensemble.py`` uses, light to dark with lambda.

A chain that already ended on the target residue has a difference of exactly
zero, and at large lambda most of them do; those chains are left out of the
histograms, with their share reported in the legend, so that the distribution of
the chains that actually changed stays legible. ``--include-unchanged`` puts them
back.

Example::

    python scripts/plot_critic_deltas.py \\
        --ensemble-dir ../results/ensembles \\
        --substituted-dir ../results_serine/ensembles \\
        --mu ../results/calibration/mu.json \\
        --save docs/critic_substitution_deltas.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import _SRC  # noqa: F401,E402  (puts src/ on sys.path)

from observables import sample_quality  # noqa: E402
from plot_ensemble import (  # noqa: E402
    GRID_KWARGS,
    SURFACE,
    lambda_colors,
    load_ensembles,
    load_mu,
    _style,
)

PANELS = [
    ("dE", r"$\Delta E_{\mathrm{DCA}}$"),
    ("dQ", r"$\Delta Q_\mu = \Delta E - \mu\,\Delta \mathrm{CDE}$"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-dir", required=True, help="The sampled ensembles, before substitution")
    parser.add_argument(
        "--substituted-dir", required=True, help="The same ensembles after substitute_critic.py"
    )
    parser.add_argument("--mu", required=True, help="A number, or a mu.json from calibrate_mu.py")
    parser.add_argument("--lambdas", type=float, nargs="+", default=None, help="Which lambdas to show")
    parser.add_argument("--bins", type=int, default=46, help="Histogram bins per panel (default: 46)")
    parser.add_argument(
        "--include-unchanged",
        action="store_true",
        help="Keep the chains that already carried the target residue (a spike at exactly zero)",
    )
    parser.add_argument("--save", default="critic_substitution_deltas.png", help="Output image")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--font-scale", type=float, default=1.35, help="Multiplier on every font size")
    parser.add_argument("--figsize", type=float, nargs=2, default=(13.0, 5.2), help="Figure size in inches")
    return parser


def deltas(
    before: Path, after: Path, mu: float, wanted: List[float] | None
) -> List[Tuple[float, pd.DataFrame]]:
    """``(lambda, frame)`` with the per-chain differences, matched on ``chain``."""
    originals = dict(load_ensembles(before, wanted))
    substituted = dict(load_ensembles(after, wanted))
    shared = sorted(set(originals) & set(substituted))
    if not shared:
        raise SystemExit(f"No lambda is present in both {before} and {after}")

    out = []
    for lam in shared:
        first = originals[lam].set_index("chain").sort_index()
        second = substituted[lam].set_index("chain").sort_index()
        if not first.index.equals(second.index):
            raise SystemExit(f"lambda {lam:g}: the two ensembles do not hold the same chains")
        frame = pd.DataFrame(
            {
                "dE": second["E_DCA"].to_numpy() - first["E_DCA"].to_numpy(),
                "dCDE": second["CDE"].to_numpy() - first["CDE"].to_numpy(),
            }
        )
        frame["dQ"] = sample_quality(frame["dE"].to_numpy(), frame["dCDE"].to_numpy(), mu)
        if "critic_residue_before" in second.columns:
            frame["was_target"] = (second["critic_residue_before"].to_numpy() == "S")
        out.append((lam, frame))
    return out


def main() -> None:
    args = build_parser().parse_args()
    mu = load_mu(args.mu)
    frames = deltas(Path(args.ensemble_dir), Path(args.substituted_dir), mu, args.lambdas)
    colors = lambda_colors(len(frames))

    drawn = frames
    if not args.include_unchanged and all("was_target" in frame for _, frame in frames):
        drawn = [(lam, frame[~frame["was_target"]]) for lam, frame in frames]
        empty = [f"{lam:g}" for lam, frame in drawn if frame.empty]
        if empty:
            raise SystemExit(
                f"Every chain of lambda {', '.join(empty)} already carried the target residue; "
                "pass --include-unchanged to draw them anyway"
            )

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

    figure, axes = plt.subplots(1, len(PANELS), figsize=tuple(args.figsize), constrained_layout=True)
    figure.patch.set_facecolor(SURFACE)

    for axis, (column, label) in zip(np.atleast_1d(axes), PANELS):
        _style(axis)
        values = [frame[column].to_numpy() for _, frame in drawn]
        low = min(float(np.min(v)) for v in values)
        high = max(float(np.max(v)) for v in values)
        bins = np.linspace(low, high, args.bins) if high > low else 10
        for (lam, frame), color in zip(drawn, colors):
            axis.hist(frame[column], bins=bins, color=color, alpha=0.55, density=True)
        axis.axvline(0.0, color="#52514e", linestyle="--", linewidth=1.4, zorder=3)
        axis.set_xlabel(label)
        axis.set_ylabel("density")

    handles = []
    for (lam, full), (_, shown), color in zip(frames, drawn, colors):
        label = rf"$\lambda = {lam:g}$"
        if "was_target" in full:
            label += f"  ({100 * (1 - len(shown) / len(full)):.0f} % already S)"
        handles.append(
            plt.Line2D([], [], color=color, marker="o", markersize=10, linestyle="none", label=label)
        )
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(3, len(handles)),
        frameon=False,
        markerscale=1.3,
        columnspacing=1.6,
        handletextpad=0.4,
    )
    subtitle = "" if args.include_unchanged else ", over the chains that did not already carry it"
    figure.suptitle(
        rf"Cost of writing S into the critic site of every final sequence{subtitle}"
        "\n"
        rf"($\mu = {mu:.3g}$)",
        fontsize=15 * scale,
    )

    save = Path(args.save)
    save.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save, dpi=args.dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    print(f"figure written to {save}")

    for (lam, full), (_, shown) in zip(frames, drawn):
        share = f"{100 * full['was_target'].mean():5.1f} % already S" if "was_target" in full else ""
        print(
            f"lambda = {lam:6g}: {share} | all chains  dE {full['dE'].mean():+6.2f}  "
            f"dQ {full['dQ'].mean():+6.2f} | changed only  dE median {shown['dE'].median():+6.2f} "
            f"mean {shown['dE'].mean():+6.2f}  dQ median {shown['dQ'].median():+6.2f} "
            f"mean {shown['dQ'].mean():+6.2f}"
        )


if __name__ == "__main__":
    main()
