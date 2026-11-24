"""S3 -- the selected runs as a 5x5 corner plot.

The analogue of ``figures_ladder/fig02_corner_grafting.py`` and ``fig06``, for the
runs ``selection.select()`` picks out of the 48. Five observables, every pair in
the lower triangle, each run's own marginal on the diagonal, the legend in the
empty upper triangle so it can never sit on top of data.

POINTS ARE DRAWN LARGER than in the ladder's corner plots (``s=3.4, alpha=0.22``
there). Six ensembles rather than one condition's lambda ramp means far fewer
overlapping clouds, so the small-and-faint setting that keeps a 4000-point
scatter readable is no longer what this needs: here the question is where six
ensembles sit relative to each other, and the points have to be visible
individually to answer it.

THE FIFTH AXIS, and why it is here at all. ``fig06`` DROPS the
natural-neighbour distance when it compares across site sets, because
``add_natural_neighbour.py`` scores it over each block's own allowed columns and
"% of 14 columns" and "% of 79 columns" are not the same quantity
(CLAMP_CONTROL caveat 3). That caveat is about the scoring, not about the
observable, and it has a fix: all three sets of this study are subsets of the
grafting window, so scoring every selected run over that ONE common 79-column
window makes the axis mean the same thing everywhere. ``s03_prepare.sh`` does
exactly that, and the figure refuses to draw the axis if it finds a column that
was scored any other way.

THE WILD TYPE is marked wherever both axes are defined for it. It has no
distance to the nearest natural S-carrying sequence -- it carries D at the
critic -- so it is absent from that row and column, and the figure says so
rather than placing it at zero.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import selection
import style

#: The five, in the order the ladder uses them.
VARS = ["hamming_to_wt", "P_S", "E_DCA", "Q", "min_hamming_natural_S_pct"]

#: Shorter axis labels than style.LABELS, which were written for a figure with
#: fewer columns: at 5x5 the rightmost label runs off the canvas.
SHORT_LABELS = {
    "hamming_to_wt": r"$d_H$ to wild type",
    "min_hamming_natural_S_pct": "to nearest nat. S (%)",
}

#: The four that survive without the natural-neighbour column, for the fallback.
VARS_4 = VARS[:4]

#: Colour says which mutation set; marker separates runs within a set.
SET_COLOUR = {"redsector": style.BLUE, "coupling20": style.ORANGE,
              "segment": style.AQUA}
MARKERS = ["o", "s", "^", "D", "v", "P"]

POINT_SIZE = 11.0      # vs 3.4 in fig02 -- see the module docstring
POINT_ALPHA = 0.45     # vs 0.22


def wild_type_values() -> dict:
    """The wild type's own position on each axis, where it has one."""
    from observables import context_dependent_entropy

    model = data.model()
    sequence = data.wild_type()
    cde = float(context_dependent_entropy(model, np.asarray([sequence]), beta=1.0)[0])
    energy = float(model.energy(sequence))
    return {
        "hamming_to_wt": 0.0,
        "E_DCA": energy,
        "Q": energy - config.mu() * cde,
        # P(S) at the wild type is ~5e-4; it is on the axis but at the floor.
        "P_S": float(np.exp(model.log_conditional(
            config.CRITIC_COL, sequence,
            model.alphabet.index(config.TARGET_RESIDUE), beta=1.0))),
        # No value: it carries D at the critic, so it is not in the S-carrying
        # subset the distance is measured to. Absent, never zero.
        "min_hamming_natural_S_pct": np.nan,
    }


def main() -> None:
    style.apply()
    config.ensure_dirs()

    frame = data.load_all(serine=True, only_saturated=True)
    table = data.summarise(frame)
    q_wt = selection.wild_type_Q()
    chosen = selection.select(table, q_wt)
    if chosen.empty:
        raise SystemExit(
            "no run is admissible on all four requirements -- see s02_means.csv "
            "and the thresholds in selection.py"
        )
    print(f"  selected {len(chosen)} of {len(table)} runs:")
    for row in chosen.itertuples():
        print(f"    {row.set}/beta{row.beta:g}/{row.run}: "
              f"<d>={row.hamming:.1f} Q={row.Q:.1f} P_S={row.P_S:.3f} "
              f"unique={row.unique_pct:.1f}%")

    keys = [(r.set, r.beta, r.run) for r in chosen.itertuples()]
    clouds = []
    for (set_name, beta, run) in keys:
        subset = frame[(frame["set"] == set_name) & np.isclose(frame["beta"], beta)
                       & (frame["run"] == run)]
        clouds.append(subset)

    have_natural = all(VARS[-1] in cloud and cloud[VARS[-1]].notna().any()
                       for cloud in clouds)
    variables = VARS if have_natural else VARS_4
    if not have_natural:
        print(f"  {VARS[-1]} absent -- drawing {len(variables)}x{len(variables)}; "
              "run s03_prepare.sh to add it over the common 165-243 window")

    wt = wild_type_values()
    n = len(variables)
    fig, axes = plt.subplots(n, n, figsize=(3.05 * n, 3.05 * n))

    # Shared limits per variable, so every panel of a row/column is on one scale.
    limits = {}
    for name in variables:
        values = np.concatenate([c[name].dropna().to_numpy(float) for c in clouds])
        if name in wt and np.isfinite(wt[name]):
            values = np.append(values, wt[name])
        pad = 0.06 * (values.max() - values.min() or 1.0)
        limits[name] = (values.min() - pad, values.max() + pad)

    per_set_index: dict[str, int] = {}
    styles = []
    for row in chosen.itertuples():
        index = per_set_index.get(row.set, 0)
        per_set_index[row.set] = index + 1
        styles.append((SET_COLOUR[row.set], MARKERS[(index + list(config.SETS).index(row.set) * 2) % len(MARKERS)]))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                for cloud, (colour, _) in zip(clouds, styles):
                    series = cloud[variables[i]].dropna()
                    if series.empty:
                        continue
                    ax.hist(series, bins=28, range=limits[variables[i]],
                            histtype="step", color=colour, lw=1.4, density=True)
                if variables[i] in wt and np.isfinite(wt[variables[i]]):
                    ax.axvline(wt[variables[i]], color=style.C_WT, ls=":", lw=1.2)
                ax.set_xlim(*limits[variables[i]])
                ax.set_yticks([])
            else:
                for cloud, (colour, marker) in zip(clouds, styles):
                    ax.scatter(cloud[variables[j]], cloud[variables[i]],
                               s=POINT_SIZE, alpha=POINT_ALPHA, marker=marker,
                               color=colour, linewidths=0, zorder=3)
                if (np.isfinite(wt.get(variables[j], np.nan))
                        and np.isfinite(wt.get(variables[i], np.nan))):
                    ax.scatter([wt[variables[j]]], [wt[variables[i]]], s=190,
                               marker="X", facecolor=style.SURFACE,
                               edgecolor=style.C_WT, linewidth=1.6, zorder=5)
                ax.set_xlim(*limits[variables[j]])
                ax.set_ylim(*limits[variables[i]])
            if i == n - 1:
                ax.set_xlabel(SHORT_LABELS.get(
                    variables[j], style.LABELS.get(variables[j], variables[j])))
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(SHORT_LABELS.get(
                    variables[i], style.LABELS.get(variables[i], variables[i])))
            elif j != 0:
                ax.set_yticklabels([])
            style.tidy(ax)

    handles = [
        plt.Line2D([], [], color=colour, marker=marker, lw=0, ms=8,
                   label=selection.label(row))
        for row, (colour, marker) in zip(chosen.itertuples(), styles)
    ]
    handles.append(plt.Line2D([], [], color=style.C_WT, marker="X", lw=0, ms=10,
                              markerfacecolor=style.SURFACE, label="wild type"))
    axes[0, n - 1].axis("off")
    axes[0, n - 1].legend(handles=handles, frameon=False, fontsize=10,
                          loc="upper right", bbox_to_anchor=(1.02, 1.0))

    fig.suptitle(f"S3 — the {len(chosen)} selected runs, {n}×{n} corner", fontsize=14)
    style.caveat(
        fig,
        "1000 chains per run, all past the S1 gate, read through the "
        r"$\beta=1$ serine rescoring.  points are larger than in fig02 "
        "(s=11 vs 3.4): six ensembles, not one $\\lambda$ ramp." "\n"
        + ("the natural-neighbour axis is scored over the COMMON 165–243 window "
           "for every run, which is what makes it comparable across sets of 14, "
           "20 and 79 columns."
           if have_natural else
           "the natural-neighbour axis is absent — run s03_prepare.sh."),
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    style.save(fig, f"s03_corner_{n}x{n}", out_dir=config.OUT_DIR)

    out = config.RESULTS_ROOT / "s03_selected.csv"
    chosen.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
