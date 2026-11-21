"""S5 -- designs against the natural family, on the design observables.

This is the comparison F2 cannot make. Natural sequences sit 20-214 mutations
from rat trypsin while the designs sit at 12-51, so on F2's axes only 0.1% of
naturals are visible and NO natural chymotrypsin is -- putting them there would
either hide the naturals or crush the lambda ladder. Here the axes are opened up
to hold both, at the cost of the designs becoming a small cluster. That cluster
being small IS the result: the designs are a tight neighbourhood of the wild
type, not a walk across the family.

Axis limits use the 1st-99th percentile of the naturals (a handful of natural
sequences are enormously far from rat trypsin and would otherwise set the scale
for everything) unioned with the full range of the designs and references.

Note the asymmetry on the ``hamming_to_wt`` axis: it is a DESIGN coordinate --
distance from this particular wild type -- so the naturals' position on it says
only how far across the family they are, not anything about their quality. The
honest family comparison is the PCA in F7.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import config
import data
import selection
import style

#: Resolved against what the naturals actually carry: natural_observables.csv
#: holds the natural-neighbour distance only when calibrate_mu.py was run with
#: --wt, and a variable the naturals lack cannot be a shared axis.
_WANTED = style.CORNER_VARS_FULL
NATURAL_CLASSES = [
    ("other", style.C_OTHER, "other naturals"),
    ("trypsin", style.C_TRYPSIN, "natural trypsins (D at $i^*$)"),
    ("chymotrypsin", style.C_CHYMO, "natural chymotrypsins (S at $i^*$)"),
]
MAX_DESIGN_POINTS = 600

#: In every other figure colour means CONDITION. Here colour already means
#: NATURAL CLASS -- and style.SCATTER_GROUPS would paint the grafting designs the
#: same blue as the natural trypsins and the cooling designs the same orange as
#: the natural chymotrypsins, which makes the panels unreadable. So in this
#: figure, and only here, every design is one colour and the condition is carried
#: by marker shape alone.
DESIGN_COLOUR = style.VIOLET


def limits(natural, designs, reference) -> dict:
    """Axes covering the natural bulk, all designs and the reference points."""
    out = {}
    for var in VARS:
        lo, hi = np.percentile(natural[var].to_numpy(dtype=float), [1, 99])
        for frame in designs:
            column = frame[var].to_numpy(dtype=float)
            lo, hi = min(lo, column.min()), max(hi, column.max())
        column = reference[var].to_numpy(dtype=float)
        lo, hi = min(lo, column.min()), max(hi, column.max())
        pad = 0.05 * (hi - lo or 1.0)
        out[var] = (lo - pad, hi + pad)
    return out


def main() -> None:
    style.apply()
    natural = data.natural_observables()
    global VARS
    VARS = [v for v in _WANTED if v in natural.columns]
    dropped = [v for v in _WANTED if v not in natural.columns]
    if dropped:
        print("  note: not available for the naturals, dropped from the panel: "
              + ", ".join(dropped))
    reference = data.reference_points()

    table = selection.select_all()
    designs = []
    rng = np.random.default_rng(0)
    for _, row in table.iterrows():
        if bool(row["placeholder"]):
            continue
        run = next(r for r in data.discover_runs(row["condition"])
                   if r.beta == row["beta"] and r.lam == row["lam"])
        frame = data.run_frame(run)
        if len(frame) > MAX_DESIGN_POINTS:
            frame = frame.iloc[rng.choice(len(frame), MAX_DESIGN_POINTS,
                                          replace=False)]
        designs.append((row, frame))

    lims = limits(natural, [f for _, f in designs], reference)
    n = len(VARS)

    fig = plt.figure(figsize=(16.5, 16.2))
    gs = fig.add_gridspec(n, n, left=0.075, right=0.985, top=0.905, bottom=0.125,
                          hspace=0.10, wspace=0.10)
    axes = {}
    for i in range(n):
        for j in range(i + 1):
            axes[(i, j)] = fig.add_subplot(gs[i, j])

    # diagonal: marginals, each family and each design run normalised to itself
    for i, var in enumerate(VARS):
        ax = axes[(i, i)]
        edges = np.linspace(*lims[var], 56)
        for key, colour, _ in NATURAL_CLASSES:
            values = natural.loc[natural["natural_class"] == key, var].to_numpy(dtype=float)
            ax.hist(values, bins=edges, density=True, color=colour, alpha=0.55,
                    zorder=2)
        for row, frame in designs:
            ax.hist(frame[var].to_numpy(dtype=float), bins=edges, density=True,
                    histtype="step", lw=1.7, color=DESIGN_COLOUR, zorder=3)
        ax.axvline(float(reference[var].iloc[0]), color=style.C_WT, lw=1.6,
                   ls=(0, (4, 2)), zorder=5)
        style.tidy(ax, grid="x")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(*lims[var])

    # lower triangle: naturals underneath, designs on top, references on top of that
    for i in range(n):
        for j in range(i):
            ax = axes[(i, j)]
            for key, colour, _ in NATURAL_CLASSES:
                subset = natural[natural["natural_class"] == key]
                ax.scatter(subset[VARS[j]], subset[VARS[i]], s=2.6, alpha=0.20,
                           color=colour, linewidths=0, rasterized=True, zorder=2)
            for row, frame in designs:
                spec = style.group(row["condition"])
                filled = int(row["rank_in_condition"]) == 0
                ax.scatter(frame[VARS[j]], frame[VARS[i]], s=10,
                           marker=spec["marker"], alpha=0.65,
                           facecolor=DESIGN_COLOUR if filled else "none",
                           edgecolor=DESIGN_COLOUR, linewidths=0.7,
                           rasterized=True, zorder=4)
            xs = reference[VARS[j]].to_numpy(dtype=float)
            ys = reference[VARS[i]].to_numpy(dtype=float)
            ax.plot(xs, ys, color=style.C_1992, ls=":", lw=1.6, zorder=8)
            for k in range(1, len(xs)):
                ax.scatter([xs[k]], [ys[k]], s=95, facecolor=style.SURFACE,
                           edgecolor=style.INK, linewidths=1.4, zorder=9)
                ax.annotate(str(k + 1), xy=(xs[k], ys[k]), ha="center",
                            va="center", fontsize=8.5, color=style.INK, zorder=10)
            ax.scatter([xs[0]], [ys[0]], s=170, marker="X", color=style.C_WT,
                       edgecolors=style.SURFACE, linewidths=1.4, zorder=11)
            style.tidy(ax)
            ax.set_xlim(*lims[VARS[j]])
            ax.set_ylim(*lims[VARS[i]])

    for (i, j), ax in axes.items():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        if i != n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(style.LABELS[VARS[j]], labelpad=7)
        if j != 0 or i == 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(style.LABELS[VARS[i]], labelpad=7)

    legend_ax = fig.add_subplot(gs[0:2, 2:5])
    legend_ax.axis("off")
    handles = [Line2D([], [], marker="o", ls="none", ms=10, color=colour, label=label)
               for _, colour, label in NATURAL_CLASSES]
    seen = set()
    for row, _ in designs:
        condition = row["condition"]
        if condition in seen:
            continue
        seen.add(condition)
        spec = style.group(condition)
        handles.append(Line2D([], [], marker=spec["marker"], ls="none", ms=9,
                              color=DESIGN_COLOUR, label=f"designs: {condition}"))
    handles += [
        Line2D([], [], marker="X", ls="none", ms=13, color=style.C_WT,
               markeredgecolor=style.SURFACE, label="wild-type trypsin (1)"),
        Line2D([], [], marker="o", ls=":", ms=10, color=style.C_1992,
               markerfacecolor=style.SURFACE, markeredgecolor=style.INK,
               label="literature series (2–4)"),
    ]
    legend_ax.legend(handles=handles, loc="center", frameon=False, ncol=2,
                     columnspacing=1.3, handletextpad=0.6, labelspacing=0.8,
                     fontsize=13, borderpad=0.0)

    fig.suptitle("The designs are a tight neighbourhood of the wild type, "
                 "not a walk across the family", y=0.962)
    style.caveat(fig,
                 "Natural sample: "
                 f"{len(natural):,} sequences from the calibration set. Axes span the "
                 "1st–99th percentile of the naturals, so a few very distant natural "
                 "sequences are off-panel.\n"
                 "'Hamming to wild type' is a DESIGN coordinate — distance from this "
                 "particular trypsin — so a natural sequence's position on it measures "
                 "family distance, not quality. F7 is the proper family comparison.\n"
                 "NOTE: here colour means natural class, so every design is drawn in one "
                 "colour and its condition is carried by marker shape — unlike F5–F7, "
                 "where colour means condition.",
                 y=0.012)
    style.save(fig, "si05_naturals_vs_designs")

    print(f"naturals: {len(natural):,}  "
          + "  ".join(f"{k}={int((natural['natural_class'] == k).sum()):,}"
                      for k, _, _ in NATURAL_CLASSES))
    for var in VARS:
        column = natural[var].to_numpy(dtype=float)
        pooled = np.concatenate([f[var].to_numpy(dtype=float) for _, f in designs])
        print(f"  {var:<28} naturals p1-p99 [{np.percentile(column, 1):8.1f},"
              f"{np.percentile(column, 99):8.1f}]   designs [{pooled.min():8.1f},"
              f"{pooled.max():8.1f}]")


if __name__ == "__main__":
    main()
