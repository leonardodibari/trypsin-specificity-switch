"""F2 -- the grafting-segment ensembles as a 5x5 corner plot.

One condition ("grafting, beta=1"), every lambda on the same axes, coloured by
lambda along a single-hue light->dark ramp built from style.BLUE. The lower
triangle carries the pairwise scatter, the diagonal the per-lambda marginal, and
the upper triangle is left blank -- the legend lives there instead, so it can
never sit on top of data.

The wild type is marked wherever both axes are defined for it. It has no
distance to the nearest natural S-carrying sequence (it carries D at i*, and the
observable is a property of a *designed* chain relative to the natural S set),
so the WT marker is omitted from that row and column rather than faked.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import data
import style

CONDITION = "grafting, beta=1"
VARS = style.CORNER_VARS_FULL


# --- colour: a single-hue ramp derived from style.BLUE --------------------------
def _mix(colour: str, towards: str, amount: float) -> tuple[float, float, float]:
    """Blend ``colour`` towards ``towards`` by ``amount``. No new hue is invented."""
    a = np.asarray(to_rgb(colour))
    b = np.asarray(to_rgb(towards))
    return tuple((1.0 - amount) * a + amount * b)


def lambda_ramp() -> LinearSegmentedColormap:
    """Light -> dark, one hue, anchored on style.BLUE between the two surfaces."""
    return LinearSegmentedColormap.from_list(
        "blue_ramp",
        [_mix(style.BLUE, style.SURFACE, 0.70),
         _mix(style.BLUE, style.SURFACE, 0.32),
         style.BLUE,
         _mix(style.BLUE, style.INK, 0.62)],
    )


def lambda_colours(n: int) -> list:
    cmap = lambda_ramp()
    return [cmap(t) for t in np.linspace(0.14, 1.0, n)]


# --- data ------------------------------------------------------------------------
def load() -> list[tuple[float, "object"]]:
    runs = data.discover_runs(CONDITION)
    if not runs:
        raise SystemExit(f"No ensembles for {CONDITION!r} -- run STAGE 1 first.")
    return [(run.lam, data.run_frame(run)) for run in runs]


def wild_type_values() -> dict:
    """WT coordinates on every variable.

    data.reference_points() computes the full observable set for the wild type
    and the literature series, including the distance to the nearest natural
    S-carrier -- so unlike earlier versions of this figure the wild type can be
    marked in that row and column too.
    """
    row = data.reference_points().iloc[0]
    return {var: float(row[var]) for var in VARS}


def series_values():
    """The reconstructed literature ladder, on the same axes (rows 1..3)."""
    return data.reference_points().iloc[1:].reset_index(drop=True)


def limits(frames, wt, series) -> dict:
    """Common axis limits per variable, padded, covering the reference points."""
    out = {}
    for var in VARS:
        pooled = np.concatenate([f[var].to_numpy(dtype=float) for _, f in frames])
        lo, hi = float(pooled.min()), float(pooled.max())
        if var in wt:
            lo, hi = min(lo, wt[var]), max(hi, wt[var])
        column = series[var].to_numpy(dtype=float)
        lo, hi = min(lo, column.min()), max(hi, column.max())
        pad = 0.055 * (hi - lo or 1.0)
        out[var] = (lo - pad, hi + pad)
    return out


# --- figure -----------------------------------------------------------------------
def main() -> None:
    style.apply()
    frames = load()
    wt = wild_type_values()
    series = series_values()
    lims = limits(frames, wt, series)
    colours = lambda_colours(len(frames))
    n = len(VARS)

    fig = plt.figure(figsize=(15.2, 14.6))
    gs = fig.add_gridspec(n, n, left=0.085, right=0.985, bottom=0.098, top=0.915,
                          hspace=0.10, wspace=0.10)

    axes = {}
    x_anchor, y_anchor = {}, {}
    for i in range(n):
        for j in range(i + 1):
            kwargs = {}
            if j in x_anchor:
                kwargs["sharex"] = x_anchor[j]
            if i != j and i in y_anchor:
                kwargs["sharey"] = y_anchor[i]
            ax = fig.add_subplot(gs[i, j], **kwargs)
            axes[(i, j)] = ax
            x_anchor.setdefault(j, ax)
            if i != j:
                y_anchor.setdefault(i, ax)

    # --- diagonal: per-lambda marginals ------------------------------------------
    for i, var in enumerate(VARS):
        ax = axes[(i, i)]
        pooled = np.concatenate([f[var].to_numpy(dtype=float) for _, f in frames])
        edges = np.histogram_bin_edges(pooled, bins=30)
        for (lam, frame), colour in zip(frames, colours):
            ax.hist(frame[var].to_numpy(dtype=float), bins=edges, density=True,
                    histtype="step", lw=1.9, color=colour, zorder=3)
        if var in wt:
            ax.axvline(wt[var], color=style.C_WT, lw=1.6, ls=(0, (4, 2)), zorder=5)
        style.tidy(ax, grid="x")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(*lims[var])

    # --- lower triangle: pairwise scatter ----------------------------------------
    for i in range(n):
        for j in range(i):
            ax = axes[(i, j)]
            for (lam, frame), colour in zip(frames, colours):
                ax.scatter(frame[VARS[j]], frame[VARS[i]], s=3.4, alpha=0.22,
                           color=colour, linewidths=0, rasterized=True, zorder=2)
            xs = series[VARS[j]].to_numpy(dtype=float)
            ys = series[VARS[i]].to_numpy(dtype=float)
            ax.plot(np.concatenate([[wt[VARS[j]]], xs]),
                    np.concatenate([[wt[VARS[i]]], ys]),
                    color=style.C_1992, ls=":", lw=1.6, zorder=7)
            for k, (x, y) in enumerate(zip(xs, ys)):
                ax.scatter([x], [y], s=115, facecolor=style.SURFACE,
                           edgecolor=style.INK, linewidths=1.5, zorder=8)
                ax.annotate(str(k + 2), xy=(x, y), ha="center", va="center",
                            fontsize=9, color=style.INK, zorder=9)
            ax.scatter([wt[VARS[j]]], [wt[VARS[i]]], s=190, marker="X",
                       color=style.C_WT, edgecolors=style.SURFACE,
                       linewidths=1.5, zorder=10)
            style.tidy(ax)
            ax.set_xlim(*lims[VARS[j]])
            ax.set_ylim(*lims[VARS[i]])

    # --- tick and label housekeeping ---------------------------------------------
    for (i, j), ax in axes.items():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        if i != j:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        if i == n - 1:
            ax.set_xlabel(style.LABELS[VARS[j]], labelpad=7)
        else:
            ax.tick_params(labelbottom=False)
        if j == 0 and i > 0:
            ax.set_ylabel(style.LABELS[VARS[i]], labelpad=7)
        else:
            ax.tick_params(labelleft=False)

    # one explicit "this is the wild type" pointer, in a panel with room for it
    pointer = axes[(1, 0)]
    pointer.annotate("wild type", xy=(wt["hamming_to_wt"], wt["P_S"]),
                     xytext=(16, 14), textcoords="offset points", fontsize=12.5,
                     color=style.C_WT, ha="left", va="bottom", zorder=10)

    # --- legend, parked in the blank upper triangle -------------------------------
    legend_ax = fig.add_subplot(gs[0:2, 2:5])
    legend_ax.axis("off")
    handles = [Line2D([], [], marker="o", ls="none", ms=10, color=colour,
                      label=f"$\\lambda$ = {lam:g}" + ("  (unbiased)" if lam == 0 else ""))
               for (lam, _), colour in zip(frames, colours)]
    handles.append(Line2D([], [], marker="X", ls="none", ms=13, color=style.C_WT,
                          markeredgecolor=style.SURFACE, label="wild-type trypsin (1)"))
    handles.append(Line2D([], [], marker="o", ls=":", ms=10, color=style.C_1992,
                          markerfacecolor=style.SURFACE, markeredgecolor=style.INK,
                          label="literature series (2–4)"))
    legend_ax.legend(handles=handles, loc="center", frameon=False,
                     title="bias strength $\\lambda$", ncol=2, columnspacing=1.4,
                     handletextpad=0.5, labelspacing=0.85, borderpad=0.0)
    legend_ax.get_legend().get_title().set_fontsize(16)

    n_chains = int(np.median([len(f) for _, f in frames]))
    fig.suptitle("Grafting-segment designs at $\\beta$ = 1: the bias $\\lambda$ moves "
                 "every observable together", y=0.965)
    style.caveat(fig,
                 f"{len(frames)} $\\lambda$ values, {n_chains} chains each, drawn on common axes; "
                 "the upper triangle is intentionally blank.\n"
                 "2 = + loops 185–188 and 221–225 (the 1992 construct), 3 = + 172, "
                 "4 = + 216 and 226; reconstructed from column positions, not the "
                 "published sequences.",
                 y=0.008)
    style.save(fig, "fig02_corner_grafting")


if __name__ == "__main__":
    main()
