"""SI 3 -- gap behaviour across the combination sets.

The per-run gap_abundance figure answers "which columns are gaps"; this one
answers the question that matters for the table: *do the designed chains keep
the wild type's gap pattern, and does that depend on the condition?*

For every sampled run under every condition in config.CONDITIONS, the per-chain
gap count (data.run_frame -> n_gaps) is drawn against the wild type's own gap
count. "consensus-19 + cooling" was never sampled and is shown as an explicit
placeholder rather than quietly omitted.

Gap counts are small integers (32..44 over the whole scan), so a kernel-density
violin would invent structure between the integers and would degenerate to a
spike wherever an ensemble is unanimous. Each run is therefore drawn as a
*discrete* violin: one bar per attainable gap count, width proportional to the
fraction of chains carrying it. A run where every chain has 42 gaps is a single
full-width bar, which is exactly what it should look like.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

import config
import data
import style

STEM = "si03_gaps"

#: Which condition goes in which cell, resolved from config rather than named
#: literally: the consensus sets carry their DERIVED size in their name, so a
#: hard-coded "consensus-19, beta=1" silently stops matching when the threshold
#: or the model changes -- and the figure still builds, just without that panel.
def _by_role(*roles):
    out = []
    for role in roles:
        out += [n for n, r in config.CONDITION_ROLE.items() if r == role]
    return out


_WARM = _by_role("grafting", "consensus_tight", "consensus_wide")
SINGLE_BETA = _WARM[:2]
COOLED = next(iter(_by_role("grafting_cool")), None)
#: Only a condition with no runs is a placeholder.
PLACEHOLDER = next((n for n in _by_role("consensus_wide_cool", "consensus_tight")
                    if config.CONDITIONS.get(n) is None), None)

Y_LIM = (31.0, 46.4)
HALF_WIDTH = 0.42


def ramp(base_hex: str, n: int) -> list:
    """A sequential light -> dark ramp built from one palette hue.

    Derived from style, never invented: a pale tint of the hue, through the hue,
    to the hue shaded towards style.INK. Both lambda and beta are ordered knobs,
    so both get a ramp rather than categorical colours.
    """
    base = np.array(mcolors.to_rgb(base_hex))
    ink = np.array(mcolors.to_rgb(style.INK))
    light = 0.56 * np.ones(3) + 0.44 * base
    dark = 0.26 * base + 0.74 * ink
    cmap = mcolors.LinearSegmentedColormap.from_list("ramp", [light, base, dark])
    if n == 1:
        return [cmap(0.55)]
    return [cmap(t) for t in np.linspace(0.0, 1.0, n)]


def wt_gaps() -> int:
    return int((data.wild_type() == data.alphabet().gap_index).sum())


def discrete_violin(ax, x0: float, values: np.ndarray, colour) -> None:
    """One bar per attainable gap count, width proportional to its frequency."""
    levels, counts = np.unique(values, return_counts=True)
    frac = counts / counts.max()
    ax.barh(levels, width=2 * HALF_WIDTH * frac, left=x0 - HALF_WIDTH * frac,
            height=0.85, color=colour, edgecolor="none", zorder=3)
    ax.plot([x0], [values.mean()], marker="o", ms=4.6, color=style.INK,
            markeredgecolor="white", markeredgewidth=0.9, zorder=5, lw=0)


def frame_for(run) -> np.ndarray:
    frame = data.run_frame(run)
    if "n_gaps" not in frame:
        raise SystemExit(f"{run.path} has no n_gaps column")
    return frame["n_gaps"].to_numpy()


def setup_axis(ax, *, ylabel: bool, wt: int) -> None:
    ax.set_ylim(*Y_LIM)
    ax.axhline(wt, color=style.C_WT, lw=1.5, ls=(0, (5, 3)), zorder=4)
    style.tidy(ax, grid="y")
    if ylabel:
        ax.set_ylabel(f"{style.LABELS['n_gaps']} per chain")
    else:
        ax.set_yticklabels([])


def pretty(condition: str) -> str:
    """The config key as it should be *set*: beta=1 -> a real beta."""
    return condition.replace("beta=1", r"$\beta=1$")


def draw_single_beta(ax, condition: str, colour_base: str, wt: int,
                     *, letter: str, ylabel: bool, dx: float) -> None:
    runs = data.discover_runs(condition)
    colours = ramp(colour_base, len(runs))
    for i, (run, colour) in enumerate(zip(runs, colours)):
        discrete_violin(ax, i, frame_for(run), colour)
    ax.set_xlim(-0.75, len(runs) - 0.25)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{run.lam:g}" for run in runs])
    ax.set_xlabel(r"bias  $\lambda$")
    setup_axis(ax, ylabel=ylabel, wt=wt)
    ax.set_title(pretty(condition), pad=12, fontsize=15, color=style.MUTED)
    style.panel_letter(ax, letter, dx=dx)


def draw_cooled(ax, wt: int, *, letter: str) -> None:
    runs = data.discover_runs(COOLED)
    betas = sorted({run.beta for run in runs})
    colours = dict(zip(betas, ramp(style.ORANGE, len(betas))))

    positions, labels, groups = [], [], []
    cursor = 0.0
    for beta in betas:
        block = sorted([r for r in runs if r.beta == beta], key=lambda r: r.lam)
        start = cursor
        for run in block:
            discrete_violin(ax, cursor, frame_for(run), colours[beta])
            positions.append(cursor)
            labels.append(f"{run.lam:g}")
            cursor += 1.0
        groups.append((beta, start, cursor - 1.0))
        cursor += 0.85

    ax.set_xlim(-0.8, cursor - 1.05)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel(r"bias  $\lambda$,  grouped by inverse sampling temperature  $\beta$")
    setup_axis(ax, ylabel=True, wt=wt)

    for beta, start, stop in groups:
        ax.text(0.5 * (start + stop), Y_LIM[1] - 0.45, rf"$\beta={beta:g}$",
                ha="center", va="top", fontsize=13.5, color=style.MUTED)
        ax.plot([start - 0.42, stop + 0.42], [Y_LIM[1] - 1.45] * 2,
                color=style.RULE, lw=1.2, solid_capstyle="butt", zorder=2)
    ax.set_title(rf"{COOLED}   ($\beta>1$)", pad=12, fontsize=15,
                 color=style.MUTED)
    style.panel_letter(ax, letter, dx=-0.045)


def draw_placeholder(ax, condition: str, *, letter: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    box = FancyBboxPatch((0.07, 0.17), 0.86, 0.60,
                         boxstyle="round,pad=0.015,rounding_size=0.035",
                         linewidth=1.4, edgecolor=style.RULE,
                         facecolor="none", linestyle=(0, (5, 4)),
                         transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    ax.text(0.5, 0.60, condition, ha="center", va="center", fontsize=15.5,
            color=style.MUTED)
    ax.text(0.5, 0.44, "never sampled", ha="center", va="center",
            fontsize=18, color=style.INK)
    ax.text(0.5, 0.30, "placeholder — no gap statistics exist", ha="center",
            va="center", fontsize=13, color=style.MUTED)
    style.panel_letter(ax, letter, dx=-0.055, dy=1.02)


def main() -> None:
    style.apply()
    wt = wt_gaps()

    fig = plt.figure(figsize=(17.0, 10.8))
    # Panel a carries six lambda slots and panel b only three; equal cell widths
    # would make b's violins look twice as fat for no reason, so the widths are
    # set to keep the per-run slot width roughly constant across the row.
    grid = fig.add_gridspec(2, 3, left=0.062, right=0.985, top=0.845,
                            bottom=0.105, hspace=0.36, wspace=0.11,
                            height_ratios=[1.0, 1.0],
                            width_ratios=[1.30, 0.83, 0.87])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, :])

    draw_single_beta(ax_a, SINGLE_BETA[0], style.BLUE, wt,
                     letter="a", ylabel=True, dx=-0.128)
    draw_single_beta(ax_b, SINGLE_BETA[1], style.AQUA, wt,
                     letter="b", ylabel=False, dx=-0.07)
    if PLACEHOLDER is not None:
        draw_placeholder(ax_c, PLACEHOLDER, letter="c")
    elif len(_WARM) > 2:
        # Nothing is missing for this model, so the third cell shows the
        # remaining warm condition instead of an empty placeholder box.
        draw_single_beta(ax_c, _WARM[2], style.VIOLET, wt,
                         letter="c", ylabel=False, dx=-0.07)
    else:
        ax_c.axis("off")
    draw_cooled(ax_d, wt, letter="d")

    handles = [
        Line2D([], [], color=style.MUTED, lw=7, solid_capstyle="butt",
               label="bar width ∝ fraction of chains at that gap count"),
        Line2D([], [], color=style.INK, lw=0, marker="o", ms=6,
               markeredgecolor="white", label="ensemble mean"),
        Line2D([], [], color=style.C_WT, lw=1.5, ls=(0, (5, 3)),
               label=f"wild type ({wt} of 265 columns)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.915),
               ncol=3, handlelength=2.4, columnspacing=2.6, handletextpad=0.8)

    # Headline READ OFF THE DATA. It used to assert the earlier model's result
    # ("strong bias fills wild-type gaps"), which is false here: this model holds
    # the gap pattern at the wild type's 42 columns across every condition.
    warm = data.discover_runs(SINGLE_BETA[0])
    strongest = max(warm, key=lambda r: r.lam)
    drift = float(data.run_frame(strongest)["n_gaps"].mean()) - wt
    if drift <= -1.5:
        headline = ("Strong bias at $\\beta=1$ fills wild-type gaps; "
                    "cooling restores them")
    elif drift >= 1.5:
        headline = "Strong bias at $\\beta=1$ opens new gaps"
    else:
        headline = ("The designs keep the wild type's gap pattern, "
                    "at every bias and every temperature")
    fig.suptitle(headline, y=0.972)
    style.caveat(fig,
                 "1000 chains per run; gap counts are integers, so each run is drawn as a discrete violin "
                 "(one bar per attainable count) rather than a kernel density.\n"
                 + ("‘grafting + cooling’ means $\\beta>1$: its $\\beta=1$ rung is "
                    "byte-identical to ‘grafting, $\\beta=1$’ in panel a, so "
                    "config.CONDITION_FILTERS drops it rather than counting it twice."
                    if any(r.beta == 1 for r in data.discover_runs(COOLED))
                    else "‘grafting + cooling’ starts at $\\beta=1.5$; this model's "
                         "cooling ladder has no $\\beta=1$ rung, so nothing overlaps panel a."),
                 y=0.012)
    style.save(fig, STEM)


if __name__ == "__main__":
    main()
