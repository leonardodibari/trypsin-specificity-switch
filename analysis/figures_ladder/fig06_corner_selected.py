"""F6 -- the selected runs only, as a 4x4 corner plot.

Four observables (the natural-neighbour distance is dropped: it is not comparable
across site sets of different size), and only the runs that selection.select_all()
picks. Colour and marker say which condition a run belongs to; filled and hollow
separate the two runs chosen within a condition. The runs are chosen by hand from
the full pool (SI S2), not by a scoring rule, so nothing here is "the winner" --
the figure is for comparing what each setting costs on the other axes.

The dashed line on the Q axes is the wild type: "at least as plausible as the
wild type" is the one absolute landmark on that scale.

The fourth condition was never sampled. It appears once, in the legend, as a
placeholder; nothing is drawn for it.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import config
import data
import selection
import style

VARS = style.CORNER_VARS_SELECTED


# --- pairing the selected table rows back to their ensembles ---------------------
def selected_runs(table) -> list[dict]:
    """One record per selected row: the table row plus the per-chain frame."""
    records = []
    for _, row in table.iterrows():
        if bool(row.get("placeholder", False)):
            records.append({"row": row, "frame": None})
            continue
        matches = [run for run in data.discover_runs(row["condition"])
                   if np.isclose(run.beta, row["beta"]) and np.isclose(run.lam, row["lam"])]
        if not matches:
            raise SystemExit(f"Selected run not found on disk: {row['condition']} "
                             f"beta={row['beta']} lam={row['lam']}")
        records.append({"row": row, "frame": data.run_frame(matches[0])})
    return records


def run_label(row) -> str:
    """Condition plus the parameters the condition name does not already give."""
    params = f"$\\lambda$={row['lam']:g}"
    if "beta=1" not in row["condition"]:
        params = f"$\\beta$={row['beta']:g}, " + params
    return f"{row['condition']},  {params}"


def limits(records, reference) -> dict:
    """Axis limits covering the design clouds AND the reference points."""
    out = {}
    for var in VARS:
        pooled = np.concatenate([r["frame"][var].to_numpy(dtype=float)
                                 for r in records if r["frame"] is not None])
        lo, hi = float(pooled.min()), float(pooled.max())
        column = reference[var].to_numpy(dtype=float)
        lo, hi = min(lo, column.min()), max(hi, column.max())
        pad = 0.07 * (hi - lo or 1.0)
        out[var] = (lo - pad, hi + pad)
    return out


def draw_reference(ax, reference, var_x: str, var_y: str) -> None:
    """Wild type and the reconstructed literature ladder, on one panel.

    Row 0 of ``reference`` is the wild type (marked X, numbered 1); rows 1..3 are
    the published ladder, numbered 2..4 and joined by a dotted line.
    """
    xs = reference[var_x].to_numpy(dtype=float)
    ys = reference[var_y].to_numpy(dtype=float)
    ax.plot(xs, ys, color=style.C_1992, ls=":", lw=1.6, zorder=11)
    for k in range(1, len(xs)):
        ax.scatter([xs[k]], [ys[k]], s=115, facecolor=style.SURFACE,
                   edgecolor=style.INK, linewidths=1.5, zorder=12)
        ax.annotate(str(k + 1), xy=(xs[k], ys[k]), ha="center", va="center",
                    fontsize=9, color=style.INK, zorder=13)
    ax.scatter([xs[0]], [ys[0]], s=190, marker="X", color=style.C_WT,
               edgecolors=style.SURFACE, linewidths=1.5, zorder=14)


# --- drawing ----------------------------------------------------------------------
def cloud_kwargs(row):
    """Colour/marker from style; fill from the rank within the condition."""
    group = style.group(row["condition"])
    filled = style.RANK_FILL[int(row["rank_in_condition"])] == "full"
    if filled:
        return dict(c=[group["color"]], marker=group["marker"], s=5.0,
                    alpha=0.22, linewidths=0.0)
    return dict(facecolors="none", edgecolors=group["color"], marker=group["marker"],
                s=8.0, alpha=0.26, linewidths=0.4)


def centroid_kwargs(row):
    group = style.group(row["condition"])
    filled = style.RANK_FILL[int(row["rank_in_condition"])] == "full"
    return dict(marker=group["marker"], s=230,
                facecolors=group["color"] if filled else style.SURFACE,
                edgecolors=style.INK if filled else group["color"],
                linewidths=1.3 if filled else 2.4)


def gate_lines(ax, var, axis: str) -> None:
    """Wild-type reference line wherever Q is on an axis.

    This used to draw two gate thresholds. The gate is gone -- runs are now
    picked by hand from the full pool -- but Q still deserves a reference, since
    "more plausible than the wild type" is the only absolute landmark on it.
    """
    if var != "Q":
        return
    level = data.wild_type_reference()["Q"]
    line = ax.axvline if axis == "x" else ax.axhline
    line(level, color=style.MUTED, lw=1.2, ls=(0, (5, 3)), zorder=1)


def main() -> None:
    style.apply()
    table = selection.select_all()
    records = selected_runs(table)
    drawn = [r for r in records if r["frame"] is not None]
    reference = data.reference_points()
    lims = limits(drawn, reference)
    n = len(VARS)

    fig = plt.figure(figsize=(13.4, 12.9))
    gs = fig.add_gridspec(n, n, left=0.095, right=0.985, bottom=0.115, top=0.912,
                          hspace=0.11, wspace=0.11)

    axes, x_anchor, y_anchor = {}, {}, {}
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

    # diagonal: the marginal of each selected run
    for i, var in enumerate(VARS):
        ax = axes[(i, i)]
        pooled = np.concatenate([r["frame"][var].to_numpy(dtype=float) for r in drawn])
        edges = np.histogram_bin_edges(pooled, bins=34)
        for record in drawn:
            row = record["row"]
            group = style.group(row["condition"])
            solid = style.RANK_FILL[int(row["rank_in_condition"])] == "full"
            ax.hist(record["frame"][var].to_numpy(dtype=float), bins=edges,
                    density=True, histtype="step", lw=2.0, color=group["color"],
                    ls="-" if solid else (0, (3, 1.6)), zorder=3)
        gate_lines(ax, var, "x")
        ax.axvline(float(reference[var].iloc[0]), color=style.C_WT, lw=1.6,
                   ls=(0, (4, 2)), zorder=5)
        style.tidy(ax, grid="x")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(*lims[var])

    # lower triangle: the clouds, then the run means on top
    for i in range(n):
        for j in range(i):
            ax = axes[(i, j)]
            for record in drawn:
                frame = record["frame"]
                ax.scatter(frame[VARS[j]], frame[VARS[i]], rasterized=True,
                           zorder=2, **cloud_kwargs(record["row"]))
            gate_lines(ax, VARS[j], "x")
            gate_lines(ax, VARS[i], "y")
            draw_reference(ax, reference, VARS[j], VARS[i])
            for record in drawn:
                row, frame = record["row"], record["frame"]
                mx = float(frame[VARS[j]].mean())
                my = float(frame[VARS[i]].mean())
                ax.scatter([mx], [my], zorder=8, **centroid_kwargs(row))
            style.tidy(ax)
            ax.set_xlim(*lims[VARS[j]])
            ax.set_ylim(*lims[VARS[i]])

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

    # --- legend, in the blank upper triangle --------------------------------------
    legend_ax = fig.add_subplot(gs[0, 1:4])
    legend_ax.axis("off")
    handles = []
    for record in records:
        row = record["row"]
        group = style.group(row["condition"])
        if record["frame"] is None:
            handles.append(Line2D([], [], marker=group["marker"], ls="none", ms=11,
                                  markerfacecolor=style.SURFACE,
                                  markeredgecolor=style.C_OTHER, markeredgewidth=1.8,
                                  label=f"{row['condition']}:\nnot sampled — placeholder"))
            continue
        filled = style.RANK_FILL[int(row["rank_in_condition"])] == "full"
        handles.append(Line2D(
            [], [], marker=group["marker"], ls="none", ms=12,
            markerfacecolor=group["color"] if filled else style.SURFACE,
            markeredgecolor=style.INK if filled else group["color"],
            markeredgewidth=1.2 if filled else 2.2, label=run_label(row)))
    seen = set()
    handles = [h for h in handles
               if not (h.get_label() in seen or seen.add(h.get_label()))]
    handles += [
        Line2D([], [], marker="X", ls="none", ms=13, color=style.C_WT,
               markeredgecolor=style.SURFACE, label="wild-type trypsin (1)"),
        Line2D([], [], marker="o", ls=":", ms=10, color=style.C_1992,
               markerfacecolor=style.SURFACE, markeredgecolor=style.INK,
               label="literature series (2–4)"),
        Line2D([], [], marker="o", ls="none", ms=12, markerfacecolor=style.C_OTHER,
               markeredgecolor=style.INK, markeredgewidth=1.2,
               label="filled: first run listed"),
        Line2D([], [], marker="o", ls="none", ms=12, markerfacecolor=style.SURFACE,
               markeredgecolor=style.C_OTHER, markeredgewidth=2.2,
               label="hollow: second (dashed on the diagonal)"),
        Line2D([], [], ls=(0, (5, 3)), lw=1.2, color=style.MUTED,
               label=r"wild-type $Q_\mu$"),
    ]
    legend = legend_ax.legend(handles=handles, loc="upper left", frameon=False,
                              title="selected runs   (large marker: ensemble mean)",
                              ncol=2, handletextpad=0.7, labelspacing=0.58,
                              columnspacing=1.5, borderpad=0.0, fontsize=12.5,
                              bbox_to_anchor=(0.0, 1.06))
    legend.get_title().set_fontsize(14)
    legend._legend_box.align = "left"

    missing = [name for name in config.CONDITIONS
               if config.CONDITIONS[name] is None]
    caption = ("runs chosen by hand from the full pool (SI S2), not by a scoring rule\n"
               "1 = wild type;   2 = + loops 185–188 and 221–225 (the 1992 construct);   "
               "3 = + 172;   4 = + 216 and 226\n"
               "the literature series is reconstructed from column positions, "
               "not the published sequences")
    if missing:
        caption += ("\n" + ", ".join(missing)
                    + ": never sampled, shown as a placeholder — no data are drawn for it")
    style.caveat(fig, caption, y=0.012)

    fig.suptitle("The selected designs, on four "
                 "common observables", y=0.962)
    style.save(fig, "fig06_corner_selected")


if __name__ == "__main__":
    main()
