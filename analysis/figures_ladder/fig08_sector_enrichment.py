"""F8 -- red-sector enrichment, grafting segment vs every site free.

The question: if the sampler is allowed to mutate *anywhere*, does it stop
concentrating its mutations on the red sector?

Enrichment is the ratio already in the CSV,

    ratio = (share of mutations that land on red) / (share of mutable sites that are red)

so ratio = 1 means "mutations fall on red exactly in proportion to its size" and
the number is dimensionless -- which is the only reason the two site sets can
share an axis at all. Raw mutation counts are deliberately NOT plotted: the free
run mutates ~5x more sites in total, so counts would say nothing about focus.

The two curves share a *scale* but not a *baseline composition*: red is 16 of 79
mutable columns (20.3%) when only the grafting segment moves, and 17 of 265
(6.4%) when every site is free, where an "other" sector exists as well. That is
stated on the figure, because a reader who misses it will over-read the gap
between the curves.

Sources: config.SECTOR_ENRICHMENT.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import style

STEM = "fig08_sector_enrichment"
SECTOR = "red"

#: One colour per source. Two groups, adjacent form -> the first two slots.
SOURCE_STYLE = {
    "grafting segment": {"color": style.BLUE, "marker": "o"},
    "all sites free": {"color": style.ORANGE, "marker": "s"},
}

#: Panel (b) breaks the unrestricted run down by sector. "grafting" here means
#: the 63 non-red columns of the grafting window, so the three sectors partition
#: the alignment and their enrichments are directly comparable.
SECTOR_STYLE = {
    "red": {"color": style.RED, "marker": "o", "label": "red sector"},
    "grafting": {"color": style.BLUE, "marker": "s",
                 "label": "grafting window (non-red)"},
    "other": {"color": style.C_OTHER, "marker": "^", "label": "everywhere else"},
}


def load_full() -> dict[str, pd.DataFrame]:
    """Every sector of every available source, unfiltered."""
    out = {}
    for name, path in config.SECTOR_ENRICHMENT.items():
        path = Path(path)
        if path.is_file():
            out[name] = pd.read_csv(path)
    return out


def load() -> dict[str, pd.DataFrame]:
    """Red-sector enrichment per source, skipping sources not sampled.

    The all-sites-free run is optional: it costs a full extra condition and
    exists only to answer "does freeing every column dilute the red sector?".
    When it is absent the figure draws the grafting curve alone and says on its
    face that the comparison is missing, rather than refusing to build.
    The grafting curve is required -- without it there is no figure at all.
    """
    frames: dict[str, pd.DataFrame] = {}
    for name, path in config.SECTOR_ENRICHMENT.items():
        path = Path(path)
        if not path.is_file():
            if name == "grafting segment":
                raise SystemExit(
                    f"{path} not found -- the grafting sector annotation is "
                    "required. Run:\n"
                    "  ANNOTATE=grafting bash analysis/run_ladder_sectors.sh annotate")
            print(f"  note: {name} not sampled; drawing without that curve")
            continue
        frame = pd.read_csv(path)
        red = frame[frame["sector"] == SECTOR].sort_values("lam").reset_index(drop=True)
        if red.empty:
            raise SystemExit(f"no rows with sector == {SECTOR!r} in {path}")
        frames[name] = red
    return frames


def main() -> None:
    style.apply()
    frames = load()

    full = load_full()
    breakdown = full.get("all sites free")
    two_panel = breakdown is not None
    if two_panel:
        fig, (ax, ax_b) = plt.subplots(1, 2, figsize=(18.0, 7.8))
        fig.subplots_adjust(left=0.072, right=0.985, top=0.800, bottom=0.165,
                            wspace=0.20)
    else:
        fig, ax = plt.subplots(figsize=(12.6, 7.8))
        fig.subplots_adjust(left=0.105, right=0.98, top=0.800, bottom=0.165)
        ax_b = None

    # "no enrichment" first, so it sits behind the data.
    ax.axhline(1.0, color=style.RULE, lw=1.6, ls=(0, (5, 3)), zorder=1)
    ax.text(12, 1.018, "no enrichment (ratio = 1):  mutations fall on red\n"
            "in proportion to its share of the mutable sites",
            ha="left", va="bottom", fontsize=12.5, color=style.MUTED,
            linespacing=1.35)

    for name, frame in frames.items():
        spec = SOURCE_STYLE[name]
        share = float(frame["share_of_mutable_sites"].iloc[0])
        n_cols = int(frame["mutable_columns"].iloc[0])
        label = (f"{name}\n"
                 f"red = {n_cols} mutable columns, {100 * share:.1f}% of those free")
        ax.plot(frame["lam"], frame["ratio"], lw=2.6, ms=9.5,
                marker=spec["marker"], color=spec["color"],
                markeredgecolor="white", markeredgewidth=1.0,
                label=label, zorder=3, clip_on=False)

    ax.set_xscale("symlog", linthresh=10, linscale=0.55)
    ax.set_xlim(-1.0, 1600)
    ticks = sorted({int(v) for f in frames.values() for v in f["lam"]})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    # From the data: the unbiased points sit near 0.3 for this model and were
    # being clipped off the bottom by a hard-coded 0.58.
    lo = min(float(f["ratio"].min()) for f in frames.values())
    hi = max(float(f["ratio"].max()) for f in frames.values())
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(r"bias on the critic site   $\lambda$")
    ax.set_ylabel("red-sector enrichment\n(share of mutations / share of mutable sites)")
    style.tidy(ax)

    unbiased = max(float(f.loc[f["lam"].idxmin(), "ratio"]) for f in frames.values())
    ax.annotate("unbiased ensemble:\nred is avoided", xy=(0.3, unbiased),
                xytext=(1.6, lo - pad + 0.03 * (hi - lo)), fontsize=12.5,
                color=style.MUTED,
                ha="left", va="bottom", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=style.RULE, lw=1.2))

    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.015),
              labelspacing=0.9, handlelength=2.2, handletextpad=0.8,
              borderaxespad=0.6)

    if ax_b is not None:
        # Where do the mutations go when NOTHING is restricted? Each sector's
        # enrichment is its share of the mutations over its share of the
        # mutable columns, so 1.0 means "in proportion to its size".
        ax_b.axhline(1.0, color=style.RULE, lw=1.6, ls=(0, (5, 3)), zorder=1)
        for sector, spec in SECTOR_STYLE.items():
            block = breakdown[breakdown["sector"] == sector].sort_values("lam")
            if block.empty:
                continue
            n_cols = int(block["mutable_columns"].iloc[0])
            share = 100 * float(block["share_of_mutable_sites"].iloc[0])
            ax_b.plot(block["lam"], block["ratio"], lw=2.6, ms=9.5,
                      marker=spec["marker"], color=spec["color"],
                      markeredgecolor="white", markeredgewidth=1.0, zorder=3,
                      clip_on=False,
                      label=f"{spec['label']}\n{n_cols} columns, {share:.1f}% of those free")
        ax_b.set_xscale("symlog", linthresh=10, linscale=0.55)
        ax_b.set_xlim(-1.0, 1600)
        ticks_b = sorted({int(v) for v in breakdown["lam"]})
        ax_b.set_xticks(ticks_b)
        ax_b.set_xticklabels([f"{t:g}" for t in ticks_b])
        ax_b.minorticks_off()
        ax_b.set_xlabel(r"bias on the critic site   $\lambda$")
        ax_b.set_ylabel("enrichment of each sector")
        ax_b.set_title("all sites free: where do the mutations go?",
                       fontsize=14, color=style.MUTED, pad=10)
        ax_b.legend(loc="lower right", labelspacing=0.8, handlelength=2.2,
                    handletextpad=0.8, borderaxespad=0.6, fontsize=12)
        style.tidy(ax_b)
        style.panel_letter(ax_b, "b", dx=-0.085, dy=1.10)
        style.panel_letter(ax, "a", dx=-0.105, dy=1.10)

    both = len(frames) > 1
    biased = {n: f[f["lam"] >= 10] for n, f in frames.items()}
    peak = {n: float(f["ratio"].max()) for n, f in biased.items() if not f.empty}
    graft_peak = peak.get("grafting segment")
    free_peak = peak.get("all sites free")

    # The headline is READ OFF THE DATA. It used to assert the earlier model's
    # conclusion ("freeing every site does not dilute the red sector"), which is
    # false here: this model's free run sits at ~0.95, i.e. no enrichment at all.
    if both and graft_peak is not None and free_peak is not None:
        if free_peak < 1.10 <= graft_peak:
            headline = "Freeing every site dilutes the red sector"
            sub = (f"restricted to the grafting segment the design puts up to "
                   f"$\\approx{graft_peak:.1f}\\times$ more mutations on the red sector "
                   f"than its size predicts;\nwith every column free that falls to "
                   f"$\\approx{free_peak:.2f}\\times$ — no enrichment at all")
        elif min(graft_peak, free_peak) >= 1.10:
            headline = "Freeing every site does not dilute the red sector"
            sub = (f"once the critic is biased ($\\lambda\\geq10$), both site sets put up to "
                   f"$\\approx{max(graft_peak, free_peak):.1f}\\times$ more mutations on the "
                   f"red sector than its size predicts")
        else:
            headline = "Red-sector enrichment depends on which columns are free"
            sub = (f"grafting segment peaks at $\\approx{graft_peak:.2f}\\times$, "
                   f"all sites free at $\\approx{free_peak:.2f}\\times$")
    else:
        only = graft_peak if graft_peak is not None else next(iter(peak.values()), float("nan"))
        headline = "The design concentrates its mutations on the red sector"
        sub = ("once the critic is biased ($\\lambda\\geq10$), the design puts up to "
               f"$\\approx{only:.1f}\\times$ more mutations on the red sector than its "
               "size predicts")
    fig.suptitle(headline, y=0.972)
    fig.text(0.5, 0.905, sub, ha="center", va="center", fontsize=13.5,
             color=style.MUTED, linespacing=1.4)

    if both:
        shares = {n: 100 * float(f["share_of_mutable_sites"].iloc[0])
                  for n, f in frames.items()}
        note = ("Enrichment is dimensionless, so the shapes of the two curves are "
                "directly comparable; their magnitudes are not. The free run also "
                "carries an ‘other’ sector,\nand red is only "
                f"{shares.get('all sites free', float('nan')):.1f}% of its mutable "
                f"sites against {shares.get('grafting segment', float('nan')):.1f}% "
                "under grafting, so the same ratio is reached from a much smaller "
                "base. 1000 chains per point.")
    else:
        note = ("Enrichment = share of mutations on the red sector divided by its "
                "share of the mutable columns. 1000 chains per point.\n"
                "The all-sites-free comparison — does freeing every column dilute "
                "this? — is NOT shown: that run was not sampled.")
    style.caveat(fig, note, y=0.015)
    style.save(fig, STEM)


if __name__ == "__main__":
    main()
