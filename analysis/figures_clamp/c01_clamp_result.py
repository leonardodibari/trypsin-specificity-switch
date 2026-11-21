"""C1 -- what a clamped serine costs, and what the model does around it.

No bias, no lambda. The critic column is fixed at S and the rest is sampled from
H = beta * E_DCA. Three conditions, read against each other:

    clamp, grafting window   S imposed, cols 165-243 free
    clamp, all sites free    S imposed, the whole chain free
    no clamp (control)       wild-type D left in place, cols 165-243 free

The control is what makes the other two readable: cold sampling moves every
observable on its own, and without it a shift could be the temperature rather
than the serine.

(a) P(S | sigma_\\i*): a DIAGNOSTIC here, not a target. Nothing optimises it, so
    whatever it reads is the model's own opinion of how well the relaxed context
    supports an imposed serine.
(b) E_DCA against the wild type's own energy -- what the clamp costs.
(c) how far the chains travel from the wild type to pay for it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import style

PANELS = [("P_S", style.P_S),
          ("E_DCA", r"$E_{\mathrm{DCA}}$"),
          ("hamming_to_wt", "Hamming to wild type")]


def table() -> pd.DataFrame:
    rows = []
    for condition in config.CONDITIONS:
        for run in data.discover_runs(condition):
            config.check_run(run.path)
            rows.append(data.run_metrics(run))
    if not rows:
        raise SystemExit("No clamp ensembles found. Run: bash work/run_clamp.sh")
    return pd.DataFrame(rows).sort_values(["condition", "beta"])


def main() -> None:
    style.apply()
    frame = table()
    wt = data.wild_type_reference()

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.165, wspace=0.28)

    for ax, (key, label) in zip(axes, PANELS):
        for condition in config.CONDITIONS:
            part = frame[frame["condition"] == condition]
            if part.empty:
                continue
            spec = style.group(condition)
            ax.plot(part["beta"], part[key], marker=spec["marker"], ms=9, lw=2.2,
                    color=spec["color"], markeredgecolor="white",
                    markeredgewidth=1.1, label=condition, zorder=4)
        # The wild type is the reference every panel is read against, so it is
        # drawn in all three rather than quoted in the caption.
        if key in wt:
            ax.axhline(wt[key], color=style.C_WT, lw=1.5, ls=(0, (5, 3)), zorder=2)
            ax.annotate("wild type", xy=(0.985, wt[key]), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=12, color=style.MUTED)
        ax.set_xscale("log")
        ax.set_xticks(sorted(frame["beta"].unique()))
        ax.set_xticklabels([f"{b:g}" for b in sorted(frame["beta"].unique())])
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(label)
        style.tidy(ax)

    axes[0].legend(loc="best", labelspacing=0.5)
    for letter, ax in zip("abc", axes):
        style.panel_letter(ax, letter, dx=-0.155, dy=1.11)

    fig.suptitle("Clamp a serine at $i^*$ and let the model relax around it", y=0.965)
    style.caveat(
        fig,
        "P(S) is a diagnostic in these runs, not a target: lambda is 0 and nothing "
        "optimises it. The control is the same sampling with the wild type's D left "
        "in place, so the gap between the curves is what the serine did.",
        y=0.012)
    style.save(fig, "c01_clamp_result")

    show = ["condition", "beta", "n_allowed_sites", "P_S", "log_odds", "E_DCA",
            "Q", "hamming_to_wt", "subst_to_wt", "indel_to_wt", "pct_unique"]
    print(frame[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwild type: E={wt['E_DCA']:.1f}, P(S)={wt['P_S']:.4f}")


if __name__ == "__main__":
    main()
