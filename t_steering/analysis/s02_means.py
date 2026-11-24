"""S2 -- what each setting does to each quantity, across the whole factorial.

THE QUESTION. With 48 runs there are too many numbers to read as a list and too
few to fit a model to. What a factorial is actually for is reading off the
MARGINAL effect of each knob: how the mean of each quantity moves when one knob
is stepped and the other two are held.

HOW TO READ IT. Each panel is one quantity. The x axis is ``t``, one line per
(set, beta, lambda) cell, so a line's SLOPE is the effect of the restraint at a
fixed setting of everything else, and the VERTICAL SPACING between lines is the
effect of the other knobs at a fixed restraint. Colour is the mutation set;
line style is beta; shade is lambda. The wild type is drawn wherever it is
defined on the axis, because "similar to the wild type" is one of the
requirements and it should be readable, not computed.

WHY MEANS AND A SPREAD, NEVER A MEAN ALONE. Every panel that can carry its
ensemble's spread does: ``<d>`` gets its interquartile band, and ``P(S)`` and
``Q`` get theirs. A mean Hamming distance of 8 built out of chains at 3 and 13
is a different object from one built out of chains at 7 and 9, and the first is
what a batch of 1000 actually is.

THE TABLE this writes (``s02_means.csv``) is the full 48-row summary: one row
per run, every quantity's mean, the spread, the library size and the
acceptance. It is the thing to read numbers off; the figure is for seeing which
way each knob pushes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import selection
import style

#: The quantities worth a panel, in the order a reader should meet them.
PANELS = [
    ("hamming", r"$\langle d_H(\sigma,\mathrm{WT})\rangle$", "hamming_q25", "hamming_q75"),
    # The energy in its own right, not only inside Q. Q = E - mu*CDE with
    # mu ~ 423, so a small change in the context entropy moves Q far more than
    # the energy does -- two runs can share a Q and sit 20 apart in E_DCA, and
    # only this panel shows it.
    ("E_DCA", r"$E_{\mathrm{DCA}}$   (lower is better)", None, None),
    ("Q", r"$Q = E_{\mathrm{DCA}} - \mu\,\mathrm{CDE}$   (lower is better)", None, None),
    ("P_S", style.P_S, None, None),
    ("LOR", style.LOG_ODDS, None, None),
    ("n_unique", "distinct designs  (of 1000)", None, None),
    ("divergence_ratio", "pairwise divergence / mean d", None, None),
    ("accepted_per_site", "accepted moves / mutable site", None, None),
]

#: One per mutation set, from the single definition in config.
SET_COLOUR = config.SET_COLOUR
#: beta -> line style. Solid is the reference beta = 1.
BETA_STYLE = {1.0: "-", 1.5: (0, (4, 2))}


def main() -> None:
    style.apply()
    config.ensure_dirs()

    frame = data.load_all(serine=True, only_saturated=True)
    table = data.summarise(frame)

    out = config.RESULTS_ROOT / "s02_means.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out}  ({len(table)} runs)")

    q_wt = selection.wild_type_Q()
    print(f"  wild-type Q = {q_wt:.2f}  (the 'similar to WT' reference)")

    # --- the marginal effect of each knob, as plain differences of means -------
    print("\n  marginal effect of each knob (mean over everything else held):")
    for knob, values in (("t", config.T_VALUES), ("beta", config.BETAS),
                         ("lam", config.LAMBDAS)):
        lo, hi = min(values), max(values)
        rows = []
        for quantity, *_ in PANELS:
            if quantity not in table:
                continue
            a = table.loc[np.isclose(table[knob], lo), quantity].mean()
            b = table.loc[np.isclose(table[knob], hi), quantity].mean()
            rows.append(f"{quantity} {b - a:+.3g}")
        print(f"    {knob}: {lo:g} -> {hi:g}   " + "   ".join(rows))

    # --- the figure ------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(21.0, 9.8))
    for axis, (quantity, ylabel, lo_col, hi_col) in zip(axes.ravel(), PANELS):
        if quantity not in table:
            axis.set_visible(False)
            continue
        for set_name, set_rows in table.groupby("set", sort=False):
            shades = dict(zip(config.LAMBDAS,
                              style.ramp(SET_COLOUR[set_name], len(config.LAMBDAS))))
            for (beta, lam), cell in set_rows.groupby(["beta", "lam"], sort=True):
                cell = cell.sort_values("t")
                # linestyle=, not a positional format string: BETA_STYLE holds
                # dash tuples, which plot() only accepts as a keyword.
                axis.plot(cell["t"], cell[quantity],
                          linestyle=BETA_STYLE.get(beta, "-"), color=shades[lam],
                          marker="o", ms=4.5, lw=1.5, zorder=3)
                if lo_col and lo_col in cell:
                    axis.fill_between(cell["t"], cell[lo_col], cell[hi_col],
                                      color=shades[lam], alpha=0.12, lw=0, zorder=2)
        if quantity == "Q":
            axis.axhline(q_wt, color=style.C_WT, ls=":", lw=1.3, zorder=2)
            axis.text(axis.get_xlim()[1], q_wt, " wild type", color=style.C_WT,
                      fontsize=9, va="bottom", ha="right")
        if quantity == "P_S":
            axis.axhline(selection.P_S_FLOOR, color=style.RED, ls="--", lw=1.1, zorder=2)
        if quantity == "n_unique":
            axis.axhline(selection.N_UNIQUE_FLOOR, color=style.RED, ls="--", lw=1.1, zorder=2)
        if quantity == "divergence_ratio":
            axis.axhline(selection.DIVERGENCE_FLOOR, color=style.RED, ls="--", lw=1.1, zorder=2)
        axis.set_xlabel("restraint strength $t$")
        axis.set_ylabel(ylabel)
        style.tidy(axis)

    for axis, letter in zip(axes.ravel(), "abcdefgh"):
        if axis.get_visible():
            style.panel_letter(axis, letter)

    handles = [plt.Line2D([], [], color=SET_COLOUR[s], lw=2.0,
                          label=f"{s}  ({config.SETS[s][1]} cols)")
               for s in config.SETS]
    handles += [plt.Line2D([], [], color=style.MUTED, linestyle=BETA_STYLE[b],
                           lw=1.6, label=r"$\beta$=" + f"{b:g}")
                for b in config.BETAS]
    handles += [plt.Line2D([], [], color=style.MUTED, lw=1.6, alpha=a,
                           label=r"$\lambda$=" + f"{l:g}")
                for l, a in zip(config.LAMBDAS, (0.45, 1.0))]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=8, ncol=2, loc="best")

    fig.suptitle("S2 — what each setting does, across the 48-run factorial",
                 fontsize=13)
    style.caveat(
        fig,
        "60,000 steps, 1000 chains, all runs past the S1 gate, read through the "
        r"$\beta=1$ serine rescoring — the only form in which a $\beta=1$ and a "
        r"$\beta=1.5$ run are comparable." "\n"
        "bands are interquartile, not error on the mean: they are what a batch of "
        "1000 chains actually spans.  dashed red lines are the admissibility "
        "thresholds of selection.py.",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.945))
    style.save(fig, "s02_means", out_dir=config.OUT_DIR)


if __name__ == "__main__":
    main()
