"""S5 -- the t ladder: one knob moved, five ensembles at increasing distance.

THE POINT. S2 shows what every knob does across the factorial, which is the
right figure for comparing knobs and the wrong one for *seeing* what one of them
does. This is the other figure: everything held except ``t``, five rungs, and
the ensembles drawn as the clouds they actually are rather than as five points
on a line. If the restraint controls the mutation budget then the five clouds
sit at increasing distance and separate cleanly -- and if they overlap, no
summary of means would have told you.

THE RUNG CHOICE, and it is not arbitrary:

  set      ``segment``, 79 columns. The widest span in <d> of the three sets at
           fixed (beta, lambda), so the clouds have room to separate. It is also
           the set the ladder's own t = 0 figures use, which keeps this
           comparable to ``figures_ladder/fig02``.
  beta     1. Warm, so none of these is the frozen pathology of CLAMP_CONTROL
           caveat 1, and acceptance stays well above the floor at every rung.
  lambda   300. P(S) ~ 0.97, so the ladder walks the budget axis at a
           specificity that is already high -- the regime a construct would be
           commissioned in. At lambda = 100 the same ladder sits ~0.06 lower in
           P(S) and the clouds overlap more in Q.
  t        -0.5, 0, 0.5, 1, 2. Four of these are the factorial's own grid; the
           negative rung is the extension, and it is what makes this a ladder
           rather than a half-ladder -- t < 0 PAYS the chains to leave the wild
           type, so it is the only rung above the unrestrained <d>.

COLOUR IS AN ORDERED MAGNITUDE, not a category: a single-hue light-to-dark ramp,
the same convention the ladder uses for lambda. Five categorical colours would
imply five unrelated conditions, which is the opposite of what a ladder is.

WHAT TO READ. Panel (a) is the claim itself -- five separated distributions in
d. (b), (c) and (d) are its price, each against the same d axis, so the figure
can be read as "what does the budget buy, and what does it cost".
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import selection
import style

#: Which mutation set the ladder is drawn for. Overridable from the command line
#: (``python analysis/s05_ladder.py redsector``) so the same figure can be made
#: for each of the three sets without a second copy of the script.
SET = os.environ.get("LADDER_SET", "segment")
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    SET = sys.argv[1]
if SET not in config.SETS:
    raise SystemExit(f"unknown set {SET!r}; expected one of {list(config.SETS)}")
BETA = float(os.environ.get("LADDER_BETA", 1.0))
LAM = float(os.environ.get("LADDER_LAM", 300.0))
#: Five rungs, all on the RESTRAINED side. A negative rung was tried first and
#: dropped: t < 0 pays the chains to leave the wild type, and at 60,000 steps
#: t = -0.5 missed the gate (drift/spread 0.065 and 0.068 against the 0.05 rule,
#: the indel part worst at 0.118) with acceptance at 47.7 per site -- still
#: moving, not frozen. Extending it to 240,000 steps would have fixed it but
#: would have made one rung cost 4x the others. t = 4 instead: same 60,000
#: steps as every other rung, so the ladder is five runs of identical cost, and
#: it still spans <d> ~ 19 -> 7.
T_LADDER = (0.0, 0.5, 1.0, 2.0, 4.0)

#: Point size: these are five clouds of 1000, not one lambda ramp, so they are
#: drawn larger than fig02's s=3.4 for the same reason s03 is.
POINT_SIZE = 20.0
POINT_ALPHA = 0.38


def ladder() -> list[tuple[float, pd.DataFrame]]:
    """``(t, per-chain frame)`` for each rung that exists and passes the gate."""
    from ensemble import run_dirname

    # grid_only=False: t = 4 is deliberately outside config.T_VALUES, so that it
    # cannot leak into the factorial that s01 and s02 read.
    # betas=(BETA,) so a ladder at a beta outside the factorial (beta = 3) is
    # found at all; grid_only=False so t = 4 is, without either leaking into the
    # 48-run factorial that s01 and s02 read.
    frame = data.load_all(serine=True, only_saturated=True, grid_only=False,
                          betas=(BETA,))
    subset = frame[(frame["set"] == SET) & np.isclose(frame["beta"], BETA)
                   & np.isclose(frame["lam"], LAM)]
    out = []
    for t in T_LADDER:
        rung = subset[np.isclose(subset["t"], t)]
        if rung.empty:
            print(f"  t = {t:g}: not sampled or failed the gate -- skipped")
            continue
        out.append((t, rung))
    return out


def main() -> None:
    style.apply()
    config.ensure_dirs()

    rungs = ladder()
    if len(rungs) < 2:
        raise SystemExit(
            f"the ladder needs at least two rungs; found {len(rungs)}. "
            f"Sample the missing t with:\n"
            f"  T_VALUES='-0.5' LAMBDAS='{LAM:g}' BETAS='{BETA:g}' "
            f"bash t_steering/run_steering.sh {SET} post check"
        )

    # The ramp is built from this SET's hue, so the same set is the same colour
    # here, in the corner plot, and in every factorial figure.
    shades = style.ramp(config.SET_COLOUR[SET], len(rungs))
    q_wt = selection.wild_type_Q()

    table = data.summarise(
        pd.concat([r for _, r in rungs], ignore_index=True)
    ).sort_values("t")
    print(f"\n  the ladder: {SET}, beta={BETA:g}, lambda={LAM:g}, "
          f"{len(rungs)} rungs")
    cols = ["t", "hamming", "hamming_sd", "E_DCA", "CDE", "Q", "P_S", "LOR",
            "n_unique", "divergence_ratio", "accepted_per_site"]
    print(table[[c for c in cols if c in table]].to_string(
        index=False, float_format=lambda v: f"{v:9.3f}"))

    # --- the scatter figure ----------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.2))

    # (a) the claim: five distributions in d
    ax = axes[0, 0]
    lo = min(r["hamming_to_wt"].min() for _, r in rungs)
    hi = max(r["hamming_to_wt"].max() for _, r in rungs)
    for (t, rung), shade in zip(rungs, shades):
        ax.hist(rung["hamming_to_wt"], bins=np.arange(lo - 0.5, hi + 1.5),
                histtype="stepfilled", color=shade, alpha=0.42, lw=0, zorder=2)
        ax.hist(rung["hamming_to_wt"], bins=np.arange(lo - 0.5, hi + 1.5),
                histtype="step", color=shade, lw=1.5, zorder=3,
                label=f"$t$ = {t:g}")
    ax.axvline(0, color=style.C_WT, ls=":", lw=1.3, zorder=1)
    ax.text(0, ax.get_ylim()[1], " wild type", color=style.C_WT, fontsize=9,
            va="top")
    ax.set_xlabel(r"$d_H(\sigma,\mathrm{WT})$")
    ax.set_ylabel("chains")
    ax.set_title("(a)  one knob, five budgets", loc="left")
    ax.legend(frameon=False, fontsize=10)
    style.tidy(ax)

    # (b)-(d) the price, each against the same d axis
    for ax, (column, label, letter, subtitle) in zip(
        axes.ravel()[1:],
        [("E_DCA", r"$E_{\mathrm{DCA}}$   (lower is better)", "b",
          "what it costs in energy"),
         ("Q", r"$Q$   (lower is better)", "c", "and in quality"),
         ("P_S", style.P_S, "d", "what it costs in specificity"),
         ("LOR", style.LOG_ODDS, "e", "the same, on an unsaturating readout"),
         ("CDE", "context-dependent entropy", "f",
          r"the other half of $Q$")],
    ):
        for (t, rung), shade in zip(rungs, shades):
            ax.scatter(rung["hamming_to_wt"], rung[column], s=POINT_SIZE,
                       alpha=POINT_ALPHA, color=shade, linewidths=0, zorder=2)
            ax.scatter([rung["hamming_to_wt"].mean()], [rung[column].mean()],
                       s=150, marker="o", facecolor=shade,
                       edgecolor=style.SURFACE, linewidth=1.8, zorder=4)
        if column == "Q":
            ax.axhline(q_wt, color=style.C_WT, ls=":", lw=1.3, zorder=1)
            ax.text(ax.get_xlim()[1], q_wt, " wild type ", color=style.C_WT,
                    fontsize=9, va="bottom", ha="right")
        ax.set_xlabel(r"$d_H(\sigma,\mathrm{WT})$")
        ax.set_ylabel(label)
        ax.set_title(f"({letter})  {subtitle}", loc="left")
        style.tidy(ax)

    for ax, letter in zip(axes.ravel(), "abcdef"):
        style.panel_letter(ax, letter)

    fig.suptitle(
        rf"S5 — the $t$ ladder: {SET}, $\beta$={BETA:g}, $\lambda$={LAM:g}, "
        rf"{len(rungs)} rungs", fontsize=13)
    style.caveat(
        fig,
        "1000 chains per rung, 60,000 steps, every rung past the S1 gate, read "
        r"through the $\beta=1$ serine rescoring.  large markers are the rung "
        "means." "\n"
        r"colour is an ordered magnitude (one hue, light to dark), not a "
        r"category: five settings of one knob, not five conditions." "\n"
        r"all five rungs are on the restrained side ($t\geq0$) and cost the same "
        r"60,000 steps.  sd$(d)$ falls 2.67 $\to$ 1.17 across the ladder, which "
        r"is $\mathrm{Var}(d)=-\partial\langle d\rangle/\partial t$ made visible.",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    style.save(fig, f"s05_ladder_{SET}_beta{BETA:g}", out_dir=config.OUT_DIR)

    # --- the table -------------------------------------------------------------
    columns = [
        ("t", "$t$", "{:g}"),
        ("hamming", r"$\langle d\rangle$", "{:.2f}"),
        ("hamming_sd", r"sd$(d)$", "{:.2f}"),
        ("E_DCA", "$E$", "{:.1f}"),
        ("CDE", "CDE", "{:.3f}"),
        ("hamming_q25", "$q_{25}$", "{:.0f}"),
        ("hamming_q75", "$q_{75}$", "{:.0f}"),
        ("Q", "$Q$", "{:.1f}"),
        ("P_S", "$P(S)$", "{:.3f}"),
        ("LOR", "LOR", "{:.2f}"),
        ("n_unique", "distinct", "{:.0f}"),
        ("divergence_ratio", "div/$\\langle d\\rangle$", "{:.2f}"),
        ("accepted_per_site", "acc/site", "{:.0f}"),
    ]
    present = [(k, h, f) for k, h, f in columns if k in table]
    cells = [[f.format(v) for v in table[k]] for k, _, f in present]

    fig2, ax2 = plt.subplots(figsize=(1.28 * len(present) + 1.0,
                                      0.62 * len(table) + 2.4))
    ax2.axis("off")
    artist = ax2.table(cellText=list(map(list, zip(*cells))),
                       colLabels=[h for _, h, _ in present],
                       cellLoc="center", loc="upper center")
    artist.auto_set_font_size(False)
    artist.set_fontsize(10.5)
    artist.scale(1.0, 1.55)
    for (row, col), cell in artist.get_celld().items():
        cell.set_edgecolor(style.FAINT)
        if row == 0:
            cell.set_text_props(weight="bold", color=style.INK)
            cell.set_facecolor(style.GRID)
    ax2.set_title(
        rf"S5 — the $t$ ladder in numbers ({SET}, $\beta$={BETA:g}, "
        rf"$\lambda$={LAM:g})", fontsize=13, pad=18)
    style.caveat(
        fig2,
        r"every row is the same model, the same start, the same 79 columns and "
        r"the same 60,000 steps.  only $t$ differs." "\n"
        rf"wild-type $Q$ = {q_wt:.1f}; the natural family's own sd$(Q)$ is "
        rf"{config.natural_quality_sd():.1f}, so the whole ladder lies within "
        "half of one family standard deviation of it.",
    )
    fig2.tight_layout(rect=(0, 0.12, 1, 1.0))
    style.save(fig2, f"s05_ladder_table_{SET}_beta{BETA:g}",
               out_dir=config.OUT_DIR)

    out = config.RESULTS_ROOT / f"s05_ladder_{SET}_beta{BETA:g}.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
