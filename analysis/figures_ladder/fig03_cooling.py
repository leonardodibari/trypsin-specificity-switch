"""F3 -- the beta / lambda-tilde cooling scan.

One claim, four panels: cooling the sampler (raising beta) buys plausibility at
essentially no cost in P(S), and pays for it in library diversity.

  (a) the switch still fires        -- P(S | sigma_\\i*) vs lambda-tilde
  (b) the designs get plausible     -- Q vs lambda-tilde, against Q of the wild type
  (c) the library collapses         -- mean pairwise Hamming
  (d) ... and stops being a library -- number of distinct chains out of 1000

beta is an ORDERED knob, so it is drawn with a sequential tint->shade ramp built
from style.BLUE. Never a rainbow: a reader must be able to say "darker = colder"
without consulting the legend.

Source: results/ladder/cooling/beta_scan_summary.csv (ensemble means, 1000
chains per point). lambda-tilde = lambda / beta is the *effective* bias, which
is what makes the beta = 1 scan and the cooled scans comparable on one x axis.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import config
import data
import style

SUMMARY = config.RESULTS_ROOT / "cooling" / "beta_scan_summary.csv"
STEM = "fig03_cooling"

LAM_TICKS = [0, 10, 100, 1000]


def beta_ramp(n: int) -> list:
    """A sequential light -> dark ramp built from style.BLUE.

    Derived from the shared palette rather than invented: a pale tint of BLUE,
    through BLUE itself, to BLUE shaded towards style.INK. No new hue enters the
    figure, and the ordering of beta is readable without the legend.
    """
    base = np.array(mcolors.to_rgb(style.BLUE))
    ink = np.array(mcolors.to_rgb(style.INK))
    light = 0.56 * np.ones(3) + 0.44 * base
    dark = 0.26 * base + 0.74 * ink
    cmap = mcolors.LinearSegmentedColormap.from_list("beta", [light, base, dark])
    return [cmap(t) for t in np.linspace(0.0, 1.0, n)]


def load() -> pd.DataFrame:
    """The cooling scan, built from the ensembles themselves.

    The first model's pipeline wrote a beta_scan_summary.csv; the second's does
    not, and regenerating it would mean keeping a third summary schema alive
    beside the two that already disagree. Computing the same quantities from the
    sampled sequences through data.run_metrics is both model-agnostic and the
    single definition the tables and corner plots already use -- in particular
    P(S) is the beta = 1 conditional recomputed from the sequences, NOT
    summary.json's mean_P_S, which is the fraction of chains carrying S and is
    1.0 for every cooled run.
    """
    # The uncooled beta = 1 ladder is the BASELINE the whole figure is measured
    # against. This model's cooling ladder starts at beta = 1.5, so without this
    # the figure plots only cooled runs and the claim "cooling buys plausibility"
    # has nothing in frame to compare with.
    runs = data.discover_runs("grafting, beta=1") + data.discover_runs("grafting + cooling")
    if not runs:
        if SUMMARY.is_file():        # the first model's stored summary
            frame = pd.read_csv(SUMMARY)
            return frame.sort_values(["beta", "lam_tilde"], kind="mergesort")
        raise SystemExit("No cooling ensembles found -- run STAGE 1 first")
    rows = []
    for run in runs:
        metrics = data.run_metrics(run)
        metrics["lam_tilde"] = run.lam / run.beta if run.beta else run.lam
        metrics["n_unique"] = metrics["pct_unique"] / 100.0 * metrics["n_chains"]
        metrics["hamming_to_wt"] = metrics["hamming_to_wt"]
        rows.append(metrics)
    frame = pd.DataFrame(rows).rename(columns={"P_S": "P_S", "Q": "Q"})
    return frame.sort_values(["beta", "lam_tilde"], kind="mergesort")


def fmt_beta(value: float) -> str:
    return f"{value:g}"


def draw_panel(ax, frame, column, colours, betas, *, ylabel, letter):
    # The lam_tilde grid differs per model (the second model needs ~10x less
    # bias), so the axis is taken from the data rather than hard-coded.
    hi = float(frame["lam_tilde"].max())
    for beta, colour in zip(betas, colours):
        block = frame[frame["beta"] == beta].sort_values("lam_tilde")
        ax.plot(block["lam_tilde"], block[column], marker="o", ms=6.5, lw=2.3,
                color=colour, markeredgecolor="white", markeredgewidth=0.8,
                clip_on=False, zorder=3 + (beta == betas[0]))
    ax.set_xscale("symlog", linthresh=10, linscale=0.55)
    ax.set_xlim(-0.12 * hi / 10, hi * 1.45)
    ax.set_xticks(LAM_TICKS)
    ax.set_xticklabels([f"{t:g}" for t in LAM_TICKS])
    ax.minorticks_off()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"effective bias   $\tilde\lambda=\lambda/\beta$")
    style.tidy(ax)
    style.panel_letter(ax, letter, dx=-0.155)


def main() -> None:
    style.apply()
    frame = load()
    betas = sorted(frame["beta"].unique())
    colours = beta_ramp(len(betas))
    q_wt = data.wild_type_reference()["Q"]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.9))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.790, bottom=0.085,
                        hspace=0.38, wspace=0.30)
    (ax_ps, ax_q), (ax_div, ax_uni) = axes

    draw_panel(ax_ps, frame, "P_S", colours, betas,
               ylabel=style.LABELS["P_S"], letter="a")
    ax_ps.set_ylim(-0.03, 1.05)
    ax_ps.axhline(0.80, color=style.RULE, lw=1.2, ls=(0, (1, 2)), zorder=1)
    ax_ps.text(1650, 0.80, "gate 0.80", ha="right", va="bottom",
               fontsize=12, color=style.MUTED)
    ax_ps.set_title("the switch still fires", pad=10, color=style.MUTED,
                    fontsize=15)

    draw_panel(ax_q, frame, "Q", colours, betas,
               ylabel=style.LABELS["Q"], letter="b")
    ax_q.axhline(q_wt, color=style.C_WT, lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax_q.text(-1.0, q_wt + 0.9, "wild type", ha="left", va="bottom",
              fontsize=12.5, color=style.C_WT)
    ax_q.set_ylim(-120, -82)
    ax_q.set_title("and the designs become plausible", pad=10,
                   color=style.MUTED, fontsize=15)
    hot = frame[(frame["beta"] == betas[0]) & (frame["lam_tilde"] == 1000)]
    cold = frame[(frame["beta"] == 3.0) & (frame["lam_tilde"] == 1000)]
    if len(hot) and len(cold):
        x_arrow = 1480
        ax_q.annotate("", xy=(x_arrow, float(cold["Q"].iloc[0])),
                      xytext=(x_arrow, float(hot["Q"].iloc[0])),
                      arrowprops=dict(arrowstyle="-|>", color=style.MUTED,
                                      lw=1.5, shrinkA=2, shrinkB=2))
        ax_q.text(x_arrow * 0.86,
                  0.5 * (float(cold["Q"].iloc[0]) + float(hot["Q"].iloc[0])),
                  "colder", ha="right", va="center", fontsize=12.5,
                  color=style.MUTED, rotation=90)

    draw_panel(ax_div, frame, "mean_pair_hamming", colours, betas,
               ylabel="mean pairwise Hamming", letter="c")
    ax_div.set_ylim(0, 30)
    ax_div.set_title("but the library collapses", pad=10, color=style.MUTED,
                     fontsize=15)

    draw_panel(ax_uni, frame, "n_unique", colours, betas,
               ylabel="distinct chains (of 1000)", letter="d")
    ax_uni.set_ylim(0, 1080)
    ax_uni.set_title("down to a handful of distinct designs", pad=10,
                     color=style.MUTED, fontsize=15)

    handles = [Line2D([], [], color=c, lw=2.6, marker="o", ms=6.5,
                      markeredgecolor="white", markeredgewidth=0.8,
                      label=fmt_beta(b)) for b, c in zip(betas, colours)]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.895), ncol=len(betas),
               title=r"inverse sampling temperature  $\beta$   (pale = hot, dark = cold)",
               handlelength=2.0, columnspacing=1.6, handletextpad=0.6)

    # Quantify the claim from the data, at matched effective bias, so the
    # headline is checkable rather than asserted.
    warm = frame[frame["beta"] == 1.0].set_index("lam_tilde")["Q"]
    gains, firsts = [], []
    for lt, block in frame[frame["beta"] > 1].groupby("lam_tilde"):
        if lt in warm.index:
            gains.append(float(warm[lt] - block["Q"].min()))
            first = block.sort_values("beta").iloc[0]
            firsts.append(float(warm[lt] - first["Q"]))
    gain = np.median(gains) if gains else float("nan")
    first_step = np.median(firsts) if firsts else float("nan")
    fig.suptitle("Cooling buys plausibility at the price of diversity", y=0.972)
    fig.text(0.5, 0.936,
             f"at matched $\\tilde\\lambda$ cooling lowers $Q_\\mu$ by "
             f"$\\approx{gain:.0f}$ with no loss of $P(\\mathrm{{S}})$ — but "
             f"$\\approx{100 * first_step / gain:.0f}\\%$ of that arrives at the "
             f"first step, $\\beta=1\\rightarrow1.5$, and the rest costs the library",
             ha="center", va="center", fontsize=13, color=style.MUTED)
    style.caveat(fig,
                 "cooling scan, 1000 chains per point, ensemble means; "
                 r"$\tilde\lambda=\lambda/\beta$ is the effective bias, so $\beta=1$ is the uncooled scan. "
                 r"$Q_\mu=E_{\mathrm{DCA}}-\mu\,\mathrm{CDE}$; the dashed rule is the wild type.")
    style.save(fig, STEM)


if __name__ == "__main__":
    main()
