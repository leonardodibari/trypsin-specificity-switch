"""S4 -- would steering on the log-odds instead of log P(S) change anything?

The sampler's bias is ``H = beta E_DCA - lambda log P(S | sigma_\\i*)``
(src/ensemble.py). On the natural alignment the log-odds
``log P(S|.)/P(D|.)`` is the better discriminator (F1: AUC 0.9946 vs 0.9921,
AP 0.9775 vs 0.9670), which raises the question of whether the bias should use it.

Switching the bias would invalidate every calibrated number -- mu, the lambda
grid, the 0.80 threshold -- so it is not a change to make casually. This figure
asks the cheap version of the question first: rescore the ensembles we already
have under both scores and see whether they disagree about which designs are
good. If the two orderings are effectively the same in the design regime, the
switch buys nothing; if they diverge, that is the evidence for making it.

Note the scores are NOT monotone transforms of one another: log P(S) ignores
P(D) entirely, so two chains with equal P(S) but different P(D) tie under the
bias and separate under the log-odds. An ROC cannot see that difference (it is
rank-invariant); a re-ranking can.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
import data
import selection
import style

TOP_K = 100


def main() -> None:
    style.apply()
    rows = []
    clouds = []
    for condition in config.CONDITIONS:
        for run in data.discover_runs(condition):
            frame = data.run_frame(run)
            log_p_s = np.log(np.clip(frame["P_S"].to_numpy(), 1e-12, None))
            odds = frame["log_odds"].to_numpy()
            rho = spearmanr(log_p_s, odds).statistic
            k = min(TOP_K, len(odds))
            top_bias = set(np.argsort(-log_p_s)[:k])
            top_odds = set(np.argsort(-odds)[:k])
            rows.append({
                "condition": condition, "beta": run.beta, "lam": run.lam,
                "spearman": rho,
                "top_overlap": 100.0 * len(top_bias & top_odds) / k,
                "mean_P_S": float(frame["P_S"].mean()),
            })
            clouds.append((condition, log_p_s, odds))
    table = pd.DataFrame(rows).sort_values(["condition", "beta", "lam"])

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.2))
    fig.subplots_adjust(top=0.84, bottom=0.30, left=0.065, right=0.985, wspace=0.30)
    axA, axB = axes
    for ax in axes:
        style.tidy(ax)

    # (a) the two scores, chain by chain
    for condition, log_p_s, odds in clouds:
        spec = style.group(condition)
        axA.scatter(log_p_s, odds, s=7, color=spec["color"], alpha=0.12,
                    linewidths=0, rasterized=True)
    axA.set_xlabel(r"$\log P(\mathrm{S}\mid\sigma_{\setminus i^*})$   (the bias)")
    axA.set_ylabel(style.LOG_ODDS)
    axA.set_title("the two scores, chain by chain", fontsize=15)

    # (b) do they agree on which chains are best?
    order = table.reset_index(drop=True)
    positions = np.arange(len(order))
    colours = [style.group(c)["color"] for c in order["condition"]]
    axB.barh(positions, order["top_overlap"], color=colours, alpha=0.85, height=0.7)
    # The condition is already carried by the bar colour and the figure legend,
    # so the tick labels only need the (beta, lambda) point. Spelling out the
    # condition here pushed the labels across into panel (a).
    axB.set_yticks(positions,
                   [f"β={b:g}, λ={l:g}"
                    for b, l in zip(order["beta"], order["lam"])],
                   fontsize=10.5)
    axB.invert_yaxis()
    axB.set_xlim(0, 100)
    axB.set_xlabel(f"overlap of the top {TOP_K} chains (%)")
    axB.set_title(f"do the two scores pick the same best {TOP_K} chains?", fontsize=15)
    axB.axvline(100, color=style.RULE, lw=1.4, ls="--")

    handles = [plt.Line2D([], [], marker="s", ls="none",
                          color=style.group(c)["color"], markersize=10,
                          label=c)
               for c in config.CONDITIONS if any(table["condition"] == c)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.075),
               ncol=3, fontsize=12.5)

    for letter, ax in zip("ab", axes):
        style.panel_letter(ax, letter, dx=-0.10, dy=1.09)
    fig.suptitle("Steering on log P(S) versus on the log-odds: do they disagree?",
                 y=0.965)
    median_overlap = table["top_overlap"].median()
    style.caveat(fig, f"Median top-{TOP_K} overlap {median_overlap:.0f}%, median "
                      f"Spearman {table['spearman'].median():.3f}. The bias is "
                      "unchanged in this work; this is the cheap test of whether "
                      "changing it would matter.", y=0.012)
    style.save(fig, "si04_logodds_rerank")

    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nmedian Spearman {table['spearman'].median():.4f}, "
          f"median top-{TOP_K} overlap {median_overlap:.1f}%")
    worst = table.nsmallest(3, "top_overlap")
    print("lowest agreement:")
    print(worst.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
