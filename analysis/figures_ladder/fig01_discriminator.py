"""F1 -- why P(S | sigma_\\i*) is a discriminator for chymotrypsin.

(a) P(S|.) and (b) P(D|.) over the natural alignment, split by the residue each
sequence actually carries at the critic site i*; (c) ROC and (d) precision-recall
for chymotrypsins against trypsins.

The conditional at i* does not depend on the residue sitting at i*, so scoring a
natural sequence is a leave-one-site-out test: can the model read a sequence's
specificity off its context alone?

WHEN A HELD-OUT SPLIT IS AVAILABLE this figure is evaluated on it, and the claim
is genuine prediction rather than a consistency check. The serine-protease model
was fitted on a 71,551-sequence train split of this alignment, so the remaining
17,888 sequences are scored by a model that never saw them. The training curve is
drawn alongside, thin and dashed, so the generalisation gap is visible rather
than hidden: if the two separate, the figure says so.

Without a split the whole alignment is used and every caption says that this is
consistency, not prediction. The split is never guessed -- see data.split_mask.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import config
import data
import metrics
import style


def main() -> None:
    style.apply()
    P, _ = data.natural_conditional()
    classes = data.natural_classes()
    held_out = data.split_mask()
    # Evaluate on the held-out split when there is one; otherwise everything.
    evaluated = held_out if held_out is not None else np.ones(len(P), dtype=bool)
    trypsin = classes["trypsin"] & evaluated
    chymo = classes["chymotrypsin"] & evaluated
    n0, n1 = int(trypsin.sum()), int(chymo.sum())
    prevalence = n1 / (n1 + n0)
    s_state, d_state = data.S_STATE(), data.D_STATE()
    # The training comparison, drawn thin and dashed where it exists.
    train_trypsin = classes["trypsin"] & ~evaluated if held_out is not None else None
    train_chymo = classes["chymotrypsin"] & ~evaluated if held_out is not None else None
    split_label = "held out" if held_out is not None else "all"

    series = data.construct_series()
    series_P = data.critic_conditional(np.stack([seq for _, seq in series]))

    scores = [
        (style.P_S, P[:, s_state], style.BLUE),
        (style.NEG_P_D, -P[:, d_state], style.ORANGE),
        (style.LOG_ODDS, data.log_odds(P), style.AQUA),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 12.4))
    fig.subplots_adjust(top=0.905, bottom=0.095, left=0.085, right=0.975,
                        hspace=0.30, wspace=0.235)
    (axA, axB), (axC, axD) = axes
    for ax in axes.ravel():
        style.tidy(ax)

    # --- (a), (b) the two conditionals over the natural families ---------------
    bins = np.linspace(0, 1, 61)
    for ax, state, symbol in [(axA, s_state, r"\mathrm{S}"), (axB, d_state, r"\mathrm{D}")]:
        for mask, colour, label in [
            (trypsin, style.C_TRYPSIN,
             f"natural trypsins (D at $i^*$), n = {n0:,} {split_label}"),
            (chymo, style.C_CHYMO,
             f"natural chymotrypsins (S at $i^*$), n = {n1:,} {split_label}"),
        ]:
            values = P[mask, state]
            ax.hist(values, bins=bins, weights=np.full(len(values), 1 / len(values)),
                    color=colour, alpha=0.72, label=label)
        for index, probabilities in enumerate(series_P):
            ax.axvline(probabilities[state], color=style.C_1992, ls=":", lw=1.8, zorder=5)
            ax.annotate(str(index + 1), xy=(probabilities[state], 1.0),
                        xycoords=("data", "axes fraction"), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=12, color=style.INK, zorder=6,
                        bbox=dict(boxstyle="circle,pad=0.16", fc="white",
                                  ec=style.INK, lw=1.0))
        ax.set_yscale("log")
        ax.set_ylim(8e-6, 1.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel(rf"$P({symbol}\mid\sigma_{{\setminus i^*}})$")
        ax.set_ylabel("fraction of family per bin")

    key = "1992 construct series (reconstructed)\n" + "\n".join(
        f"{i + 1}.  {name}" for i, (name, _) in enumerate(series))
    axA.text(0.295, 2.3e-5, key, fontsize=11.5, color=style.MUTED, va="bottom",
             ha="left", linespacing=1.55,
             bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=style.FAINT, lw=1.0))

    # --- (c) ROC ---------------------------------------------------------------
    axC.plot([0, 1], [0, 1], color=style.RULE, lw=1.6, ls="--", zorder=1)
    axC.text(0.205, 0.228, "chance", color=style.MUTED, fontsize=13, rotation=36,
             ha="center", va="center")
    roc_curves = []
    for label, score, colour in scores:
        fpr, tpr, area = metrics.roc(score, chymo, trypsin)
        roc_curves.append((label, fpr, tpr, area, colour))
        axC.plot(fpr, tpr, color=colour, lw=2.6, zorder=3,
                 label=f"{label}   AUC = {area:.4f}")
    if train_chymo is not None:
        # The same score on the sequences the model was fitted on. The distance
        # between this and the solid green curve IS the generalisation gap.
        t_fpr, t_tpr, t_auc = metrics.roc(scores[2][1], train_chymo, train_trypsin)
        axC.plot(t_fpr, t_tpr, color=style.AQUA, lw=1.5, ls=(0, (5, 3)), zorder=2,
                 label=f"      same, on training seqs   {t_auc:.4f}")
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1.004)
    axC.set_xlabel("false positive rate")
    axC.set_ylabel("true positive rate")
    axC.legend(loc="lower right", bbox_to_anchor=(1.01, 0.025), handlelength=1.5,
               fontsize=13.5)
    inset = axC.inset_axes([0.40, 0.42, 0.56, 0.40])
    for _, fpr, tpr, _, colour in roc_curves:
        inset.plot(fpr, tpr, color=colour, lw=2.1)
    inset.set_xlim(0, 0.08)
    inset.set_ylim(0.80, 1.004)
    _dress_inset(inset)
    axC.indicate_inset_zoom(inset, edgecolor=style.RULE, lw=1.1, alpha=1.0)

    # --- (d) precision-recall ---------------------------------------------------
    axD.axhline(prevalence, color=style.RULE, lw=1.6, ls="--", zorder=1)
    axD.text(0.74, prevalence + 0.030, f"chance = {prevalence:.3f}",
             color=style.MUTED, fontsize=13, ha="left", va="bottom")
    pr_curves = []
    for label, score, colour in scores:
        recall, precision, ap = metrics.pr(score, chymo, trypsin)
        pr_curves.append((label, recall, precision, ap, colour))
        axD.plot(recall, precision, color=colour, lw=2.6, zorder=3,
                 label=f"{label}   AP = {ap:.4f}")
    if train_chymo is not None:
        t_rec, t_prec, t_ap = metrics.pr(scores[2][1], train_chymo, train_trypsin)
        axD.plot(t_rec, t_prec, color=style.AQUA, lw=1.5, ls=(0, (5, 3)), zorder=2,
                 label=f"      same, on training seqs   {t_ap:.4f}")
    axD.set_xlim(0, 1)
    axD.set_ylim(0, 1.02)
    axD.set_xlabel("recall")
    axD.set_ylabel("precision")
    axD.legend(loc="lower left", bbox_to_anchor=(-0.012, 0.055), handlelength=1.5,
               fontsize=12.5)
    # The three curves are indistinguishable until ~0.85 recall and then fan out.
    # An inset spanning the full precision range wastes four fifths of its area
    # on the vertical collapse at the very end and squashes the fan into a sliver,
    # so it is cropped to where the curves actually separate.
    inset2 = axD.inset_axes([0.44, 0.56, 0.53, 0.34])
    for _, recall, precision, _, colour in pr_curves:
        inset2.plot(recall, precision, color=colour, lw=2.2)
    inset2.set_xlim(0.85, 1.0)
    inset2.set_ylim(0.55, 1.01)
    inset2.set_xticks([0.85, 0.90, 0.95, 1.00])
    inset2.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    _dress_inset(inset2)
    axD.indicate_inset_zoom(inset2, edgecolor=style.RULE, lw=1.1, alpha=1.0)

    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, fontsize=15, loc="upper center",
               bbox_to_anchor=(0.5, 0.962))
    headline = ("The critic-site conditional separates trypsins from chymotrypsins"
                + ("  —  on sequences the model never saw"
                   if held_out is not None else ""))
    fig.suptitle(headline, y=0.985)
    for letter, ax in zip("abcd", (axA, axB, axC, axD)):
        style.panel_letter(ax, letter)
    style.caveat(fig, data.split_status().capitalize() + ".\n"
                 "Dashed: the same score on the sequences the model was fitted on — "
                 "the distance between the solid and dashed curves is the "
                 "generalisation gap.", y=0.014)

    style.save(fig, "fig01_discriminator")

    for label, _, _, area, _ in roc_curves:
        ap = next(c[3] for c in pr_curves if c[0] == label)
        print(f"  {label:<48} AUC = {area:.4f}   AP = {ap:.4f}")
    for (name, _), probabilities in zip(series, series_P):
        print(f"  {name:<30} P(S) = {probabilities[s_state]:.4f}  "
              f"P(D) = {probabilities[d_state]:.4f}")


def _dress_inset(ax) -> None:
    ax.tick_params(labelsize=11, colors=style.MUTED)
    ax.grid(color=style.GRID, lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(style.FAINT)


if __name__ == "__main__":
    main()
