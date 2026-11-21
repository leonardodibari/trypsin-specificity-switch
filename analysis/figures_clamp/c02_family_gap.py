"""C2 -- does relaxing around a clamped serine move the protein toward the family?

SI S7 of the lambda pipeline found that the wild type sits ~82 residues from the
nearest natural chymotrypsin, that the 1992 constructs close none of that gap,
and that the lambda-steered designs close at most ~5%. The obvious next question
is whether the gap stays open because the BIAS is local, or because the model
does not actually know how to get there.

Clamp-and-relax separates those. There is no bias here at all: the serine is
imposed and the model is free to rearrange whatever it likes around it. If the
gap stays open under a free relaxation of the whole chain, the limitation is the
model's, not the objective's.

(a) distance to the nearest natural chymotrypsin, per condition, against beta.
(b) the same thing as a fraction of the wild type's own gap -- 0% means no
    progress, 100% would mean landing on a natural chymotrypsin.

The control answers the obvious objection: a chain that wanders far enough gets
closer to everything, so the clamped curves only mean something next to a run
that wandered the same way without the serine.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import config
import data
import style

MAX_SEQS = 300    # per run; the distance sweep is O(N * 89,439 * L)


def main() -> None:
    style.apply()
    classes = data.natural_classes()
    _, encoded = data.alignment()
    chymos = encoded[classes["chymotrypsin"]]
    wild = data.wild_type()
    base_gap = float(data.nearest_natural(wild[None, :], chymos)[0][0])
    length = encoded.shape[1]
    rng = np.random.default_rng(0)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.6, 5.6))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.155, wspace=0.26)

    summary = []
    for condition in config.CONDITIONS:
        runs = sorted(data.discover_runs(condition), key=lambda r: r.beta)
        if not runs:
            continue
        betas, gaps = [], []
        for run in runs:
            config.check_run(run.path)
            sequences = data.run_sequences(str(run.path))
            if len(sequences) > MAX_SEQS:
                sequences = sequences[rng.choice(len(sequences), MAX_SEQS, replace=False)]
            gap = float(data.nearest_natural(sequences, chymos)[0].mean())
            betas.append(run.beta)
            gaps.append(gap)
            summary.append((condition, run.beta, gap, 100 * (base_gap - gap) / base_gap))
        spec = style.group(condition)
        for ax, values in ((ax_a, gaps),
                           (ax_b, [100 * (base_gap - g) / base_gap for g in gaps])):
            ax.plot(betas, values, marker=spec["marker"], ms=9, lw=2.2,
                    color=spec["color"], markeredgecolor="white",
                    markeredgewidth=1.1, label=condition, zorder=4)

    for ax, value, text in ((ax_a, base_gap, "wild type"), (ax_b, 0.0, "wild type")):
        ax.axhline(value, color=style.C_WT, lw=1.5, ls=(0, (5, 3)), zorder=2)
        ax.annotate(text, xy=(0.985, value), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=12, color=style.MUTED)

    betas_all = sorted({b for _, b, _, _ in summary})
    for ax in (ax_a, ax_b):
        ax.set_xscale("log")
        ax.set_xticks(betas_all)
        ax.set_xticklabels([f"{b:g}" for b in betas_all])
        ax.set_xlabel(r"$\beta$")
        style.tidy(ax)
    ax_a.set_ylabel("Hamming to nearest\nnatural chymotrypsin")
    ax_b.set_ylabel("% of the wild-type gap closed")
    ax_a.legend(loc="best", labelspacing=0.5)
    for letter, ax in zip("ab", (ax_a, ax_b)):
        style.panel_letter(ax, letter, dx=-0.155, dy=1.11)

    fig.suptitle("Relaxing around the serine does not reach the chymotrypsins", y=0.965)
    style.caveat(
        fig,
        f"The wild type sits {int(base_gap)} residues ({100 * base_gap / length:.0f}% of L) "
        "from the nearest natural chymotrypsin. There is no bias in these runs: the model "
        "is free to rearrange anything it likes around the imposed serine, including the "
        "whole chain. The gap it closes is the gap it knows how to close.",
        y=0.012)
    style.save(fig, "c02_family_gap")

    print(f"wild-type gap: {base_gap:.0f} residues ({100 * base_gap / length:.1f}% of L)")
    print(f"{'condition':<26}{'beta':>6}{'gap':>8}{'closed':>9}")
    for condition, beta, gap, closed in summary:
        print(f"{condition:<26}{beta:>6g}{gap:>8.1f}{closed:>8.1f}%")


if __name__ == "__main__":
    main()
