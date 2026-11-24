"""S6 -- the t ladder as a 5x5 corner plot.

``s05`` puts every observable against the budget axis, which answers "what does a
shorter budget cost". This asks the other question: how do the five ensembles sit
against *each other* in every pair of observables at once. A ladder in a corner
plot should look like a ladder -- five clouds marching along the diagonal of the
pairs that the restraint acts on, and stacked on top of each other in the pairs
it does not.

THE FIVE, and why not the five of ``s03``. ``s03`` uses the ladder-figure set,
whose fifth axis is the distance to the nearest natural S-carrying sequence.
That column is written by ``add_natural_neighbour.py``, and ``s03_prepare.sh``
only ran it on the blocks holding a SELECTED run -- the ladder lives in
``segment/beta1``, which is not one of them. Rather than score 89,439 naturals
for a figure that does not need them, this uses

    d_H, E_DCA, CDE, Q, P(S)

which has two advantages of its own: every one of the five is defined for the
WILD TYPE, so the reference point appears in all twenty panels rather than
fifteen; and it carries ``E_DCA`` and ``CDE`` as separate axes, which is the
decomposition ``s05`` showed matters -- ``Q = E - mu*CDE`` with ``mu ~ 423``, so
89 % of the ladder's energy gain is cancelled by its entropy loss, and the
``E`` against ``CDE`` panel is where that cancellation is visible as a
correlation rather than as two numbers.

POINTS ARE LARGE (``s = 20``) and the colour is a single-hue light-to-dark ramp
in ``t``, as everywhere in this subproject: an ordered magnitude, not a category.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import selection
import style

import s05_ladder

#: Every one of these is defined for the wild type -- see the module docstring.
VARS = ["hamming_to_wt", "E_DCA", "CDE", "Q", "P_S"]

LABELS = {
    "hamming_to_wt": r"$d_H$ to wild type",
    "E_DCA": r"$E_{\mathrm{DCA}}$",
    "CDE": "CDE",
    "Q": r"$Q$",
    "P_S": style.P_S,
}

POINT_SIZE = 20.0
POINT_ALPHA = 0.38


def wild_type_values() -> dict:
    """The wild type on each of the five axes. It has a value on all of them."""
    from observables import context_dependent_entropy

    model = data.model()
    sequence = data.wild_type()
    cde = float(context_dependent_entropy(
        model, np.asarray([sequence]), beta=1.0)[0])
    energy = float(model.energy(sequence))
    return {
        "hamming_to_wt": 0.0,
        "E_DCA": energy,
        "CDE": cde,
        "Q": energy - config.mu() * cde,
        "P_S": float(np.exp(model.log_conditional(
            config.CRITIC_COL, sequence,
            model.alphabet.index(config.TARGET_RESIDUE), beta=1.0))),
    }


def main() -> None:
    style.apply()
    config.ensure_dirs()

    rungs = s05_ladder.ladder()
    if len(rungs) < 2:
        raise SystemExit("the ladder needs at least two rungs; see s05_ladder")

    shades = style.ramp(config.SET_COLOUR[s05_ladder.SET], len(rungs))
    wt = wild_type_values()
    n = len(VARS)

    fig, axes = plt.subplots(n, n, figsize=(3.05 * n, 3.05 * n))

    limits = {}
    for name in VARS:
        values = np.concatenate([r[name].dropna().to_numpy(float) for _, r in rungs])
        # The wild type is included in the P(S) range on purpose: it sits at
        # ~5e-4 while every design is above 0.9, and a reader should see that
        # the ladder has moved three orders of magnitude, not be shielded from it.
        values = np.append(values, wt[name])
        pad = 0.06 * (values.max() - values.min() or 1.0)
        limits[name] = (values.min() - pad, values.max() + pad)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                for (t, rung), shade in zip(rungs, shades):
                    ax.hist(rung[VARS[i]].dropna(), bins=30,
                            range=limits[VARS[i]], histtype="stepfilled",
                            color=shade, alpha=0.40, lw=0, zorder=2)
                    ax.hist(rung[VARS[i]].dropna(), bins=30,
                            range=limits[VARS[i]], histtype="step",
                            color=shade, lw=1.4, zorder=3)
                ax.axvline(wt[VARS[i]], color=style.C_WT, ls=":", lw=1.2, zorder=4)
                ax.set_xlim(*limits[VARS[i]])
                ax.set_yticks([])
            else:
                for (t, rung), shade in zip(rungs, shades):
                    ax.scatter(rung[VARS[j]], rung[VARS[i]], s=POINT_SIZE,
                               alpha=POINT_ALPHA, color=shade, linewidths=0,
                               zorder=2)
                    ax.scatter([rung[VARS[j]].mean()], [rung[VARS[i]].mean()],
                               s=130, marker="o", facecolor=shade,
                               edgecolor=style.SURFACE, linewidth=1.6, zorder=4)
                ax.scatter([wt[VARS[j]]], [wt[VARS[i]]], s=180, marker="X",
                           facecolor=style.SURFACE, edgecolor=style.C_WT,
                           linewidth=1.6, zorder=5)
                ax.set_xlim(*limits[VARS[j]])
                ax.set_ylim(*limits[VARS[i]])
            if i == n - 1:
                ax.set_xlabel(LABELS[VARS[j]])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(LABELS[VARS[i]])
            elif j != 0:
                ax.set_yticklabels([])
            style.tidy(ax)

    handles = [
        plt.Line2D([], [], color=shade, marker="o", lw=0, ms=9,
                   label=f"$t$ = {t:g}")
        for (t, _), shade in zip(rungs, shades)
    ]
    handles.append(plt.Line2D([], [], color=style.C_WT, marker="X", lw=0, ms=11,
                              markerfacecolor=style.SURFACE, label="wild type"))
    axes[0, n - 1].axis("off")
    axes[0, n - 1].legend(handles=handles, frameon=False, fontsize=11,
                          loc="upper right", bbox_to_anchor=(1.02, 1.0))

    fig.suptitle(
        rf"S6 — the $t$ ladder as a {n}×{n} corner "
        rf"({s05_ladder.SET}, $\beta$={s05_ladder.BETA:g}, "
        rf"$\lambda$={s05_ladder.LAM:g})", fontsize=14)
    style.caveat(
        fig,
        "1000 chains per rung, 60,000 steps, every rung past the S1 gate, read "
        r"through the $\beta=1$ serine rescoring.  large markers are rung means; "
        r"the wild type is on all five axes." "\n"
        r"the $E_{\mathrm{DCA}}$–CDE panel is the decomposition of $Q$: with "
        r"$\mu\approx423$ the two move together and 89 % of the ladder's energy "
        r"gain is cancelled in $Q$.",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    style.save(fig,
               f"s06_ladder_corner_{s05_ladder.SET}_beta{s05_ladder.BETA:g}_{n}x{n}",
               out_dir=config.OUT_DIR)


if __name__ == "__main__":
    main()
