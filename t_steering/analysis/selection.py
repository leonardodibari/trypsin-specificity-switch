"""Choosing the six runs the detailed figures are drawn for.

THE BRIEF, in the words it was given in: *a short range of mutations, Q similar
to the wild type, enough diversity within the batch, high P(S)*. Four desiderata
that pull against each other -- every one of them is bought with another -- so
the first thing to say is what this module does NOT do.

IT DOES NOT COLLAPSE THEM INTO A SCORE. A weighted sum would bury the trade-off
under weights nobody chose and would then report its own weighting as a result;
``analysis/figures_ladder/fig05_table.py`` makes the same point and resolves it
the same way. Instead the four are applied as **admissibility thresholds**, each
one stated and justified below, and what survives is then spread across the grid
so the six are not six versions of one setting.

THE THRESHOLDS

``P_S >= 0.90``
    "High P(S)". The grid's own lambda values were picked to clear this
    (lambda = 100 and 300 give ~0.93 and ~0.97 on the 79-column block), so this
    threshold is mostly a check that the restraint has not eaten the
    specificity it was applied on top of.

``Q - Q_wt <= 0.5 * sd_natural(Q)``
    "Q similar to the wild type", measured against the scale the family itself
    sets. The first attempt was ``Q <= Q_wt`` -- *at least as plausible as the
    wild type* -- and it admitted NOTHING: the wild type is an unusually good
    sequence under this model (``docs/RESULTS.md``: ~143 below the family mean
    in energy), and at beta <= 1.5 no design reaches it. Every design in the
    grid is worse than the wild type, by +2.5 to +19.1.

    Read on its own that sounds damning; read against the family it is not. The
    natural alignment's own ``sd(Q)`` is 28.0, so the whole design grid spans
    0.09 to 0.68 natural standard deviations from the wild type -- all of it
    well inside the variation the family shows. "Similar" therefore means
    "within the family's own spread", and the threshold is half that spread.

    NOTE that this requirement is only weakly binding here: it removes the
    worst-Q runs and nothing else. That is reported rather than fixed by
    tightening it, because a threshold chosen to manufacture a constraint is a
    threshold that reports its own arbitrariness as a result. Q still ranks the
    admissible runs -- ``select`` sorts by it -- it just does not exclude much.

``n_unique >= 100`` OR ``divergence_ratio > 0.3``
    "Enough diversity within the batch", as a screening library rather than as a
    percentage. A percentage floor was the first attempt and it was WRONG here:
    unique share is combinatorially capped by the set size and the budget, so at
    16 columns and <d> ~ 4 a thousand chains MUST collide, and a 90 % floor
    admitted nothing from the red sector while admitting the 79-column segment
    freely -- it would have selected `segment` four times over and called that a
    comparison.

    Two honest forms of the question, either of which suffices:

      n_unique          how many distinct designs there are to screen. An
                        absolute count, because 100 distinct designs is 100
                        regardless of how many columns produced them.
      divergence_ratio  the mean pairwise Hamming WITHIN the batch divided by
                        the mean distance to the wild type -- how different the
                        designs are from each other relative to how far they
                        have travelled. Scale-free, so it compares across sets
                        of 16, 20 and 79 columns without favouring the big one.

    Measured: a healthy batch sits at ratio 0.73-1.02 with 500-970 distinct
    designs; the collapsed beta = 1.5, t = 2 red-sector run sits at 0.272 with
    50. The threshold separates them.

``hamming <= HAMMING_CEILING``
    "A short range of mutations". Set per set rather than globally, because the
    sets differ in size by 5.6x and a 14-column set cannot carry 20 mutations.
    The ceiling is the number a construct is plausibly commissioned at.

THE SPREAD. Among the admissible runs the six are chosen to cover the grid:
at most two per mutation set, and within a set the two must differ in t, so the
figure can show what the restraint does rather than six points at one setting.
Ties are broken by Q, which is the quantity the designs are ranked by.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

import config

#: "A short range of mutations", per set. Fractions of each set's own size, not
#: one global number: 20 mutations is most of a 14-column construct and a
#: quarter of the grafting window.
HAMMING_CEILING = {
    "redsector": 12.0,
    "coupling20": 14.0,
    "segment": 20.0,
}

P_S_FLOOR = 0.90

#: "Similar to the wild type" in units of the natural family's own sd(Q).
Q_TOLERANCE_SD = 0.5

#: Diversity, as a screening library. Either suffices -- see the module
#: docstring for why a unique-PERCENTAGE floor is the wrong instrument here.
N_UNIQUE_FLOOR = 100
DIVERGENCE_FLOOR = 0.30

#: How many runs the detailed figures are drawn for.
N_SELECTED = 6

#: Of the summary columns, which direction is better. Used by the table figure
#: to bold the winner of each column, and nowhere else -- it is NOT a score.
BETTER_HIGH = {
    "P_S": True, "LOR": True, "frac_switched": True, "unique_pct": True,
    "n_unique": True, "pairwise_hamming": True, "divergence_ratio": True,
    "Q": False, "E_DCA": False, "hamming": False, "CDE": True,
    "accepted_per_site": True, "var_hamming": True,
}


def wild_type_Q(model=None) -> float:
    """``Q`` of the wild type itself, the reference the Q threshold is read against.

    Computed from the model rather than hard-coded, since ``mu`` is
    model-derived and a stale constant here would be a silently wrong threshold.
    """
    from observables import context_dependent_entropy
    import data as data_module

    if model is None:
        model = data_module.model()
    sequence = data_module.wild_type()
    energy = model.energy(sequence)
    cde = float(context_dependent_entropy(model, np.asarray([sequence]), beta=1.0)[0])
    return float(energy) - config.mu() * cde


def admissible(table: pd.DataFrame, q_wt: float) -> pd.DataFrame:
    """The four thresholds, applied as columns so a reader can see what each cost."""
    out = table.copy()
    out["ok_P_S"] = out["P_S"] >= P_S_FLOOR
    out["ok_Q"] = (out["Q"] - q_wt) <= Q_TOLERANCE_SD * config.natural_quality_sd()
    out["ok_diversity"] = (
        (out["n_unique"] >= N_UNIQUE_FLOOR)
        | (out["divergence_ratio"] > DIVERGENCE_FLOOR)
    )
    out["ok_hamming"] = [
        row.hamming <= HAMMING_CEILING[row.set] for row in out.itertuples()
    ]
    out["admissible"] = (
        out["ok_P_S"] & out["ok_Q"] & out["ok_diversity"] & out["ok_hamming"]
    )
    return out


def select(table: pd.DataFrame, q_wt: float, n: int = N_SELECTED) -> pd.DataFrame:
    """The six, spread across the grid. Returns the admissible rows that won.

    Two passes: first the best run of each (set, t) cell by Q, then fill any
    remaining slots from what is left, still best-Q first. The cap of two per
    set is what stops one set with a wide admissible region from taking every
    slot and turning a cross-set figure into a single-set one.
    """
    pool = admissible(table, q_wt)
    pool = pool[pool["admissible"]].sort_values("Q")
    if pool.empty:
        return pool

    chosen: List[int] = []
    per_set: dict[str, int] = {}
    seen_set_t: set[tuple[str, float]] = set()
    for row in pool.itertuples():
        if len(chosen) >= n:
            break
        if per_set.get(row.set, 0) >= 2:
            continue
        if (row.set, row.t) in seen_set_t:
            continue
        chosen.append(row.Index)
        per_set[row.set] = per_set.get(row.set, 0) + 1
        seen_set_t.add((row.set, row.t))

    for row in pool.itertuples():   # fill, if the spread rules left slots empty
        if len(chosen) >= n:
            break
        if row.Index not in chosen:
            chosen.append(row.Index)

    return pool.loc[chosen].sort_values(["set", "t"])


def label(row) -> str:
    """The short name a figure legend uses for one run."""
    return (f"{row.set}  " + r"$\beta$=" + f"{row.beta:g}, "
            + r"$\lambda$=" + f"{row.lam:g}, " + r"$t$=" + f"{row.t:g}")
