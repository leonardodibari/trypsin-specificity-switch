"""H4 -- is t just a reparametrised temperature?

H2 shows that the restraint and cooling both pull a design back towards the wild
type, and that they pay for it differently: cooling collapses the library and the
acceptance rate, the restraint does not. That is a statement about *summary
statistics* -- Q, P(S), <d>, unique designs, accepted moves. It leaves the
sharper question open, and it is the question a referee will ask first:

    at the same mutation budget, do the two protocols produce the SAME designs?

If they do -- if the only effect of t is to slide a design along the one path the
temperature already traces -- then t is a reparametrisation of beta, the term is
cosmetic, and the project must say so plainly. If they do not, then t and beta
select genuinely different sequences out of the same model at the same cost, and
*which* columns they disagree about is the result.

The test
--------
Summary statistics cannot answer this; two very different ensembles can share a
mean. So this figure goes down to the **per-column mutation spectrum**: for each
of the 79 columns of the grafting window, the frequency profile over the 21
states. Two ensembles are "the same design" exactly when those 79 profiles agree.

Matching. A spectrum comparison is only meaningful at a matched mutation budget,
so every restraint run (lambda, t) is paired with a cooled run of the SAME
lambda-tilde = lambda/beta -- cooling's own ladder holds lambda-tilde fixed down
a rung, so this is the cooled run that is chasing the same objective -- choosing
the rung whose mean Hamming <d> is closest. We do NOT interpolate between rungs.
Interpolation would mean mixing two cooled ensembles, and a mixture of two
cooled ensembles is not a cooled ensemble: it carries the spread BETWEEN the two
temperatures as if it were spread within one, which inflates exactly the entropy
panel (d) is built to measure. So the match is discrete and the residual is
reported rather than hidden: every pair carries its |D<d>| in the figure, in the
CSV and in the caption. Two tolerances:

  * matched  |D<d>| <= 0.5 mutations  -- the comparison the verdict rests on;
  * loose    |D<d>| <= 1.1 mutations  -- kept only for beta = 1.5, because that
    is the one cooled rung that is still an equilibrium sample (see below), and
    a comparison that exists only against frozen chains is worth less.

Everything else is dropped, and the dropped runs are listed with their best
achievable residual in ``h04_degeneracy_matching.csv``. The largest single fact
in that list: **cooling cannot reach the restraint's tightest budgets at all**.
Even at beta = 10 the cooled ladder floors at <d> ~ 9, while t = 2 reaches 7.2,
so the t = 2 runs have no cooled counterpart to be compared with.

Divergence. Jensen-Shannon, in bits, is the primary measure, for two reasons.
(i) It is finite. With 1000 chains most of the 21 states are unobserved in any
given column, so KL has zeros in the denominator and is either infinite or an
artefact of whatever pseudocount was chosen. (ii) It is symmetric and bounded by
1 bit, so a per-column number can be read as "what fraction of a complete
disagreement is this" without a direction convention. KL is reported in both
directions in the CSV, under a Laplace pseudocount, for anyone who wants it --
but it is the pseudocount's number as much as the data's, which is why it is not
what the figure plots.

The null. Two independent runs of the SAME protocol do not give JS = 0 at 1000
chains, so a raw JS means nothing without a floor. Two floors are used:
  * a cluster permutation null per pair -- pool the two ensembles, split them at
    random into two groups of the original sizes with all copies of a repeated
    sequence kept together, recompute JS. Keeping duplicates together matters:
    a frozen chain's ensemble is 1000 copies of 60 distinct designs, and a null
    that broke those clusters apart would pretend the cold run had 1000
    independent samples and would make everything look significant.
  * the t = 0 pairs, which are an independent repeat of the same warm protocol
    (restraint block at t = 0 against the ladder's own beta = 1 run) and so
    measure the real run-to-run floor directly.

The panels
----------
  (a) how each pair was matched: both ladders on a <d> axis, one row per
      lambda-tilde, the accepted pairs joined and labelled with their residual,
      and the budget cooling cannot reach shaded.
  (b) the answer: per-column JS for every accepted pair, as a heat map. The
      t = 0 control rows are the floor; if the test rows look like them, t is a
      reparametrised temperature.
  (c) how they differ: at the best-matched pair's most divergent columns, the
      residue frequencies side by side -- which residue each protocol prefers.
  (d) how much each protocol explores: effective number of residues per column,
      restraint against cooling, one point per column.

The caveat that does not get softened
-------------------------------------
``data.load_ladder`` flags every cooled run at beta >= 2 as ``stuck``: at those
temperatures acceptance has collapsed and the chain has stopped moving rather
than settled. A stuck run is not an equilibrium sample of anything, so it is not
a sample of the cold Potts model either. And it is precisely those runs that are
tight enough to match the restraint: of the three strictly matched test pairs,
all three are against a stuck chain. Every orange mark drawn from one is hatched
and every such row is flagged, in the figure and in the CSV.

That cuts both ways, and the honest reading has to carry both branches. If the
spectra differ, the cause is either (i) t and beta genuinely select different
designs, or (ii) the cooled chain never reached the cold equilibrium it was
supposed to represent. The figure cannot separate them -- nothing can, until a
cold run is sampled to convergence. What it CAN say is the thing the project
actually needs: at a matched mutation budget, cooling as the project is able to
run it does not reproduce the restraint's designs. Whether that is temperature
or merely freezing, t is not a knob that could have been replaced by beta.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

import config
import data
import h02_pareto as h02
import style

BLOCK = h02.BLOCK

#: Keep h02's mapping: the restraint is blue and round, cooling is orange and
#: square. A reader who has seen h02 must not have to relearn the colours.
RESTRAINT_COLOUR = h02.RESTRAINT_COLOUR
COOLING_COLOUR = h02.COOLING_COLOUR
#: The divergence between the two belongs to neither, so it gets its own hue.
DIVERGENCE_COLOUR = style.VIOLET

#: Mean-Hamming tolerances, in mutations. The strict one is what the verdict
#: rests on; the loose one exists only to let the single non-frozen cooled rung
#: (beta = 1.5) into the comparison at all.
TOL_MATCHED = 0.5
TOL_LOOSE = 1.1
#: The one cooled rung that data.load_ladder does NOT flag as stuck above beta=1.
WARM_BETA = 1.5

#: Permutation replicates for the per-pair null. 200 is enough for a 95th
#: percentile that does not move visibly between seeds, and the whole figure
#: still builds in under a minute.
NULL_REPS = 200
NULL_SEED = 20240517

#: Residue frequency a state must reach in one of the two ensembles before it is
#: drawn in panel (c). Below this the bars are noise and the axis is unreadable.
PANEL_C_MIN_FREQ = 0.10
PANEL_C_N_COLUMNS = 6


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------

def _alphabet():
    from encoding import Alphabet

    return Alphabet()


def allowed_columns() -> np.ndarray:
    """The 79 columns the sampler was allowed to move, 0-based.

    EVERY spectrum below is restricted to these. The ensembles are full 265-column
    alignments and the other 186 columns are frozen at the wild type in both
    protocols, so including them would divide every divergence by 265 instead of
    79 and shrink the whole figure by a factor of 3.4 for no reason -- the same
    wrong-denominator mistake this repository has already made once.
    """
    return np.arange(*config.GRAFTING_SEGMENT)


@lru_cache(maxsize=None)
def _encoded(path: str) -> np.ndarray:
    """``(n_chains, 79)`` token indices of one ensemble's final sequences."""
    from encoding import read_sequences

    alphabet = _alphabet()
    sequences = [s for _, s in read_sequences(Path(path))]
    if not sequences:
        raise ValueError(f"{path} holds no sequences")
    matrix = np.array([alphabet.encode(s) for s in sequences])
    return matrix[:, allowed_columns()]


def _profile(encoded: np.ndarray) -> np.ndarray:
    """``(79, 21)`` per-column state frequencies of an encoded ensemble."""
    q = _alphabet().q
    counts = np.stack(
        [np.bincount(encoded[:, j], minlength=q) for j in range(encoded.shape[1])]
    ).astype(float)
    return counts / counts.sum(axis=1, keepdims=True)


def _js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-column Jensen-Shannon divergence in BITS, so the bound is 1.

    Written with ``np.where`` rather than a mask because 0*log0 = 0 is the right
    convention here and the alternative is 79 small loops.
    """
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(p > 0, p * np.log2(p / m), 0.0)
        right = np.where(q > 0, q * np.log2(q / m), 0.0)
    return 0.5 * left.sum(axis=1) + 0.5 * right.sum(axis=1)


def _kl(p: np.ndarray, q: np.ndarray, *, pseudo: float = 1.0) -> np.ndarray:
    """Per-column KL(p || q) in bits, under a Laplace pseudocount.

    Reported, not plotted. Without the pseudocount this is infinite for almost
    every column -- any state the restraint visits and the cooled run never does
    puts a zero in the denominator -- and WITH it the number depends on a
    constant nobody calibrated. That dependence is the reason the figure is
    built on JS instead.
    """
    n = 1000.0  # ensembles are 1000 chains throughout; only sets the pseudo scale
    p = (p * n + pseudo) / (n + pseudo * p.shape[1])
    q = (q * n + pseudo) / (n + pseudo * q.shape[1])
    return (p * np.log2(p / q)).sum(axis=1)


def _n_eff(profile: np.ndarray) -> np.ndarray:
    """Effective number of residues per column, ``exp(H)`` with H in nats.

    Plain entropy is in the wrong units for a reader: ``n_eff`` says "this column
    is effectively 3.4 residues wide", which is what panel (d) is asking.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(profile > 0, profile * np.log(profile), 0.0).sum(axis=1)
    return np.exp(h)


def _null_js(a: np.ndarray, b: np.ndarray, *, reps: int = NULL_REPS,
             seed: int = NULL_SEED) -> np.ndarray:
    """``(reps, 79)`` JS under the hypothesis that the two ensembles agree.

    Cluster permutation: pool the chains, then shuffle at the level of DISTINCT
    sequences, so all 140 copies of a frozen design land on the same side. That
    is what makes this null honest for a cold run. A plain chain-level shuffle
    would split those copies evenly, manufacture 1000 independent samples out of
    60, and drive the null floor to roughly nothing -- every column would then
    come out "significant" whatever the data said.
    """
    rng = np.random.default_rng(seed)
    pool = np.vstack([a, b])
    n_a = len(a)
    _, inverse, counts = np.unique(
        pool, axis=0, return_inverse=True, return_counts=True
    )
    n_clusters = len(counts)
    out = np.empty((reps, pool.shape[1]))
    for rep in range(reps):
        order = rng.permutation(n_clusters)
        # take whole clusters until the first group is at least the size of `a`
        cut = int(np.searchsorted(np.cumsum(counts[order]), n_a)) + 1
        chosen = np.zeros(n_clusters, dtype=bool)
        chosen[order[:cut]] = True
        mask = chosen[inverse]
        out[rep] = _js(_profile(pool[mask]), _profile(pool[~mask]))
    return out


# ---------------------------------------------------------------------------
# the ensembles, and the pairing between them
# ---------------------------------------------------------------------------

def _restraint_fasta(lam: float, t: float) -> Path:
    """The serine-stamped ensemble of one restraint run.

    ``_serine`` and not the raw block: the critic column is stamped with S and
    everything is rescored at beta = 1, which is the only form in which a cooled
    run and a warm one are comparable at all (RESULTS.md section 5). Reading the
    raw tree here would compare a beta = 3 ensemble's own scores against a
    beta = 1 ensemble's, which is not a comparison.
    """
    return (
        config.RESULTS_ROOT
        / f"{BLOCK}_serine"
        / h02._run_name(lam, t)
        / "final_sequences.fasta"
    )


def _cooling_fasta(lam: float, beta: float) -> Path:
    directory = (
        config.LADDER_GRAFTING if beta == 1.0 else config.LADDER_COOLING[beta]
    )
    return Path(directory) / f"lam{lam:g}" / "final_sequences.fasta"


def match_table(restraint: pd.DataFrame, cooling: pd.DataFrame) -> pd.DataFrame:
    """Pair every restraint run with a cooled run at the same objective.

    Pairing is WITHIN lambda-tilde = lambda/beta, not across the whole cooled
    set. Matching on <d> alone would happily pair the lambda = 10, t = 0.5 run
    (P(S) = 0.73) with a lambda-tilde = 100 cooled run (P(S) = 0.93) because
    their mean distances agree to 0.3 -- same mutation budget, completely
    different design goal, and any spectral difference would then be the
    specificity pressure rather than the protocol. Holding lambda-tilde fixed
    keeps P(S) matched to within about 0.03 as a by-product.

    Two candidates are recorded per restraint run: the nearest cooled rung of
    any temperature (``cold_*``), and the nearest rung that is still an
    equilibrium sample (``warm_*``, beta <= 1.5). The second exists because the
    first is almost always a stuck chain.
    """
    rows = []
    for row in restraint.itertuples():
        group = cooling[cooling["lam_tilde"] == row.lam]
        if group.empty:
            rows.append(dict(lam=row.lam, t=row.t, d_restraint=row.hamming,
                             P_S_restraint=row.P_S, tier="no cooled run"))
            continue

        def nearest(sub: pd.DataFrame):
            if sub.empty:
                return None
            best = sub.iloc[(sub["hamming"] - row.hamming).abs().argsort().iloc[0]]
            return best

        cold = nearest(group)
        warm = nearest(group[group["beta"] <= WARM_BETA])
        entry = dict(
            lam=row.lam, t=row.t,
            d_restraint=row.hamming, P_S_restraint=row.P_S,
            cold_beta=cold["beta"], cold_lam=cold["lam"],
            d_cold=cold["hamming"], P_S_cold=cold["P_S"],
            resid_cold=abs(cold["hamming"] - row.hamming),
            cold_stuck=bool(cold["stuck"]),
        )
        if warm is not None:
            entry.update(
                warm_beta=warm["beta"], warm_lam=warm["lam"],
                d_warm=warm["hamming"], P_S_warm=warm["P_S"],
                resid_warm=abs(warm["hamming"] - row.hamming),
                warm_stuck=bool(warm["stuck"]),
            )
        rows.append(entry)
    table = pd.DataFrame(rows)

    # The cooled floor: the tightest budget cooling reaches in this group. Runs
    # below it are not "badly matched", they are unreachable, and the two cases
    # must not be reported as the same kind of failure.
    floor = cooling.groupby("lam_tilde")["hamming"].min()
    table["cooling_floor"] = table["lam"].map(floor)
    table["below_cooling_floor"] = table["d_restraint"] < table["cooling_floor"]
    return table


def accepted_pairs(table: pd.DataFrame) -> List[dict]:
    """The pairs the figure is allowed to draw, in plotting order.

    A pair is kept when its residual is inside TOL_MATCHED, or when it is the
    beta = 1.5 competitor and inside TOL_LOOSE. The warm pair is only kept where
    it is genuinely a different run from the cold one, so nothing is drawn twice.
    """
    pairs = []
    for row in table.itertuples():
        if getattr(row, "cold_beta", None) is None or pd.isna(getattr(row, "cold_beta", np.nan)):
            continue
        if row.resid_cold <= TOL_MATCHED:
            pairs.append(dict(
                lam=row.lam, t=row.t, beta=row.cold_beta, cool_lam=row.cold_lam,
                d_restraint=row.d_restraint, d_cooling=row.d_cold,
                P_S_restraint=row.P_S_restraint, P_S_cooling=row.P_S_cold,
                residual=row.resid_cold, stuck=row.cold_stuck,
                tier="matched",
            ))
        warm_beta = getattr(row, "warm_beta", np.nan)
        if (
            not pd.isna(warm_beta)
            and warm_beta != 1.0                       # beta = 1 is the control
            and row.resid_warm <= TOL_LOOSE
            and not (row.resid_cold <= TOL_MATCHED and warm_beta == row.cold_beta)
        ):
            pairs.append(dict(
                lam=row.lam, t=row.t, beta=warm_beta, cool_lam=row.warm_lam,
                d_restraint=row.d_restraint, d_cooling=row.d_warm,
                P_S_restraint=row.P_S_restraint, P_S_cooling=row.P_S_warm,
                residual=row.resid_warm, stuck=row.warm_stuck,
                tier="loose (warm competitor)",
            ))
    # controls first (t = 0, the floor), then the real tests, then the loose set
    order = {"matched": 0, "loose (warm competitor)": 1}
    pairs.sort(key=lambda p: (p["t"] > 0, order[p["tier"]], p["lam"], p["t"]))
    return pairs


def spectra(pairs: List[dict]) -> Tuple[pd.DataFrame, Dict[int, dict]]:
    """Per-column divergence of every accepted pair, plus the raw profiles.

    Returns a long table (one row per pair per column, which is what the CSV
    wants) and a dict of the arrays panels (c) and (d) need back.
    """
    columns = allowed_columns()
    rows, cache = [], {}
    for index, pair in enumerate(pairs):
        a = _encoded(str(_restraint_fasta(pair["lam"], pair["t"])))
        b = _encoded(str(_cooling_fasta(pair["cool_lam"], pair["beta"])))
        p, q = _profile(a), _profile(b)
        js = _js(p, q)
        null = _null_js(a, b)
        cache[index] = dict(
            pair=pair, profile_restraint=p, profile_cooling=q, js=js,
            null_95=np.percentile(null, 95, axis=0),
            null_mean_95=float(np.percentile(null.mean(axis=1), 95)),
            n_eff_restraint=_n_eff(p), n_eff_cooling=_n_eff(q),
        )
        for j, column in enumerate(columns):
            rows.append(dict(
                pair=index, lam=pair["lam"], t=pair["t"], beta=pair["beta"],
                tier=pair["tier"], cooled_stuck=pair["stuck"],
                d_restraint=pair["d_restraint"], d_cooling=pair["d_cooling"],
                residual_d=pair["residual"],
                P_S_restraint=pair["P_S_restraint"],
                P_S_cooling=pair["P_S_cooling"],
                msa_column=int(column),
                js_bits=float(js[j]),
                null95_js_bits=float(cache[index]["null_95"][j]),
                kl_restraint_to_cooling=float(_kl(p, q)[j]),
                kl_cooling_to_restraint=float(_kl(q, p)[j]),
                n_eff_restraint=float(cache[index]["n_eff_restraint"][j]),
                n_eff_cooling=float(cache[index]["n_eff_cooling"][j]),
                is_critic_column=bool(column == config.CRITIC_COL),
            ))
    return pd.DataFrame(rows), cache


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def _sequential(colour: str):
    """A white -> hue -> ink ramp, built out of style.mix so the heat map does
    not introduce a colormap the rest of the figure set has never seen."""
    stops = [style.SURFACE, style.mix(style.SURFACE, colour, 0.45), colour,
             style.mix(colour, style.INK, 0.55)]
    return LinearSegmentedColormap.from_list("h04", stops)


def _panel_matching(ax, restraint, cooling, pairs) -> None:
    """(a) Where every pair came from, and what it cost to make it."""
    groups = sorted(restraint["lam"].unique())
    y_of = {lam: i for i, lam in enumerate(groups)}
    offset = 0.17

    for lam in groups:
        y = y_of[lam]
        part = restraint[restraint["lam"] == lam].sort_values("hamming")
        cool = cooling[cooling["lam_tilde"] == lam].sort_values("hamming")

        # The budget the restraint reaches and cooling cannot: not a bad match,
        # an absent one, and the figure must not let the two look alike.
        if not cool.empty and part["hamming"].min() < cool["hamming"].min():
            ax.add_patch(plt.Rectangle(
                (part["hamming"].min(), y - 0.33),
                cool["hamming"].min() - part["hamming"].min(), 0.66,
                facecolor=style.FAINT, edgecolor="none", zorder=0,
            ))

        ax.plot(part["hamming"], np.full(len(part), y + offset), color=style.FAINT,
                lw=1.2, zorder=1)
        ax.scatter(part["hamming"], np.full(len(part), y + offset), s=62,
                   color=RESTRAINT_COLOUR, marker="o", edgecolor="white",
                   linewidth=0.8, zorder=3)
        if not cool.empty:
            ax.plot(cool["hamming"], np.full(len(cool), y - offset),
                    color=style.FAINT, lw=1.2, zorder=1)
            frozen = cool["stuck"].to_numpy()
            ax.scatter(cool.loc[~frozen, "hamming"],
                       np.full(int((~frozen).sum()), y - offset), s=62,
                       color=COOLING_COLOUR, marker="s", edgecolor="white",
                       linewidth=0.8, zorder=3)
            ax.scatter(cool.loc[frozen, "hamming"],
                       np.full(int(frozen.sum()), y - offset), s=62,
                       color=COOLING_COLOUR, marker="s", edgecolor=style.RED,
                       linewidth=1.6, zorder=3)

    # The accepted pairs, with the residual written on each. A reader must be
    # able to see that a "matched" pair really is matched without the CSV.
    # A matched and a loose pair often share the same restraint end, so the two
    # labels are parked at different fractions along the connector; at the
    # midpoint they overprinted each other.
    for pair in pairs:
        y = y_of[pair["lam"]]
        solid = pair["tier"] == "matched"
        ax.plot([pair["d_restraint"], pair["d_cooling"]],
                [y + offset, y - offset],
                color=style.INK if solid else style.MUTED,
                lw=1.6 if solid else 1.1,
                ls="-" if solid else (0, (3, 2)), zorder=2, alpha=0.85)
        # Matched labels ride the connector; loose ones are parked BELOW the
        # cooled row, where nothing else is drawn -- on the connector they
        # landed on top of the cooled markers, which cluster tightly.
        frac = 0.34 if solid else 0.5
        y_label = (y + offset - frac * 2 * offset) if solid else (y - offset - 0.26)
        ax.text(pair["d_restraint"] + frac * (pair["d_cooling"] - pair["d_restraint"]),
                y_label,
                f"{pair['residual']:.2f}", fontsize=9.5, ha="center",
                va="center", color=style.INK if solid else style.MUTED,
                bbox=dict(boxstyle="round,pad=0.12", fc=style.SURFACE,
                          ec="none", alpha=0.92), zorder=4)

    ax.set_yticks(list(y_of.values()))
    ax.set_yticklabels([rf"$\lambda=\tilde\lambda={g:g}$" for g in groups])
    # Extra room below the last row: the legend needs a band of its own, and at
    # four rows there is no empty corner left inside the data.
    ax.set_ylim(-0.6, len(groups) - 1 + 0.95)
    ax.invert_yaxis()
    ax.set_xlabel(style.LABELS["hamming_to_wt"])
    ax.set_title("every pair, and the $\\langle d\\rangle$ it is matched at",
                 fontsize=14, color=style.MUTED, pad=8)
    style.panel_letter(ax, "a")
    style.tidy(ax, grid="x")

    ax.legend(handles=[
        plt.Line2D([], [], color=RESTRAINT_COLOUR, marker="o", ls="-", lw=1.2,
                   markersize=7, label=r"restraint, $t$ = 0 $\to$ 2"),
        plt.Line2D([], [], color=COOLING_COLOUR, marker="s", ls="-", lw=1.2,
                   markersize=7, label=r"cooling, $\beta$ = 1 $\to$ 10"),
        plt.Line2D([], [], color=COOLING_COLOUR, marker="s", ls="none",
                   markersize=7, markeredgecolor=style.RED, markeredgewidth=1.6,
                   label=r"stuck ($\beta\geq2$), not an equilibrium sample"),
        Patch(facecolor=style.FAINT, label="beyond cooling's reach"),
    ], loc="lower left", fontsize=10.0, handletextpad=0.6, borderaxespad=0.2,
       ncol=2, columnspacing=1.0)


def _panel_heatmap(ax, cax, pairs, cache) -> None:
    """(b) The answer: per-column JS, one row per pair."""
    columns = allowed_columns()
    matrix = np.stack([cache[i]["js"] for i in range(len(pairs))])
    cmap = _sequential(DIVERGENCE_COLOUR)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0,
                      vmax=float(matrix.max()), interpolation="nearest",
                      extent=(columns[0] - 0.5, columns[-1] + 0.5,
                              len(pairs) - 0.5, -0.5))

    labels = []
    for index, pair in enumerate(pairs):
        kind = "control" if pair["t"] == 0 else "test"
        label = (rf"$\lambda${pair['lam']:g} $t${pair['t']:g} vs "
                 rf"$\beta${pair['beta']:g}  ({kind})")
        if pair["stuck"]:
            # The stuck rows are named stuck on the axis itself and coloured
            # red, so the figure cannot be read -- or cropped -- without the
            # warning travelling with it.
            label += "  STUCK"
        labels.append(label)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels, fontsize=10.5)
    for tick, pair in zip(ax.get_yticklabels(), pairs):
        tick.set_color(style.RED if pair["stuck"]
                       else (style.MUTED if pair["t"] == 0 else style.INK))

    # separate the control block from the test block
    n_control = sum(1 for p in pairs if p["t"] == 0)
    ax.axhline(n_control - 0.5, color=style.INK, lw=1.4)
    ax.axvline(config.CRITIC_COL, color=style.RED, lw=1.0, ls=(0, (2, 2)),
               alpha=0.7)
    ax.text(config.CRITIC_COL, -0.72, f"critic {config.CRITIC_COL}", fontsize=9.5,
            color=style.RED, ha="center", va="bottom")

    ax.set_xlabel("MSA column (grafting window, 79 allowed columns)")
    ax.set_xlim(columns[0] - 0.5, columns[-1] + 0.5)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    style.panel_letter(ax, "b", dx=-0.265, dy=1.26)

    bar = plt.colorbar(image, cax=cax)
    bar.set_label("JS divergence (bits)", fontsize=11.5)
    bar.ax.tick_params(labelsize=10)

    control = matrix[:n_control].mean()
    matched = [i for i, p in enumerate(pairs)
               if p["t"] > 0 and p["tier"] == "matched"]
    test = matrix[matched].mean() if matched else float("nan")
    ax.set_title(
        f"control rows average {control:.4f} bits; the strictly matched test rows "
        f"average {test:.4f} -- {test / control:.0f}x the run-to-run floor",
        fontsize=13, color=style.MUTED, pad=20,
    )


def _panel_composition(ax, cache, pairs) -> None:
    """(c) The columns that differ most, and what each protocol puts there."""
    alphabet = _alphabet()
    columns = allowed_columns()
    # Use the best-matched TEST pair -- smallest residual among t > 0 -- because
    # this panel is read as "these are the actual residues", and the residue at
    # a column is only comparable where the budget really is the same.
    tests = [i for i, p in enumerate(pairs) if p["t"] > 0 and p["tier"] == "matched"]
    best = min(tests, key=lambda i: pairs[i]["residual"])
    entry = cache[best]
    pair = entry["pair"]

    order = np.argsort(-entry["js"])[:PANEL_C_N_COLUMNS]
    order = sorted(order, key=lambda j: -entry["js"][j])

    xticks, xlabels, left_edges, boundaries = [], [], [], []
    x = 0.0
    for rank, j in enumerate(order):
        p, q = entry["profile_restraint"][j], entry["profile_cooling"][j]
        states = [s for s in range(alphabet.q)
                  if max(p[s], q[s]) >= PANEL_C_MIN_FREQ]
        states.sort(key=lambda s: -max(p[s], q[s]))
        for state in states:
            ax.bar(x - 0.19, p[state], width=0.36, color=RESTRAINT_COLOUR,
                   edgecolor="white", linewidth=0.6, zorder=3)
            ax.bar(x + 0.19, q[state], width=0.36, color=COOLING_COLOUR,
                   edgecolor=style.RED if pair["stuck"] else "white",
                   linewidth=1.0 if pair["stuck"] else 0.6,
                   hatch="///" if pair["stuck"] else None, zorder=3)
            xticks.append(x)
            xlabels.append(alphabet.tokens[state])
            x += 1.0
        boundaries.append(x - 0.5)
        left_edges.append((x - len(states), x - 1, columns[j], entry["js"][j]))
        x += 0.6

    for start, end, column, js in left_edges:
        centre = 0.5 * (start + end)
        ax.text(centre, 1.055, f"col {column}", fontsize=12.5, ha="center",
                va="bottom", color=style.INK, fontweight="bold")
        ax.text(centre, 0.985, f"JS {js:.2f} bits", fontsize=10.5, ha="center",
                va="bottom", color=style.MUTED)
    for boundary in boundaries[:-1]:
        ax.axvline(boundary, color=style.FAINT, lw=1.0, zorder=0)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_xlim(-0.9, x - 1.1)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("frequency in the ensemble")
    ax.set_title(
        rf"the six most divergent columns of the best-matched pair "
        rf"($\lambda$={pair['lam']:g}, $t$={pair['t']:g} vs $\beta$={pair['beta']:g}; "
        rf"$\langle d\rangle$ {pair['d_restraint']:.2f} vs {pair['d_cooling']:.2f}, "
        rf"residual {pair['residual']:.2f})",
        fontsize=13, color=style.MUTED, pad=30,
    )
    style.panel_letter(ax, "c", dx=-0.062, dy=1.30)
    style.tidy(ax, grid="y")
    # Below the axis: every column header already occupies the top of the panel
    # and the tallest bar in it reaches 0.96, so there is no interior corner the
    # legend could sit in without covering data.
    ax.legend(handles=[
        Patch(facecolor=RESTRAINT_COLOUR, label="restraint"),
        Patch(facecolor=COOLING_COLOUR, hatch="///", edgecolor=style.RED,
              label=r"cooling -- STUCK chain, not an equilibrium sample"
                    if pair["stuck"] else "cooling"),
    ], loc="upper right", bbox_to_anchor=(1.0, -0.11), fontsize=11, ncol=2,
       handlelength=1.6)
    return best


def _panel_exploration(ax, pairs, cache) -> None:
    """(d) How wide each protocol leaves each column."""
    tests = [i for i, p in enumerate(pairs) if p["t"] > 0]
    shades = dict(zip(tests, style.ramp(DIVERGENCE_COLOUR, max(len(tests), 2))))
    limit = 1.0
    for index in tests:
        entry = cache[index]
        pair = entry["pair"]
        limit = max(limit, entry["n_eff_restraint"].max(),
                    entry["n_eff_cooling"].max())
        ax.scatter(
            entry["n_eff_restraint"], entry["n_eff_cooling"], s=34,
            facecolor=shades[index] if pair["tier"] == "matched" else "none",
            edgecolor=shades[index], linewidth=1.1, alpha=0.85, zorder=3,
            label=(rf"$\lambda${pair['lam']:g} $t${pair['t']:g} vs "
                   rf"$\beta${pair['beta']:g}"
                   + ("" if pair["tier"] == "matched" else "  (loose)")),
        )
    top = limit * 1.06
    ax.plot([1, top], [1, top], color=style.INK, lw=1.2, ls=(0, (4, 3)), zorder=2,
            label="equal exploration")
    # The text goes bottom-right, which is where the cloud is NOT: almost every
    # column sits below the diagonal, and that asymmetry is the panel's message.
    ax.text(0.975, 0.055, "cooling explores LESS", transform=ax.transAxes,
            fontsize=11.5, color=style.MUTED, ha="right", va="bottom")
    ax.set_xlim(0.9, top)
    ax.set_ylim(0.9, top)
    ax.set_xlabel("effective residues per column, restraint")
    ax.set_ylabel("effective residues per column, cooling")
    ax.set_title("one point per allowed column", fontsize=14, color=style.MUTED,
                 pad=8)
    style.panel_letter(ax, "d")
    style.tidy(ax)
    # Upper left is the half of the plane the data mostly avoids, so the legend
    # costs nothing there.
    ax.legend(loc="upper left", fontsize=10.5, borderaxespad=0.5,
              handletextpad=0.5)


# ---------------------------------------------------------------------------

def main() -> None:
    style.apply()
    config.ensure_dirs()

    # Both tables come from h02 rather than being rebuilt: they already go
    # through data.designs (so unsaturated runs are gone), already read the
    # serine tree, and already carry lam_tilde and the stuck flag. A second copy
    # is a second thing to keep in step.
    restraint = h02.restraint_table()
    cooling = h02.cooling_table()
    # lambda-tilde = 0 is the unrestrained, unspecific control run; it has no
    # restraint counterpart and would only widen the axes.
    cooling = cooling[cooling["lam_tilde"] > 0]

    matching = match_table(restraint, cooling)
    pairs = accepted_pairs(matching)
    print(f"  {len(pairs)} pair(s) accepted out of {len(matching)} restraint runs")
    for pair in pairs:
        print(f"    lam={pair['lam']:>5g} t={pair['t']:<4g} vs beta={pair['beta']:<4g}"
              f"  <d> {pair['d_restraint']:6.3f} / {pair['d_cooling']:6.3f}"
              f"  residual {pair['residual']:.3f}  [{pair['tier']}]"
              f"{'  STUCK' if pair['stuck'] else ''}")
    dropped = matching[~matching.apply(
        lambda r: any(p["lam"] == r["lam"] and p["t"] == r["t"] for p in pairs),
        axis=1)]
    for row in dropped.itertuples():
        why = ("below the cooling floor" if row.below_cooling_floor
               else "no cooled rung within tolerance")
        print(f"    DROPPED lam={row.lam:>5g} t={row.t:<4g}  <d> {row.d_restraint:6.3f}"
              f"  best residual {getattr(row, 'resid_cold', float('nan')):.3f}  ({why})")

    long, cache = spectra(pairs)

    fig = plt.figure(figsize=(15.0, 15.8))
    grid = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 0.80, 0.92], width_ratios=[1.0, 1.0],
        hspace=0.60, wspace=0.26,
        left=0.085, right=0.965, top=0.868, bottom=0.115,
    )

    _panel_matching(fig.add_subplot(grid[0, 0]), restraint, cooling, pairs)
    _panel_exploration(fig.add_subplot(grid[0, 1]), pairs, cache)

    heat = fig.add_subplot(grid[1, :])
    # The heat map carries ten long row labels and a colour bar, neither of
    # which fits in the gridspec's margins, so this one panel gets its own
    # geometry: pulled in from the left for the labels, stopped short of the
    # right so the bar and its ticks have somewhere to go.
    box = heat.get_position()
    heat.set_position([0.215, box.y0, 0.695, box.height])
    cax = fig.add_axes([0.924, box.y0, 0.011, box.height])
    _panel_heatmap(heat, cax, pairs, cache)

    best = _panel_composition(fig.add_subplot(grid[2, :]), cache, pairs)

    # The headline is the verdict, so it is computed, never typed by hand.
    n_control = sum(1 for p in pairs if p["t"] == 0)
    control_js = float(np.mean([cache[i]["js"].mean() for i in range(n_control)]))
    test_indices = [i for i, p in enumerate(pairs)
                    if p["t"] > 0 and p["tier"] == "matched"]
    test_js = float(np.mean([cache[i]["js"].mean() for i in test_indices]))
    n_sig = int(np.mean([
        (cache[i]["js"] > cache[i]["null_95"]).sum() for i in test_indices
    ]))
    answer = ("No -- at a matched mutation budget "
              r"$t$ and $\beta$ select different designs"
              if test_js > 3 * control_js else
              r"Yes -- $t$ and $\beta$ reach the same designs")
    fig.suptitle("Is $t$ a reparametrised temperature?", y=0.978, fontsize=19.5)
    fig.text(0.5, 0.948, answer, ha="center", va="bottom", fontsize=16,
             color=style.INK)
    fig.text(0.5, 0.900,
             f"mean per-column JS {test_js:.4f} bits across the strictly matched "
             f"pairs, against a {control_js:.4f}-bit run-to-run floor"
             f"\n{n_sig} of {len(allowed_columns())} columns differ beyond their own "
             "permutation null",
             ha="center", va="bottom", fontsize=13.5, color=style.MUTED,
             linespacing=1.5)

    # Hard-wrapped rather than left to the renderer: at 15 inches a single line
    # of this runs off both ends of the page, and a caveat nobody can read is
    # the same as no caveat.
    style.caveat(
        fig,
        "grafting window, 79 columns; every spectrum is normalised over those "
        "columns only, never over all 265."
        r"  both sets read from the serine trees, rescored at $\beta=1$,"
        "\nthe only form in which a cooled run and a warm one are comparable."
        "\n"
        r"EVERY strictly matched cooled run in this figure is flagged STUCK "
        r"($\beta\geq2$): acceptance has collapsed and the chain has stopped "
        "moving rather than settled,"
        "\nso those rows compare the restraint against a NON-EQUILIBRIUM "
        "ensemble, not against the cold Potts model."
        r"  cooling also floors at $\langle d\rangle\approx9$,"
        "\nso the $t=2$ runs have no cooled counterpart at any temperature.",
    )
    style.save(fig, "h04_degeneracy")

    out = config.RESULTS_ROOT / "h04_degeneracy_table.csv"
    long.to_csv(out, index=False)
    print(f"wrote {out}")
    out_match = config.RESULTS_ROOT / "h04_degeneracy_matching.csv"
    matching.to_csv(out_match, index=False)
    print(f"wrote {out_match}")

    # The numbers the README will quote, printed so they can be copied rather
    # than re-derived (and so a change in the data is visible in the log).
    print("\n  per-pair summary (mean over the 79 allowed columns):")
    for index, pair in enumerate(pairs):
        entry = cache[index]
        print(f"    lam={pair['lam']:>5g} t={pair['t']:<4g} vs beta={pair['beta']:<4g}"
              f"  JS {entry['js'].mean():.4f}  (null95 {entry['null_mean_95']:.4f})"
              f"  max JS {entry['js'].max():.3f} at column "
              f"{int(allowed_columns()[entry['js'].argmax()])}"
              f"  n_eff {entry['n_eff_restraint'].mean():.2f} vs "
              f"{entry['n_eff_cooling'].mean():.2f}"
              f"  KL(r||c) {_kl(entry['profile_restraint'], entry['profile_cooling']).mean():.3f}"
              f"  KL(c||r) {_kl(entry['profile_cooling'], entry['profile_restraint']).mean():.3f}")
    top = (long[(long["t"] > 0) & (long["tier"] == "matched")]
           .groupby("msa_column")["js_bits"].mean().sort_values(ascending=False))
    print("\n  columns that differ most, averaged over the matched test pairs:")
    for column, value in top.head(10).items():
        print(f"    column {column}: JS {value:.3f} bits")


if __name__ == "__main__":
    main()
