"""H5 -- the mutation-budget curve, benchmarked against the 1992 constructs.

An experimentalist does not ask "how specific can the model make a sequence".
They ask "I can afford N point mutations -- what is the best I can get for N?"
That is the question t was introduced to answer: t is the chemical potential
conjugate to the Hamming distance, so scanning t at fixed lambda walks the
design ensemble along the mutation-budget axis without freezing the chain.

The readout is the LOG-ODDS at the critic site,

    LOR = log P(S | sigma_\\i*) / P(D | sigma_\\i*),

and NOT P(S). Two reasons, and they matter enough to be restated on the figure:

* P(S) saturates. Half the grid sits above 0.93 and the best designs sit at
  0.99, so a readout on [0, 1] has no room left to separate a good design from
  an excellent one. LOR is unbounded above and keeps resolving -- the same
  designs that differ by 0.04 in P(S) differ by 8 units in LOR, a factor of
  3000 in odds. Panel (b) is that comparison, drawn so it cannot be waved away.
* LOR is a READOUT, never the steering quantity. The sampler optimises
  -lambda*log P(S|.) and knows nothing about P(D|.); the odds against the
  wild-type residue are therefore a quantity the chain was not pointed at, and
  an improvement in it is not a tautology. Nothing in this module feeds LOR
  back into a selection over runs -- the frontier is taken over mutation
  budget, which is an observable, not over LOR-ranked runs.

The benchmark is the Hedstrom, Szilagyi & Rutter 1992 series: three constructs
grafted from rat chymotrypsin into rat trypsin that sit at 12, 13 and 14
mutations from the wild type and that were actually built and assayed. They are
RECONSTRUCTED here from the column positions the literature implicates (the
ladder's ``data.construct_series``), not from sequences reported in the paper,
and only their MODEL score is plotted -- this repository encodes no assayed
activity number, so none is drawn. What the overlay supports is one sentence:
at the same number of mutations, our designs reach X where the real constructs
reach Y.

Four panels:

  (a) the budget curve itself.  Every surviving design as a point in
      (d_H, LOR), with three curves against the budget: the grid mean, the
      single best design, and the FRONTIER -- for each budget, the best over
      the whole lambda x t grid of what one run's top decile reaches at that
      budget. The 1992 series is overlaid at its own (d, LOR). The experimental
      question -- "can a design at d = 14 work" -- is about the best of a
      library, not the mean, so the frontier is the curve drawn heavy.
  (b) why LOR and not P(S).  The frontier designs read out both ways: P(S)
      pinned against the top of its axis, LOR still climbing.
  (c) the trade-off, run by run.  One path per lambda, t increasing along it.
      lambda buys log-odds; t buys budget. The paths never cross, which is why
      the two knobs can be set independently.
  (d) the lookup table.  For each (lambda, t) cell: the mean budget it lands
      on and the best-decile LOR it delivers there. This is the panel someone
      reads when they have a budget and need a run to go and sample.

Read (a) with (d): (a) says what is achievable at a budget, (d) says which
knob setting achieves it.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import config
import data
import style

BLOCK = "a1_graft"

#: Budgets with fewer chains than this are drawn in the cloud but excluded from
#: the frontier curves. A "best of 2 designs" is not a frontier, it is noise,
#: and the tails of the budget distribution are exactly where n collapses.
MIN_CHAINS = 50

#: Chains a single run must put at a budget before its own quantile at that
#: budget is believed, and the number of such runs the frontier needs before it
#: is drawn. The frontier is a MAXIMUM over runs, so it is biased upward by
#: however many noisy estimates it is allowed to choose between; requiring two
#: well-populated runs means the max is a real choice rather than the only
#: number available, and it is what trims the frontier to d = 5..21.
MIN_RUN_CHAINS = 30
MIN_FRONTIER_RUNS = 2

#: The upper quantile that stands for "the best of a library". A screen of a
#: few hundred designs plausibly reaches its own top decile; it does not
#: plausibly reach the single best of 20,000.
BEST_QUANTILE = 0.90

#: Short keys for the 1992 series, in the order ``construct_series`` returns it.
#: Construct 1 IS the wild type, which is why the series starts at d = 0.
CONSTRUCT_KEYS = ["WT", "loops", "+172", "+216,226"]


# --- the 1992 constructs, borrowed from the ladder ------------------------------
@functools.lru_cache(maxsize=1)
def _ladder_data():
    """The ladder's own ``data`` module, loaded with the ladder's own config.

    This is fiddly for one reason: BOTH analysis trees have a module called
    ``config``, and the ladder's data module does a plain ``import config`` at
    top level. Whichever one is in ``sys.modules`` at that moment is the one it
    binds for the rest of the process. So the ladder config is installed under
    the name ``config`` only for the duration of the exec and then put back --
    if it were left there, ``style.save`` would resolve OUT_DIR through the
    ladder's config and quietly write this figure into the ladder's directory.

    Everything about the construct series is taken from the ladder rather than
    rebuilt here: the graft positions, the chymotrypsin donor, the critic
    conditional and the log-odds definition. A second implementation of any of
    them would be a second thing to keep in step, and the whole point of the
    overlay is that our designs and the constructs are scored by one function.
    """
    saved = sys.modules.get("config")

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    ladder_config = _load("ladder_config", config.LADDER_ANALYSIS / "config.py")
    sys.modules["config"] = ladder_config
    try:
        return _load("ladder_data", config.LADDER_ANALYSIS / "data.py")
    finally:
        # Restore even on failure: a half-loaded ladder must not leave this
        # process pointing at the wrong config.
        if saved is not None:
            sys.modules["config"] = saved
        else:                                     # pragma: no cover
            sys.modules.pop("config", None)


def construct_table() -> pd.DataFrame:
    """The 1992 series at its own (d_H, LOR), scored by the same model.

    ``d`` is counted the way the designs count it -- differing columns against
    the wild type over the whole alignment -- so the two are on one axis. The
    loop swap changes length, and the gap columns it introduces are counted
    like any other substitution, because they are mutations an experimentalist
    would have to make.
    """
    ladder = _ladder_data()
    series = ladder.construct_series()
    sequences = np.stack([s for _, s in series])
    wild = ladder.wild_type()

    # P(. | sigma_\i*) conditions on everything EXCEPT the critic site, so it is
    # the same leave-one-site-out prediction used on the designs, and the same
    # function computes it.
    P = ladder.critic_conditional(sequences)
    return pd.DataFrame({
        "key": CONSTRUCT_KEYS,
        "label": [label for label, _ in series],
        "n": np.arange(1, len(series) + 1),
        "hamming_to_wt": [int((seq != wild).sum()) for _, seq in series],
        "P_S": P[:, ladder.S_STATE()],
        "LOR": ladder.log_odds(P),
    })


# --- the designs ----------------------------------------------------------------
def design_frame() -> pd.DataFrame:
    """Every saturated design of A1, with both its budget and its log-odds.

    The two columns come from two trees, for a reason worth spelling out.

    ``<block>_serine`` is the canonical form: the critic column is stamped with
    S and everything is rescored at beta = 1, so ``hamming_to_wt`` counts the
    mutations of the sequence one would actually order -- including the D189S
    itself, on the ~6 % of chains that had drifted off S at the critic. That is
    the honest mutation budget, and it is the one plotted.

    That tree carries no ``LOR`` column (the rescoring writes
    ``critic_residue_before`` in its place), so the log-odds is merged in from
    the raw sampled tree. This is exact rather than approximate: LOR conditions
    on sigma_\\i*, which the critic substitution does not touch, so the two
    trees agree on it by construction -- and indeed they agree on P(S) to the
    last digit on all 20,000 chains. If a future rescoring does emit LOR, the
    merge is skipped and the serine tree's own column is used.
    """
    frame = data.designs(BLOCK)
    if frame["LOR"].notna().any():
        return frame
    raw = data.designs(BLOCK, serine=False)
    merged = frame.drop(columns=["LOR"]).merge(
        raw[["run", "chain", "LOR"]], on=["run", "chain"], how="left", validate="1:1")
    missing = int(merged["LOR"].isna().sum())
    if missing:
        raise SystemExit(f"{missing} chains have no LOR in either tree of {BLOCK}")
    print(f"  {BLOCK}: LOR merged from the raw tree "
          f"({len(merged)} chains); hamming from the serine tree")
    return merged


def frontier(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per mutation budget: what the whole grid can reach at that budget.

    Two different "best" statistics are computed here, and the distinction is
    the methodological heart of the panel.

    ``LOR_best_decile`` pools every chain of the grid that sits at that budget
    and takes the top tenth. It is easy to describe but it is NOT the frontier:
    the pool at d = 10 is dominated by the lambda = 10 and lambda = 30 runs,
    which simply have more mass down there, so a tenth of a diluted library is
    worse than a tenth of the right one. At d = 14 it reads 8.3 where a single
    well-chosen run reaches 10.2.

    ``LOR_frontier`` is the frontier proper: for each budget, the best over all
    (lambda, t) of what THAT run's top decile reaches among its own chains at
    that budget. That is the quantity an experimentalist can act on -- they are
    free to choose which run to sample, they are not free to choose which
    budget they can afford -- and ``frontier_run`` names the run to sample,
    which is what makes panel (d) a lookup rather than a decoration.

    A maximum over runs is biased upward by every noisy estimate it is allowed
    to pick from, so a run must put ``MIN_RUN_CHAINS`` chains at a budget
    before its quantile is used at all, and the frontier is only reported where
    ``MIN_FRONTIER_RUNS`` runs clear that bar.
    """
    rows = []
    for budget, group in frame.groupby("hamming_to_wt", sort=True):
        k = max(1, int(round((1 - BEST_QUANTILE) * len(group))))
        top = group.nlargest(k, "LOR")

        per_run = group.groupby("run").agg(
            n=("LOR", "size"),
            LOR=("LOR", lambda s: s.quantile(BEST_QUANTILE)),
            P_S=("P_S", lambda s: s.quantile(BEST_QUANTILE)),
            lam=("lam", "first"),
            t=("t", "first"),
        )
        eligible = per_run[per_run["n"] >= MIN_RUN_CHAINS]
        best = eligible.loc[eligible["LOR"].idxmax()] if len(eligible) else None

        rows.append({
            "hamming_to_wt": int(budget),
            "n_chains": len(group),
            "LOR_mean": group["LOR"].mean(),
            "LOR_median": group["LOR"].median(),
            "LOR_best_decile": group["LOR"].quantile(BEST_QUANTILE),
            "LOR_max": group["LOR"].max(),
            "P_S_mean": group["P_S"].mean(),
            "P_S_best_decile": group["P_S"].quantile(BEST_QUANTILE),
            "Q_mean": group["Q"].mean(),
            "n_runs": int(len(eligible)),
            "LOR_frontier": float(best["LOR"]) if best is not None else np.nan,
            "P_S_frontier": float(best["P_S"]) if best is not None else np.nan,
            "frontier_run": best.name if best is not None else "",
            "frontier_lam": float(best["lam"]) if best is not None else np.nan,
            "frontier_t": float(best["t"]) if best is not None else np.nan,
            # The modal run among the pooled top decile, as a cross-check that
            # the frontier run is not an artefact of one lucky cell.
            "pooled_top_run": top["run"].value_counts().idxmax(),
        })
    return pd.DataFrame(rows)


def drawable(curve: pd.DataFrame) -> pd.DataFrame:
    """The budgets the frontier is believed at. Everything else stays a cloud."""
    return curve[(curve["n_chains"] >= MIN_CHAINS)
                 & (curve["n_runs"] >= MIN_FRONTIER_RUNS)]


def grid_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (lambda, t) run: where it lands and what it delivers there.

    ``hamming`` is the budget the run is centred on and ``LOR_best_decile`` is
    what the top tenth of that run reaches -- the pair panel (d) prints in each
    cell, and the pair someone with a budget needs in order to choose a run.
    """
    table = data.summarise(frame, by=("lam", "t"))
    extra = frame.groupby(["lam", "t"]).agg(
        LOR_best_decile=("LOR", lambda s: s.quantile(BEST_QUANTILE)),
        LOR_max=("LOR", "max"),
    ).reset_index()
    return table.merge(extra, on=["lam", "t"]).sort_values(["lam", "t"])


# --- drawing --------------------------------------------------------------------
def _lam_shades(frame: pd.DataFrame) -> dict:
    """lambda -> shade along one hue. lambda is ordered, so it gets a ramp and
    not categorical colours, exactly as in h02."""
    values = sorted(frame["lam"].unique())
    return dict(zip(values, style.ramp(style.BLUE, max(len(values), 2))))


def _draw_constructs(ax, constructs: pd.DataFrame, *, numbers: bool = True) -> None:
    """The 1992 series: ink, dotted, numbered -- the house notation for it."""
    ax.plot(constructs["hamming_to_wt"], constructs["LOR"], ls=":", lw=1.8,
            color=style.C_1992, zorder=6)
    for row in constructs.itertuples():
        ax.scatter([row.hamming_to_wt], [row.LOR], s=165, facecolor="white",
                   edgecolor=style.C_1992, linewidths=1.7, zorder=7)
        if numbers:
            ax.annotate(str(row.n), xy=(row.hamming_to_wt, row.LOR), ha="center",
                        va="center", fontsize=10.5, color=style.INK, zorder=8)


def panel_budget(ax, frame: pd.DataFrame, curve: pd.DataFrame,
                 constructs: pd.DataFrame) -> None:
    """(a) the headline: what log-odds is reachable at each mutation budget."""
    shades = _lam_shades(frame)

    # The cloud. Hamming is an integer, so without jitter 20,000 designs stack
    # into twenty-odd vertical lines and the density is unreadable; the jitter
    # is cosmetic only and is smaller than one mutation, so no point moves into
    # a neighbouring budget.
    rng = np.random.default_rng(0)
    for lam, shade in shades.items():
        part = frame[frame["lam"] == lam]
        ax.scatter(part["hamming_to_wt"] + rng.uniform(-0.32, 0.32, len(part)),
                   part["LOR"], s=5, color=shade, alpha=0.16, linewidths=0,
                   zorder=2, rasterized=True)

    good = drawable(curve)
    ax.plot(good["hamming_to_wt"], good["LOR_max"], color=style.MUTED, lw=1.4,
            ls=(0, (1, 2)), zorder=4)
    ax.plot(good["hamming_to_wt"], good["LOR_frontier"], color=style.INK,
            lw=3.0, zorder=5)
    ax.plot(good["hamming_to_wt"], good["LOR_mean"], color=style.MUTED, lw=2.0,
            ls=(0, (5, 3)), zorder=4)

    # LOR = 0 is the point where S and D are equally likely at the critic site.
    # Everything above it is a design the model prefers to read as chymotrypsin.
    ax.axhline(0.0, color=style.RULE, lw=1.5)
    ax.text(ax.get_xlim()[1], 0.3, "equal odds, S vs D", ha="right", va="bottom",
            fontsize=11, color=style.MUTED)

    _draw_constructs(ax, constructs)
    # The series is identified beside its own curve rather than in the legend:
    # four entries made the legend wide enough to sit on constructs 2-4, which
    # are the whole point of the panel.
    ax.text(2.0, -4.0, "the 1992 series\n1 = wild-type trypsin\n"
                       "2–4 were built and assayed",
            fontsize=11.5, color=style.INK, va="bottom", ha="left", zorder=8)

    # The target sentence, drawn. Construct 4 is the deepest of the series and
    # the one with the best model score, so it is the honest comparison point.
    deepest = constructs.iloc[-1]
    budget = int(deepest.hamming_to_wt)
    reached = float(
        curve.loc[curve["hamming_to_wt"] == budget, "LOR_frontier"].iloc[0])
    gap = reached - float(deepest.LOR)
    ax.annotate("", xy=(budget + 0.45, reached), xytext=(budget + 0.45, deepest.LOR),
                arrowprops=dict(arrowstyle="<->", color=style.RED, lw=2.0,
                                shrinkA=0, shrinkB=0), zorder=9)
    ax.text(budget + 0.9, 0.5 * (reached + deepest.LOR),
            f"{gap:+.1f} log-odds\nat the same\n{budget} mutations",
            color=style.RED, fontsize=12.5, va="center", ha="left", zorder=9,
            # The label sits on top of the cloud and the mean line; a plain
            # white backing is the cheapest way to keep it legible there.
            bbox=dict(facecolor=style.SURFACE, edgecolor="none", alpha=0.82,
                      pad=2.5))

    ax.set_xlabel(style.LABELS["hamming_to_wt"] + "   (the mutation budget $d$)")
    ax.set_ylabel(style.LOG_ODDS)
    ax.set_xlim(0, frame["hamming_to_wt"].max() + 1.5)
    ax.legend(handles=[
        Line2D([], [], color=style.MUTED, ls=(0, (1, 2)), lw=1.4,
               label="best single design at $d$"),
        Line2D([], [], color=style.INK, lw=3.0,
               label=f"frontier: best run's top {100 * (1 - BEST_QUANTILE):.0f} % at $d$"),
        Line2D([], [], color=style.MUTED, ls=(0, (5, 3)), lw=2.0,
               label="grid mean at $d$"),
    ], loc="lower right", fontsize=11.5, labelspacing=0.5)
    style.tidy(ax)


def panel_readout(ax, curve: pd.DataFrame) -> None:
    """(b) the same designs, read out two ways. Why this figure is not in P(S).

    Both curves are the frontier of panel (a) -- the same run, the same chains,
    the same top decile at each budget -- so the only thing that differs
    between them is the function applied at the end. P(S) is drawn on its full
    [0, 1] range because that is the range a reader will assume; compressing
    the axis to make it look like a signal would be exactly the error this
    panel exists to warn against.
    """
    good = drawable(curve)
    ax.plot(good["hamming_to_wt"], good["P_S_frontier"], color=style.ORANGE,
            lw=2.8, marker="o", markersize=5, markeredgecolor="white",
            markeredgewidth=0.7, zorder=4)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel(style.P_S, color=style.ORANGE)
    ax.tick_params(axis="y", colors=style.ORANGE)
    ax.set_xlabel(style.LABELS["hamming_to_wt"])

    twin = ax.twinx()
    twin.plot(good["hamming_to_wt"], good["LOR_frontier"], color=style.BLUE,
              lw=2.8, marker="s", markersize=5, markeredgecolor="white",
              markeredgewidth=0.7, zorder=4)
    twin.set_ylabel(style.LOG_ODDS, color=style.BLUE)
    twin.tick_params(axis="y", colors=style.BLUE)
    twin.spines[["top"]].set_visible(False)

    span_p = good["P_S_frontier"].max() - good["P_S_frontier"].min()
    span_l = good["LOR_frontier"].max() - good["LOR_frontier"].min()
    ax.set_title(f"the same designs: $P(\\mathrm{{S}})$ moves {span_p:.2f} over the "
                 f"whole budget range,\nthe log-odds moves {span_l:.1f} "
                 f"(a factor of {np.exp(span_l):,.0f} in odds)",
                 fontsize=12.5, color=style.MUTED, pad=10)
    style.tidy(ax, grid="x")


def panel_paths(ax, grid: pd.DataFrame) -> None:
    """(c) the two knobs, run by run: lambda buys log-odds, t buys budget."""
    shades = _lam_shades(grid)
    for lam, shade in shades.items():
        part = grid[grid["lam"] == lam].sort_values("t")
        ax.plot(part["hamming"], part["LOR_best_decile"], color=shade, lw=1.6,
                ls=(0, (4, 2)), zorder=3)
        ax.plot(part["hamming"], part["LOR"], color=shade, lw=2.2, marker="o",
                markersize=7, markeredgecolor="white", markeredgewidth=0.9,
                zorder=4)

    # Only the top path is annotated with t. Labelling all twenty markers would
    # bury the panel, and the ladder runs the same way on every path.
    top = grid[grid["lam"] == grid["lam"].max()].sort_values("t")
    for row, align in ((top.iloc[0], "left"), (top.iloc[-1], "right")):
        ax.annotate(f"$t$={row.t:g}", xy=(row.hamming, row.LOR),
                    xytext=(6 if align == "left" else -6, -16),
                    textcoords="offset points", ha=align, fontsize=11.5,
                    color=style.MUTED)

    handles = [Line2D([], [], color=shade, marker="o", lw=2.2, markersize=7,
                      label=rf"$\lambda$={lam:g}")
               for lam, shade in shades.items()]
    handles += [Line2D([], [], color=style.MUTED, lw=2.2, label="run mean"),
                Line2D([], [], color=style.MUTED, lw=1.6, ls=(0, (4, 2)),
                       label=f"run's best {100 * (1 - BEST_QUANTILE):.0f} %")]
    ax.legend(handles=handles, loc="lower right", fontsize=11, ncol=2,
              columnspacing=1.1, labelspacing=0.4)
    ax.set_xlabel(r"$\langle d\rangle$ of the run")
    ax.set_ylabel(style.LOG_ODDS)
    style.tidy(ax)


def panel_lookup(ax, grid: pd.DataFrame) -> None:
    """(d) which (lambda, t) lands on which budget, and what it delivers there.

    Colour is the budget, because the budget is what the reader arrives with;
    the number under it is the log-odds they would get. Printing both means the
    cell can be chosen on the constraint and judged on the payoff.
    """
    import matplotlib.colors as mcolors

    lams = sorted(grid["lam"].unique())
    ts = sorted(grid["t"].unique())
    budgets = grid.pivot(index="t", columns="lam", values="hamming").loc[ts, lams]
    scores = grid.pivot(index="t", columns="lam", values="LOR_best_decile").loc[ts, lams]

    # A single-hue sequential map built from the house blue, so the heatmap
    # belongs to the same palette as the ramps in (a) and (c).
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "budget", [style.mix(style.SURFACE, style.BLUE, 0.10),
                   style.mix(style.SURFACE, style.BLUE, 1.0),
                   style.mix(style.BLUE, style.INK, 0.45)])
    image = ax.imshow(budgets.to_numpy(), cmap=cmap, aspect="auto", origin="lower")

    norm = mcolors.Normalize(budgets.to_numpy().min(), budgets.to_numpy().max())
    for i, t in enumerate(ts):
        for j, lam in enumerate(lams):
            value = budgets.iloc[i, j]
            # White on the dark end, ink on the light end -- fixed by the
            # normalised value, not by eye, so it stays right if the data move.
            colour = "white" if norm(value) > 0.60 else style.INK
            ax.text(j, i, f"$\\langle d\\rangle$={value:.0f}\n"
                          f"LOR {scores.iloc[i, j]:.1f}",
                    ha="center", va="center", fontsize=11.5, color=colour)

    ax.set_xticks(range(len(lams)), [f"{v:g}" for v in lams])
    ax.set_yticks(range(len(ts)), [f"{v:g}" for v in ts])
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$t$")
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    bar.set_label(r"mean budget $\langle d\rangle$", fontsize=12)
    bar.ax.tick_params(labelsize=11)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


# --- the figure -----------------------------------------------------------------
def main() -> None:
    style.apply()
    config.ensure_dirs()

    frame = design_frame()
    curve = frontier(frame)
    grid = grid_table(frame)
    constructs = construct_table()

    fig = plt.figure(figsize=(16.2, 11.4))
    # The bottom band is reserved for the caveat, which is four lines long and
    # load-bearing: it is what stops the headline number being over-read.
    gs = fig.add_gridspec(2, 2, left=0.062, right=0.935, top=0.880, bottom=0.172,
                          wspace=0.32, hspace=0.40, width_ratios=[1.32, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_budget(ax_a, frame, curve, constructs)
    panel_readout(ax_b, curve)
    panel_paths(ax_c, grid)
    panel_lookup(ax_d, grid)
    for letter, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        style.panel_letter(ax, letter, dx=-0.115, dy=1.07)

    fig.suptitle("At the mutation budget the 1992 constructs were built at, "
                 "the designed library is far better placed", y=0.965)
    style.caveat(
        fig,
        "Grafting window, 79 columns, $\\beta=1$; all 20 runs of the "
        "$\\lambda\\times t$ grid saturated.\n"
        "The frontier is the best over the grid of one run's top decile at that exact "
        f"budget, drawn only where $\\geq${MIN_FRONTIER_RUNS} runs put "
        f"$\\geq${MIN_RUN_CHAINS} chains at it.\n"
        "The 1992 series is RECONSTRUCTED from the column positions the literature "
        "implicates, and only its MODEL score is shown:\n"
        "this repository encodes no assayed activity for those constructs, so none is "
        "plotted or claimed.\n"
        "$\\mathrm{LOR}$ is a readout, never the steering quantity -- the sampler sees "
        "$-\\lambda\\log P(\\mathrm{S}\\mid\\sigma_{\\setminus i^*})$, never "
        "$P(\\mathrm{D}\\mid\\sigma_{\\setminus i^*})$ -- and a high "
        "$\\mathrm{LOR}$ is a locally rewired S1 site, not a chymotrypsin (SI S7).",
        y=0.012)
    style.save(fig, "h05_budget")

    # --- the numbers behind the figure -----------------------------------------
    # The construct score is attached to the budget it sits at, so a reader can
    # check the headline comparison on one row of one file.
    by_budget = constructs.set_index("hamming_to_wt")
    out = curve.copy()
    out["construct"] = out["hamming_to_wt"].map(by_budget["label"])
    out["construct_LOR"] = out["hamming_to_wt"].map(by_budget["LOR"])
    out["construct_P_S"] = out["hamming_to_wt"].map(by_budget["P_S"])
    out["delta_frontier"] = out["LOR_frontier"] - out["construct_LOR"]
    out["delta_best_decile"] = out["LOR_best_decile"] - out["construct_LOR"]
    out["drawn"] = out["hamming_to_wt"].isin(drawable(curve)["hamming_to_wt"])
    path = config.RESULTS_ROOT / "h05_budget_table.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}")

    grid_path = config.RESULTS_ROOT / "h05_budget_grid.csv"
    grid[["lam", "t", "hamming", "var_hamming", "LOR", "LOR_best_decile", "LOR_max",
          "P_S", "Q", "n_chains"]].to_csv(grid_path, index=False)
    print(f"wrote {grid_path}")

    print(f"\n{'construct':<28}{'d':>4}{'LOR':>9}{'frontier':>10}{'(run)':>14}"
          f"{'best design':>13}")
    for row in constructs.itertuples():
        match = curve[curve["hamming_to_wt"] == row.hamming_to_wt]
        if not len(match):      # the wild type, at d = 0, where we sample nothing
            print(f"{row.label:<28}{row.hamming_to_wt:>4}{row.LOR:>9.2f}"
                  f"{'--':>10}{'--':>14}{'--':>13}")
            continue
        line = match.iloc[0]
        print(f"{row.label:<28}{row.hamming_to_wt:>4}{row.LOR:>9.2f}"
              f"{line['LOR_frontier']:>10.2f}{line['frontier_run']:>14}"
              f"{line['LOR_max']:>13.2f}")


if __name__ == "__main__":
    main()
