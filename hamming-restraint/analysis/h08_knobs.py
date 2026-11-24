"""H8 -- test 2: are the three knobs independent?

THE QUESTION. The Hamiltonian now carries three parameters,

    H = beta*E_DCA - lambda*log P(S | sigma_\\i*) + t*d_H(sigma, WT)

and the stated reason for adding the third was that the mutation count had been
a *byproduct* of the other two: it fell out of (beta, lambda) together with the
start and the allowed-column set, so a design could not be asked for at a chosen
budget. The claim to be tested is therefore a claim about a division of labour:

    lambda  should trade P(S) against Q
    beta    should trade Q against diversity
    t       should set <d> and move the other two ONLY THROUGH d

That last clause is the whole content of "t does its own job". If t moves Q or
P(S) by some route other than the budget it spends, then it is not a budget
knob, and if it moves them along the SAME path beta already traces then it is a
reparametrised temperature and h04 would have caught it.

THE TEST. A knob's identity is the *path it drags the ensemble along*, not the
point it lands on, so each knob is measured by its cost per unit of budget --
the slope of each observable against <d> along that knob's own axis, every other
parameter held:

    dQ/dd      quality bought per substitution of budget given up
    dP_S/dd    specificity bought per substitution
    d(unique)/dd   library kept per substitution
    d(acc)/dd      and whether the chain was still moving while it happened

Two knobs that move the ensemble along the same path have the same slopes. Two
knobs with different slopes are doing different things, whatever their endpoints
look like. The slopes are read off a regression over each path's own runs, which
is why this figure needs the whole grid and not a pair of matched points.

WHY THIS BLOCK. Everything here is the 79-column grafting block at 60,000 steps,
where all 20 runs of the lambda x t grid saturate on every observable
(`results/EQUILIBRATION_NOTE.md`: the all-265-free runs do not, and no number is
read off them here). The beta arm is the ladder's own cooling run, read through
the same serine door as everything else.

THE ONE THING THIS FIGURE CANNOT SAY. The cooled rungs at beta >= 2 are *stuck*,
not settled -- CLAMP_CONTROL caveat 1, and panel (d) is where that is visible.
Their Q is therefore not an equilibrium number, and the beta slope is reported
with that written next to it rather than quietly averaged in. The honest
comparison against a WARM competitor is the beta = 1.5 rung, and it is marked.

THE PANELS
  (a) Q against <d>          the quality cost of a substitution, per knob
  (b) P(S) against <d>       the specificity cost, per knob
  (c) unique designs against <d>   the library
  (d) accepted moves per site against <d>   settled, or merely frozen
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import style

import h02_pareto

#: The three paths. Each is a set of runs along which exactly one knob moves.
#:   t      -- a1_graft, beta = 1, lambda held, t stepped 0 -> 2
#:   beta   -- the cooling ladder, lambda/beta held (matched P(S)), beta stepped
#:   lambda -- the ladder's own beta = 1 row, t = 0, lambda stepped
KNOBS = {
    "t": (r"$t$  (restraint)", style.BLUE, "o"),
    "beta": (r"$\beta$  (cooling)", style.ORANGE, "s"),
    "lam": (r"$\lambda$  (bias)", style.AQUA, "^"),
}

#: A run is frozen rather than settled when the project's own criterion says so:
#: ``data.cooling_ladder`` sets ``stuck`` at beta >= 2 (CLAMP_CONTROL caveat 1,
#: where RESULTS.md cannot tell a converged cold run from a stuck one). The
#: measured accepted-moves-per-site is reported beside it in panel (d) rather
#: than used as the criterion. Here the two are directly comparable and checked
#: to be: both blocks sampled 60,000 steps over the same 79 mutable columns
#: (``results/ladder/cooling/b*/ensembles/lam*/summary.json`` against
#: ``run_phase_a.sh``), so accepted/site means the same thing on both axes. That
#: is a property of these runs, not of the quantity -- it would stop holding the
#: moment a block were sampled for a different number of steps.
STUCK_COLUMN = "stuck"


# --------------------------------------------------------------------------
# the three paths
# --------------------------------------------------------------------------

def paths() -> Dict[str, pd.DataFrame]:
    """One frame per knob, each carrying <d>, Q, P_S, unique_pct, acceptance.

    Both tables come from h02 rather than being rebuilt: the Pareto figure
    already reads the grid and the ladder through the serine door and already
    attaches the library size and the acceptance, and a second copy of that
    would be a second thing to keep in step.
    """
    grid = h02_pareto.restraint_table().copy()
    cooling = h02_pareto.cooling_table().copy()

    out: Dict[str, pd.DataFrame] = {}

    # t: one sub-path per lambda. beta = 1 throughout, so every run is warm.
    grid["path"] = grid["lam"].map(lambda v: f"$\\lambda={v:g}$")
    grid["knob_value"] = grid["t"]
    grid["beta"] = 1.0
    out["t"] = grid

    # beta: one sub-path per lam_tilde = lambda/beta, which is what the ladder
    # holds fixed down a rung -- cooling at matched bias-to-energy ratio.
    cooling["path"] = cooling["lam_tilde"].map(lambda v: f"$\\lambda/\\beta={v:g}$")
    cooling["knob_value"] = cooling["beta"]
    cooling["t"] = 0.0
    out["beta"] = cooling[cooling["lam_tilde"].isin([10.0, 30.0, 100.0])].copy()

    # lambda: the ladder's beta = 1 row. One path only -- there is nothing else
    # to hold fixed, which is itself the point: lambda is the knob the study had.
    #
    # cooling_table has already dropped what the drift test failed, which for
    # this row is lambda = 0 (P_S flat at 0.0014) and lambda = 1000 (still
    # drifting in E_DCA and hamming_to_wt). What is left is lambda = 10..300,
    # and that is the whole lambda path -- there is nothing else to hold fixed,
    # which is itself the point: lambda is the knob the study already had.
    lam_path = cooling[np.isclose(cooling["beta"], 1.0)].copy()
    lam_path["path"] = r"$\beta=1,\ t=0$"
    lam_path["knob_value"] = lam_path["lam"]
    out["lam"] = lam_path

    return out


def slope(frame: pd.DataFrame, column: str) -> float:
    """d(column)/d<d> along one path, by least squares over that path's runs.

    A slope, not a difference of endpoints: the endpoints of the three paths sit
    at different <d>, so any endpoint comparison would be comparing the paths at
    different budgets, which is the error this whole figure exists to avoid.
    """
    frame = frame.dropna(subset=["hamming", column])
    if len(frame) < 2:
        return float("nan")
    return float(np.polyfit(frame["hamming"].to_numpy(float),
                            frame[column].to_numpy(float), 1)[0])


def slope_table(knob_paths: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The figure's number: every observable's cost per unit of budget, per path."""
    rows: List[dict] = []
    for knob, frame in knob_paths.items():
        for name, group in frame.groupby("path", sort=False):
            group = group.sort_values("hamming")
            warm = ~group[STUCK_COLUMN].to_numpy(bool)
            rows.append(
                dict(
                    knob=knob,
                    path=name,
                    n_runs=len(group),
                    d_min=group["hamming"].min(),
                    d_max=group["hamming"].max(),
                    dQ_dd=slope(group, "Q"),
                    dP_S_dd=slope(group, "P_S"),
                    dunique_dd=slope(group, "unique_pct"),
                    daccepted_dd=slope(group, "accepted_per_site"),
                    # A slope averaged over stuck runs is not an equilibrium
                    # slope. Reported, never silently folded in.
                    n_stuck=int((~warm).sum()),
                    # The same slopes over the WARM runs alone. For beta that is
                    # beta = 1 and 1.5 only, so it is a two-point slope and the
                    # single honest version of the comparison; for t and lambda
                    # every run is warm and it equals the full slope.
                    dQ_dd_warm=slope(group[warm], "Q"),
                    dunique_dd_warm=slope(group[warm], "unique_pct"),
                    acc_max=group["accepted_per_site"].max(),
                    acc_min=group["accepted_per_site"].min(),
                    unique_max=group["unique_pct"].max(),
                    unique_min=group["unique_pct"].min(),
                )
            )
    return pd.DataFrame(rows)


def matched_budget(knob_paths, tolerance: float = 0.6) -> pd.DataFrame:
    """Every (t-run, cooled-run) pair that sits at the same <d>, side by side.

    The slopes say the paths differ; this says what the difference IS at a budget
    an experimentalist would actually name. Pairs are formed within a matched
    lambda-tilde so that the bias-to-energy ratio is held too, and the residual
    in <d> is carried so a reader can see how good the match is.
    """
    rows = []
    cooled = knob_paths["beta"]
    for _, left in knob_paths["t"].iterrows():
        family = cooled[np.isclose(cooled["lam_tilde"], left["lam"])]
        for _, right in family.iterrows():
            residual = abs(float(left["hamming"]) - float(right["hamming"]))
            if residual > tolerance or np.isclose(right["beta"], 1.0):
                continue
            rows.append(dict(
                lam_tilde=left["lam"], t=left["t"], beta=right["beta"],
                d_t=left["hamming"], d_beta=right["hamming"], residual_d=residual,
                Q_t=left["Q"], Q_beta=right["Q"], dQ=right["Q"] - left["Q"],
                P_S_t=left["P_S"], P_S_beta=right["P_S"],
                unique_t=left["unique_pct"], unique_beta=right["unique_pct"],
                acc_t=left["accepted_per_site"], acc_beta=right["accepted_per_site"],
                beta_stuck=bool(right[STUCK_COLUMN]),
            ))
    return pd.DataFrame(rows).sort_values(["lam_tilde", "d_t"]) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------

def _draw(axis, knob_paths: Dict[str, pd.DataFrame], column: str, ylabel: str) -> None:
    for knob, frame in knob_paths.items():
        label, colour, marker = KNOBS[knob]
        shades = style.ramp(colour, max(frame["path"].nunique(), 2))
        for index, (name, group) in enumerate(frame.groupby("path", sort=False)):
            group = group.sort_values("hamming")
            if column not in group:
                continue
            stuck = group[STUCK_COLUMN].to_numpy(bool)
            axis.plot(group["hamming"], group[column], "-", color=shades[index],
                      lw=1.6, zorder=2)
            # Open markers for a run that is frozen rather than settled: the
            # distinction panel (d) measures, carried into every other panel so
            # a reader cannot take a cold point for an equilibrium one.
            axis.plot(group.loc[~stuck, "hamming"], group.loc[~stuck, column],
                      marker, color=shades[index], ms=6.0, mew=1.3, zorder=3)
            axis.plot(group.loc[stuck, "hamming"], group.loc[stuck, column],
                      marker, color="none", mec=shades[index], ms=6.0, mew=1.3,
                      zorder=3)
    axis.set_xlabel(r"$\langle d_H(\sigma,\mathrm{WT})\rangle$   (mutation budget)")
    axis.set_ylabel(ylabel)
    style.tidy(axis)


def main() -> None:
    style.apply()
    knob_paths = paths()
    table = slope_table(knob_paths)

    print("\n  cost per unit of budget, along each knob's own axis")
    print("  (dQ/dd: quality per substitution; dP_S/dd: specificity per substitution)")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.6))
    _draw(axes[0, 0], knob_paths, "Q", style.LABELS.get("Q", "$Q = E - \\mu\\,\\mathrm{CDE}$"))
    _draw(axes[0, 1], knob_paths, "P_S", style.P_S)
    _draw(axes[1, 0], knob_paths, "unique_pct", "unique designs  (%)")
    _draw(axes[1, 1], knob_paths, "accepted_per_site", "accepted moves per mutable site")
    axes[1, 1].set_yscale("log")

    for axis, letter in zip(axes.ravel(), "abcd"):
        style.panel_letter(axis, letter)

    handles = [
        plt.Line2D([], [], color=colour, marker=marker, lw=1.6, ms=6.0, label=label)
        for label, colour, marker in KNOBS.values()
    ]
    handles.append(
        plt.Line2D([], [], color="none", mec=style.MUTED, marker="o", lw=0, ms=6.0,
                   mew=1.3, label=r"open: $\beta\geq2$, frozen rather than settled")
    )
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=9, loc="best")

    fig.suptitle("H8 — are the three knobs independent? Cost per unit of mutation budget",
                 fontsize=13)
    style.caveat(
        fig,
        "79-column grafting block, 60,000 steps on both axes, all 20 runs of the "
        r"$\lambda\times t$ grid saturated.  $t$ at $\beta=1$;  $\beta$ at matched "
        r"$\lambda/\beta$;  $\lambda$ at $\beta=1,t=0$." "\n"
        r"Open markers are cooled rungs that are frozen rather than settled "
        r"($\beta\geq2$, CLAMP_CONTROL caveat 1), so their $Q$ is not an equilibrium "
        r"number — the warm competitor is $\beta=1.5$.  "
        "Slopes: h08_knob_slopes.csv.  Per-column spectra at matched budget: h04.",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    style.save(fig, "h08_knobs")

    out = config.RESULTS_ROOT / "h08_knob_slopes.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out}")

    matched = matched_budget(knob_paths)
    if not matched.empty:
        print("\n  the same budget, reached two ways")
        with pd.option_context("display.width", 220, "display.max_columns", 24):
            print(matched.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        path = config.RESULTS_ROOT / "h08_matched_budget.csv"
        matched.to_csv(path, index=False)
        print(f"wrote {path}")

    # The verdict, stated as the ratio the figure is built to produce.
    t_slopes = table[table["knob"] == "t"]["dQ_dd"]
    b_warm = table[table["knob"] == "beta"]["dQ_dd_warm"]
    lam_slopes = table[table["knob"] == "lam"]["dP_S_dd"]
    print(f"\n  dQ/dd: t = {t_slopes.mean():+.3f} (mean over lambda), "
          f"beta (warm rungs only) = {b_warm.mean():+.3f}  "
          f"-> ratio {b_warm.mean()/t_slopes.mean():.1f}x")
    print(f"  dP_S/dd: lambda = {lam_slopes.mean():+.4f}, "
          f"t = {table[table['knob']=='t']['dP_S_dd'].mean():+.4f}  "
          f"-> ratio {lam_slopes.mean()/table[table['knob']=='t']['dP_S_dd'].mean():.0f}x")


if __name__ == "__main__":
    main()
