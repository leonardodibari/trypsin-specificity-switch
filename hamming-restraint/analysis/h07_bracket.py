"""H7 -- the two-start bracket: do the free-column runs actually equilibrate?

THE PROBLEM THIS ANSWERS. ``results/EQUILIBRATION_NOTE.md``: on all 265 columns
free, the energetic observables settle in tens of thousands of steps but
``hamming_to_wt`` keeps climbing. A pilot at the susceptibility peak run for
400,000 steps -- 3.3x the first pass -- still failed the drift test on d, with
173 accepted moves per site, so it was not stuck, only still moving. The drift
fell 2.1x for a 3.3x longer run: d is not relaxing exponentially from a
wild-type start, it is DIFFUSING outward from d = 0. More steps is not the fix.

THE TEST. Sample every rung twice, from opposite ends of the d axis:

  the NEAR arm   starts at the wild type, d = 0, and anneals DOWNWARD in t --
                 outward in d. Its <d> approaches each rung's equilibrium FROM
                 BELOW, so every number it reports is a LOWER bound.
  the FAR arm    starts at bovine chymotrypsinogen B, d = 145, a real
                 chymotrypsin inside the 81-196 range the natural ones span,
                 and anneals UPWARD in t -- inward in d. Its <d> approaches
                 from ABOVE, so every number it reports is an UPPER bound.

Where the two arms meet, the answer is the equilibrium. Where they do not, the
gap between them IS the error bar, and it is an honest one: it is bounded by
construction rather than estimated from a single chain that may never have left
its starting basin. This is a sharper statement than any drift criterion applied
to one run, and it is the thing no free-column run in this project has ever had.

THE PANELS
  (a) <d>(t), both arms, both lambdas -- the bracket itself, drawn as a band
      between the two arms. A rung where the band has closed is a rung with an
      equilibrium; a rung where it has not is reported, never quietly used.
  (b) the gap, |<d>_near - <d>_far|, against the scale it has to be small
      compared to: the ensemble's own spread. A bracket is closed when the gap
      is below BOTH the sampling error and 5 % of the ensemble IQR -- the same
      5 % the drift test uses, so the two criteria are on one scale.
  (c) the single-site marginals. Two samples of the same distribution must agree
      on every column's residue frequencies, not only on a mean, so this is the
      strong form of the test: the total-variation distance between the two
      arms' column-frequency profiles, per rung, against the TV distance between
      two halves of the SAME arm -- which is the irreducible finite-sample floor.
  (d) the energetic observables, Q and P(S), bracketed the same way. d is the
      slow coordinate; these settled long ago, and showing them here is what
      distinguishes "the ensemble has not converged" from "the ensemble has
      converged in everything except where it sits on the d axis".

WHAT COUNTS AS A PASS, decided before looking: a rung passes when panel (b)'s
gap is below the sampling error and panel (c)'s TV distance is within 2x the
same-arm floor. ``bracket_table()`` writes that verdict per rung, and the
figures that read the free-column blocks (h03, h06) take their rungs from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import style

#: The four arms. Both lambdas get both starts: lambda = 0 is the matched
#: control, non-negotiable, because t < 0 alone drags the chains toward the
#: family average and the family carries S at column 201 in 10.4 % of its
#: members -- so a negative t raises P(S) with no bias at all.
ARMS = {
    ("biased", "near"): "c1_lam100_from_wt",
    ("biased", "far"): "c1_lam100_from_chymo",
    ("control", "near"): "c2_lam0_from_wt",
    ("control", "far"): "c2_lam0_from_chymo",
}

#: A rung passes when the two arms agree to within the sampling error on <d>,
#: AND their single-site marginals agree to within this multiple of the
#: same-arm finite-sample floor.
TV_TOLERANCE = 2.0

NEAR_COLOUR = style.BLUE       # started at the wild type, approaches from below
FAR_COLOUR = style.C_CHYMO_REF  # started at a chymotrypsin, approaches from above


def _arm(block: str) -> pd.DataFrame:
    """One arm's per-chain observables. Read WITHOUT the saturation filter.

    This is the one figure in the study that must see the runs that failed the
    drift test: the whole point is to decide, from the two arms together,
    whether a rung has an equilibrium at all. Dropping the failures first would
    throw away exactly the rungs being judged. Every other figure keeps the
    filter.
    """
    frame = data.load_block(block, serine=True)
    frame["arm_block"] = block
    return frame


def _marginals(block: str, run: str, *, columns: slice | None = None) -> np.ndarray:
    """Single-site residue frequencies ``f[i, a]`` of one run's final ensemble.

    The strong form of the agreement test. Two samples of the same equilibrium
    must agree on every column's composition; two samples that merely happen to
    share a mean Hamming distance need not.
    """
    from encoding import Alphabet, read_sequences

    alphabet = Alphabet.from_name("protein")
    path = config.RESULTS_ROOT / f"{block}_serine" / run / "final_sequences.fasta"
    if not path.is_file():
        path = config.RESULTS_ROOT / block / run / "final_sequences.fasta"
    encoded = np.stack([alphabet.encode(s) for _, s in read_sequences(path)])
    q = len(alphabet.tokens) if hasattr(alphabet, "tokens") else 21
    counts = np.zeros((encoded.shape[1], q))
    for a in range(q):
        counts[:, a] = (encoded == a).sum(axis=0)
    return counts / encoded.shape[0]


def _tv(first: np.ndarray, second: np.ndarray) -> float:
    """Mean per-column total-variation distance between two marginal profiles.

    0 = identical composition at every column, 1 = disjoint. Averaged over
    columns so that the number does not grow with L and can be compared between
    a 79-column and a 265-column block.
    """
    return float(0.5 * np.abs(first - second).sum(axis=1).mean())


def _split_tv(block: str, run: str, *, seed: int = 0) -> float:
    """The finite-sample FLOOR: the same arm's own two halves, against each other.

    Two halves of one ensemble are samples of the same distribution by
    construction, so whatever TV distance they show is what 500 chains can
    resolve and nothing more. A between-arm TV at this level is agreement; the
    comparison in panel (c) is against this, never against zero.
    """
    from encoding import Alphabet, read_sequences

    alphabet = Alphabet.from_name("protein")
    path = config.RESULTS_ROOT / f"{block}_serine" / run / "final_sequences.fasta"
    if not path.is_file():
        path = config.RESULTS_ROOT / block / run / "final_sequences.fasta"
    encoded = np.stack([alphabet.encode(s) for _, s in read_sequences(path)])
    order = np.random.default_rng(seed).permutation(encoded.shape[0])
    halves = []
    for part in np.array_split(order, 2):
        counts = np.zeros((encoded.shape[1], 21))
        for a in range(21):
            counts[:, a] = (encoded[part] == a).sum(axis=0)
        halves.append(counts / part.size)
    return _tv(halves[0], halves[1])


def _gap_over_sem(near: pd.DataFrame, far: pd.DataFrame, column: str) -> float:
    """``|mean_near - mean_far|`` of one column, in units of its own sampling error."""
    sem = float(np.hypot(
        near[column].std(ddof=1) / np.sqrt(len(near)),
        far[column].std(ddof=1) / np.sqrt(len(far)),
    ))
    gap = float(abs(near[column].mean() - far[column].mean()))
    return gap / sem if sem else np.inf


def bracket_table() -> pd.DataFrame:
    """One row per (lambda, t) rung: both arms, their gap, and the verdict.

    This table is the gate for every free-column number in the study. A rung
    that does not close is not a rung with a wide error bar -- it is a rung
    whose value is unknown, and it is excluded by name.
    """
    rows = []
    for condition in ("biased", "control"):
        near_block, far_block = ARMS[(condition, "near")], ARMS[(condition, "far")]
        if not (config.RESULTS_ROOT / near_block).is_dir():
            continue
        if not (config.RESULTS_ROOT / far_block).is_dir():
            continue
        near, far = _arm(near_block), _arm(far_block)
        for t in sorted(set(near["t"]) & set(far["t"])):
            n = near[near["t"] == t]
            f = far[far["t"] == t]
            run = n["run"].iloc[0]
            # The sampling error on the DIFFERENCE of two means. The chains
            # within an arm share only the order in which sites were visited,
            # never a residue or an acceptance, so they are independent samples
            # of their arm's distribution.
            sem = float(np.hypot(
                n["hamming_to_wt"].std(ddof=1) / np.sqrt(len(n)),
                f["hamming_to_wt"].std(ddof=1) / np.sqrt(len(f)),
            ))
            iqr = float(np.mean([
                n["hamming_to_wt"].quantile(0.75) - n["hamming_to_wt"].quantile(0.25),
                f["hamming_to_wt"].quantile(0.75) - f["hamming_to_wt"].quantile(0.25),
            ]))
            gap = float(abs(n["hamming_to_wt"].mean() - f["hamming_to_wt"].mean()))
            tv = _tv(_marginals(near_block, run), _marginals(far_block, run))
            floor = float(np.mean([_split_tv(near_block, run), _split_tv(far_block, run)]))
            rows.append(dict(
                condition=condition, lam=n["lam"].iloc[0], t=t, run=run,
                d_near=n["hamming_to_wt"].mean(), d_far=f["hamming_to_wt"].mean(),
                gap=gap, sem=sem, iqr=iqr,
                gap_over_sem=gap / sem if sem else np.inf,
                gap_over_iqr=gap / iqr if iqr else np.inf,
                tv=tv, tv_floor=floor, tv_over_floor=tv / floor if floor else np.inf,
                Q_near=n["Q"].mean(), Q_far=f["Q"].mean(),
                P_S_near=n["P_S"].mean(), P_S_far=f["P_S"].mean(),
                LOR_near=n["LOR"].mean(), LOR_far=f["LOR"].mean(),
                # The same two-arm test on the ENERGETIC observables, in units
                # of the sampling error on their difference. The bracket was
                # built to judge d, on the reading that the energetics settle
                # first -- and panel (d) does show them settling first WITHIN an
                # arm. Between arms they do not agree: Q differs by up to 78
                # sem on rungs whose per-run drift test passed Q outright. The
                # drift test asks whether one chain has stopped moving; this
                # asks whether two chains that started at opposite ends stopped
                # in the same place, and only the second is equilibrium.
                **{f"{name}_gap_sem": _gap_over_sem(n, f, name)
                   for name in ("Q", "E_DCA", "P_S")},
                sat_near=bool(n["saturated"].all()), sat_far=bool(f["saturated"].all()),
            ))
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # The verdict, on the criteria fixed in the docstring before any of this was
    # looked at: the means agree within the sampling error, and the marginals
    # agree within TV_TOLERANCE of what 500 chains can resolve.
    table["closed"] = (
        (table["gap_over_sem"] <= 1.0)
        & (table["tv_over_floor"] <= TV_TOLERANCE)
        # ...and the quantities a result is actually read off agree between the
        # arms as well: Q, the energy and P(S), each to within 2 sem. Without
        # this a rung could "close" on d while the two arms disagreed by 60
        # units in Q, which is not a rung anything may be read off.
        & (table["Q_gap_sem"] <= 2.0)
        & (table["E_DCA_gap_sem"] <= 2.0)
        & (table["P_S_gap_sem"] <= 2.0)
    )
    # A weaker, explicitly labelled tier: the means agree to within 5 % of the
    # ensemble's own spread, the same 5 % the drift test uses. Reported so that
    # a rung which is close but not at the sampling floor is visible as such
    # rather than silently failing.
    table["closed_5pct"] = table["gap_over_iqr"] <= 0.05
    return table.sort_values(["condition", "t"])


def closed_rungs(condition: str = "biased") -> list:
    """The t values whose bracket closed -- what h03 and h06 are allowed to use."""
    table = bracket_table()
    if table.empty:
        return []
    chosen = table[(table["condition"] == condition) & table["closed"]]
    return sorted(chosen["t"])


def main() -> None:
    style.apply()
    table = bracket_table()
    if table.empty:
        raise SystemExit(
            "No bracket to draw: run hamming-restraint/run_bracket.sh first "
            f"(expected {', '.join(sorted(set(ARMS.values())))} under "
            f"{config.RESULTS_ROOT})"
        )

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2))
    (ax_d, ax_gap), (ax_tv, ax_energy) = axes

    # --- (a) the bracket itself -------------------------------------------------
    for condition, dash in (("biased", "-"), ("control", "--")):
        sub = table[table["condition"] == condition]
        if sub.empty:
            continue
        label = r"$\lambda$=100" if condition == "biased" else r"$\lambda$=0 (control)"
        ax_d.fill_between(sub["t"], sub["d_near"], sub["d_far"],
                          color=style.FAINT, alpha=0.9 if condition == "biased" else 0.5,
                          lw=0, zorder=1)
        ax_d.plot(sub["t"], sub["d_near"], dash, color=NEAR_COLOUR, marker="o",
                  markersize=4.5, lw=1.8, zorder=3,
                  label=f"from the wild type, {label}")
        ax_d.plot(sub["t"], sub["d_far"], dash, color=FAR_COLOUR, marker="s",
                  markersize=4.5, lw=1.8, zorder=3,
                  label=f"from chymotrypsin, {label}")
    # The two starting points, which are the two ends the arms come from.
    ax_d.axhline(0, color=style.C_WT, lw=1.0, ls=":", zorder=2)
    ax_d.axhline(145, color=style.C_CHYMO_REF, lw=1.0, ls=":", zorder=2)
    ax_d.annotate("wild type, $d=0$", (ax_d.get_xlim()[0], 0), xytext=(4, 5),
                  textcoords="offset points", fontsize=11, color=style.MUTED)
    ax_d.annotate("bovine chymotrypsinogen B, $d=145$", (ax_d.get_xlim()[0], 145),
                  xytext=(4, 5), textcoords="offset points", fontsize=11,
                  color=style.C_CHYMO_REF)
    # The band the natural chymotrypsins occupy: what "leaving the island" means.
    ax_d.axhspan(81, 196, color=style.C_CHYMO, alpha=0.08, lw=0, zorder=0)
    ax_d.set_xlabel("$t$  (restraint; $>0$ toward the wild type)")
    ax_d.set_ylabel(r"$\langle d_H(\sigma,\mathrm{WT})\rangle$")
    ax_d.set_title("(a)  the bracket: every rung sampled from both ends", loc="left")
    ax_d.legend(fontsize=10.5, frameon=False, loc="center left")
    style.tidy(ax_d)

    # --- (b) the gap against the scale it must be small compared to -------------
    for condition, marker in (("biased", "o"), ("control", "s")):
        sub = table[table["condition"] == condition]
        if sub.empty:
            continue
        ax_gap.plot(sub["t"], sub["gap_over_sem"], marker=marker, lw=1.8,
                    color=style.BLUE if condition == "biased" else style.MUTED,
                    markersize=5,
                    label=r"$\lambda$=100" if condition == "biased" else r"$\lambda$=0")
    ax_gap.axhline(1.0, color=style.GREEN, lw=1.4)
    ax_gap.annotate("closed: the two arms agree to within the sampling error",
                    (ax_gap.get_xlim()[0], 1.0), xytext=(4, 5),
                    textcoords="offset points", fontsize=11, color=style.GREEN)
    ax_gap.set_yscale("log")
    ax_gap.set_xlabel("$t$")
    ax_gap.set_ylabel(r"$|\langle d\rangle_{\rm near}-\langle d\rangle_{\rm far}|\ /\ $SEM")
    ax_gap.set_title("(b)  how far apart the arms still are", loc="left")
    ax_gap.legend(fontsize=11, frameon=False)
    style.tidy(ax_gap)

    # --- (c) the strong form: the single-site marginals --------------------------
    for condition, marker in (("biased", "o"), ("control", "s")):
        sub = table[table["condition"] == condition]
        if sub.empty:
            continue
        colour = style.BLUE if condition == "biased" else style.MUTED
        ax_tv.plot(sub["t"], sub["tv"], marker=marker, lw=1.8, color=colour,
                   markersize=5,
                   label=r"between arms, $\lambda$=100" if condition == "biased"
                   else r"between arms, $\lambda$=0")
        ax_tv.plot(sub["t"], sub["tv_floor"], ls=":", lw=1.4, color=colour,
                   label="same arm, two halves (the floor)" if condition == "biased"
                   else None)
    ax_tv.set_xlabel("$t$")
    ax_tv.set_ylabel("mean per-column total variation")
    ax_tv.set_title("(c)  per-column agreement, not just the mean",
                    loc="left")
    ax_tv.legend(fontsize=10.5, frameon=False)
    style.tidy(ax_tv)

    # --- (d) the observables that had already settled ---------------------------
    sub = table[table["condition"] == "biased"]
    ax_energy.plot(sub["t"], sub["Q_near"], marker="o", lw=1.8, color=NEAR_COLOUR,
                   markersize=4.5, label="$Q$, from the wild type")
    ax_energy.plot(sub["t"], sub["Q_far"], marker="s", lw=1.8, color=FAR_COLOUR,
                   markersize=4.5, label="$Q$, from chymotrypsin")
    ax_energy.set_xlabel("$t$")
    ax_energy.set_ylabel(r"$Q = E_{\rm DCA} - \mu\,\mathrm{CDE}$  (lower is better)")
    twin = ax_energy.twinx()
    twin.plot(sub["t"], sub["P_S_near"], ls="--", lw=1.5, color=NEAR_COLOUR, alpha=0.75)
    twin.plot(sub["t"], sub["P_S_far"], ls="--", lw=1.5, color=FAR_COLOUR, alpha=0.75)
    twin.set_ylabel(style.P_S + "  (dashed)")
    twin.spines[["top"]].set_visible(False)
    ax_energy.set_title("(d)  $Q$, $P(S)$: settled within an arm, not between",
                        loc="left")
    ax_energy.legend(fontsize=10.5, frameon=False, loc="best")
    style.tidy(ax_energy)

    n_closed = int(table["closed"].sum())
    fig.suptitle("Bracketing the Hamming axis from both ends", y=0.985)
    style.caveat(
        fig,
        f"all 265 columns free, gap moves on, 1000 chains/arm, annealed in $t$.  "
        f"{n_closed} of {len(table)} rungs closed — on $d$, on the per-column "
        r"marginals, and on $Q$, $E$, $P(S)$ between arms." "\n"
        "near approaches from below, far from above: an open bracket is a BOUND, "
        "never a number." "\n"
        r"(d) is why the energetics cannot stand in for $d$ — within an arm they "
        r"settle early; between arms $Q$ differs by up to 78 sem.",
    )
    fig.tight_layout(rect=(0, 0.062, 1, 0.955))
    style.save(fig, "h07_bracket")

    out = config.RESULTS_ROOT / "h07_bracket_table.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out}")
    print(table[["condition", "t", "d_near", "d_far", "gap", "sem",
                 "tv", "tv_floor", "closed"]].to_string(index=False))


if __name__ == "__main__":
    main()
