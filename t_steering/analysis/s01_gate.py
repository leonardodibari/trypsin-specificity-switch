"""S1 -- the gate. Did all 48 runs saturate on E, Q, d and P(S)?

THIS FIGURE COMES FIRST AND EVERY OTHER ONE DEPENDS ON IT. The grid is a full
factorial -- 3 mutation sets x 2 beta x 2 lambda x 4 t -- and a factorial is
only a factorial if every cell is usable. One unsaturated cell does not merely
weaken its own row: it breaks the comparison the design exists to make, because
the means of the other cells are then being compared against a number that is
still moving.

WHAT IS TESTED. The drift test of ``hamming-restraint/analysis/saturation.py``,
applied to each of ``E_DCA``, ``Q``, ``hamming_to_wt`` and ``P_S`` SEPARATELY:
residual drift over the second half of the run as a fraction of the ensemble's
own IQR, against a 5 % threshold. Separately, because flattening in one says
nothing about the others -- Q is dominated by the energy and keeps falling after
P(S) has long since saturated.

THE SECOND FAILURE MODE, and it is the one a drift test cannot see. A run that
is flat because it is FROZEN passes a drift test trivially: zero motion is zero
drift. So the accepted moves per mutable site is reported beside every verdict,
and a run under 20 is marked as stuck whatever its drift says. This is why
``beta >= 2`` is not in this grid at all (CLAMP_CONTROL caveat 1), and panel (b)
is where that choice is checked rather than assumed.

THE PANELS
  (a) the drift test itself: residual drift as a fraction of the ensemble IQR,
      per observable per run, against the 5 % rule. Anything to the right fails.
  (b) accepted moves per mutable site, per run, against the stuck floor.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import saturation
import style

#: Below this many accepted moves per mutable site a run is frozen rather than
#: settled, and nothing may be read off it however flat its trace.
STUCK_ACCEPTED_PER_SITE = 20.0

#: The drift test's own threshold, restated so the figure can draw it.
DRIFT_THRESHOLD = 0.05


def verdict_table() -> pd.DataFrame:
    """One row per run x observable: its drift, its verdict, its acceptance."""
    rows = []
    for set_name in config.SETS:
        for beta in config.BETAS:
            sampled = config.block_dir(set_name, beta)
            if not sampled.is_dir():
                continue
            for verdict in saturation.check_tree(sampled):
                passes, failed = data.gated_verdict(verdict)
                for observable in verdict.observables:
                    if observable.name not in config.GATED:
                        continue
                    rows.append(dict(
                        set=set_name, beta=beta, run=verdict.run,
                        observable=observable.name,
                        drift=observable.drift,
                        drift_over_spread=observable.drift_over_spread,
                        negligible=observable.negligible,
                        saturated=observable.saturated,
                        run_passes_gate=passes,
                        run_failed_on=",".join(failed) or "-",
                        accepted_per_site=verdict.accepted_per_site,
                        stuck=verdict.accepted_per_site < STUCK_ACCEPTED_PER_SITE,
                        n_steps=verdict.n_steps,
                    ))
    return pd.DataFrame(rows)


def main() -> int:
    style.apply()
    config.ensure_dirs()

    table = verdict_table()
    if table.empty:
        raise SystemExit(
            f"no traced runs under {config.RESULTS_ROOT} -- "
            "run t_steering/run_steering.sh first"
        )

    runs = table.drop_duplicates(subset=["set", "beta", "run"])
    n_runs = len(runs)
    n_pass = int(runs["run_passes_gate"].sum())
    n_stuck = int(runs["stuck"].sum())

    print(f"\n  {n_runs} of {config.N_RUNS} runs sampled")
    print(f"  {n_pass} / {n_runs} pass the gate on {', '.join(config.GATED)}")
    print(f"  {n_stuck} / {n_runs} are frozen rather than settled "
          f"(< {STUCK_ACCEPTED_PER_SITE:g} accepted moves per mutable site)")

    failures = runs[~runs["run_passes_gate"]]
    if not failures.empty:
        print("\n  FAILED:")
        for row in failures.itertuples():
            print(f"    {row.set}/beta{row.beta:g}/{row.run}  "
                  f"on {row.run_failed_on}  (acc/site {row.accepted_per_site:.1f})")
    stuck = runs[runs["stuck"]]
    if not stuck.empty:
        print("\n  STUCK (flat because frozen, not because settled):")
        for row in stuck.itertuples():
            print(f"    {row.set}/beta{row.beta:g}/{row.run}  "
                  f"acc/site {row.accepted_per_site:.1f}")

    out = config.RESULTS_ROOT / "s01_gate.csv"
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")

    # --- the figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.4))

    ax = axes[0]
    colours = dict(zip(config.GATED, style.ramp(style.BLUE, len(config.GATED))))
    for index, name in enumerate(config.GATED):
        subset = table[table["observable"] == name]
        # An observable whose absolute drift is negligible has a meaningless
        # relative drift; it is plotted at the axis floor and said to be so.
        x = np.where(subset["negligible"], 1e-4, subset["drift_over_spread"].clip(lower=1e-4))
        y = index + np.linspace(-0.3, 0.3, len(subset))
        ax.scatter(x, y, s=26, color=colours[name],
                   edgecolor=np.where(subset["saturated"], "white", style.RED),
                   linewidth=0.9, zorder=3)
    ax.axvline(DRIFT_THRESHOLD, color=style.RED, ls="--", lw=1.2, zorder=2)
    ax.text(DRIFT_THRESHOLD, len(config.GATED) - 0.35, " 5 % rule", color=style.RED,
            fontsize=10, va="top")
    ax.set_xscale("log")
    ax.set_yticks(range(len(config.GATED)))
    ax.set_yticklabels([style.LABELS.get(n, n) for n in config.GATED])
    ax.set_xlabel("residual drift over the second half  /  ensemble IQR")
    ax.set_title("(a)  the drift test, per observable per run", loc="left")
    style.tidy(ax)

    ax = axes[1]
    for index, set_name in enumerate(config.SETS):
        subset = runs[runs["set"] == set_name]
        marker = ["o", "s", "^"][index % 3]
        for beta, shade in zip(config.BETAS, style.ramp(style.AQUA, len(config.BETAS))):
            arm = subset[np.isclose(subset["beta"], beta)]
            ax.scatter(arm["accepted_per_site"], np.full(len(arm), index)
                       + (0.18 if beta > 1.0 else -0.18),
                       s=34, marker=marker, color=shade, edgecolor="white",
                       linewidth=0.8, zorder=3,
                       label=f"{set_name}, " + r"$\beta$=" + f"{beta:g}")
    ax.axvline(STUCK_ACCEPTED_PER_SITE, color=style.RED, ls="--", lw=1.2, zorder=2)
    ax.text(STUCK_ACCEPTED_PER_SITE, -0.45, " stuck", color=style.RED, fontsize=10)
    ax.set_xscale("log")
    ax.set_yticks(range(len(config.SETS)))
    ax.set_yticklabels(list(config.SETS))
    ax.set_xlabel("accepted moves per mutable site")
    ax.set_title("(b)  settled, or merely frozen?", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style.tidy(ax)

    fig.suptitle(
        f"S1 — the gate: {n_pass} of {n_runs} runs saturated on "
        + ", ".join(config.GATED), fontsize=13)
    style.caveat(
        fig,
        "60,000 steps, 1000 chains, every run of the 3 sets x 2 "
        r"$\beta$ x 2 $\lambda$ x 4 $t$ factorial.  each observable is tested "
        "separately — flattening in one says nothing about the others." "\n"
        "a red ring marks an observable that failed; points at the axis floor "
        "have negligible absolute drift, so their relative drift is not "
        "meaningful.  (b) is the failure a drift test cannot see.",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    style.save(fig, "s01_gate", out_dir=config.OUT_DIR)

    # The exit code IS the gate: a non-zero status stops a driver that chains
    # the figures, so a later figure cannot be built on an ungated run.
    return 0 if n_pass == n_runs and n_runs == config.N_RUNS else 1


if __name__ == "__main__":
    sys.exit(main())
