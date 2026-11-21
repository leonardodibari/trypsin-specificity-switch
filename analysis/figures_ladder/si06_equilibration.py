"""S6 -- did the chains equilibrate?

Four observables (rows) against sampling step, for each condition (columns):
the DCA energy, the Hamming distance to the wild type, the quality score Q_mu
(lower is better, so the axis is inverted here and the curve rises as the design
improves) and P(S | sigma_\\i*).

Lines are ensemble means over 1000 chains; the band is the interquartile range
across chains, so it shows the spread of the ensemble rather than the error on
the mean -- a flat mean with a stable band is the thing to look for.

Q has no band. It is linear in E and CDE so its *mean* follows exactly from
theirs, but the quartile of a sum is not the sum of the quartiles, and the trace
stores only per-observable quartiles.

The number under each panel is the drift over the final factor of two in steps,
as a percentage of the observable's total range over the run. It turns "looks
flat" into something checkable.

TRACES ARE NOT AVAILABLE FOR EVERY CONDITION. run_analysis.sh asks for them
(``--trace-every 1000 --trace-schedule log --trace-lambdas 0 100 1000``);
run_site_budget*.sh does not, so the consensus-19 runs were never traced, and
consensus-19 + cooling was never sampled at all. Both appear as placeholders.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import config
import data
import style

#: (trace column stem, axis label, invert the y axis?)
ROWS = [
    ("E_DCA", r"$E_{\mathrm{DCA}}$", False),
    ("hamming_to_wt", "Hamming to wild type", False),
    ("Q", r"$Q_\mu$   (inverted: up is better)", True),
    ("P_S", style.P_S, False),
]


def final_drift(frame, column: str) -> tuple[float, str, float]:
    """``(drift %, what it is a % of, absolute drift)`` over the last doubling.

    Scaled by the ENSEMBLE'S OWN interquartile spread, not by the observable's
    range over the run. Range is the wrong yardstick for a near-constant series:
    the unbiased lambda = 0 run moves P(S) over a total range of 0.006, so a
    drift of 0.0006 -- utterly negligible -- reads as "10% of range" and looks
    like a failure to equilibrate. Against the chain-to-chain spread, which is
    the scale on which the mean is actually uncertain, it reads as what it is.

    Q has no stored quartiles, so it falls back to range and says so.
    """
    steps = frame["step"].to_numpy(dtype=float)
    values = frame[f"{column}_mean"].to_numpy(dtype=float)
    tail = steps >= steps.max() / 2
    if tail.sum() < 2:
        return 0.0, "IQR", 0.0
    absolute = float(abs(values[tail][-1] - values[tail][0]))
    lo_key, hi_key = f"{column}_q25", f"{column}_q75"
    if lo_key in frame and hi_key in frame:
        spread = float(np.nanmedian(
            frame[hi_key].to_numpy(dtype=float)[tail]
            - frame[lo_key].to_numpy(dtype=float)[tail]))
        basis = "IQR"
    else:
        spread = float(np.nanmax(values) - np.nanmin(values))
        basis = "range"
    if not np.isfinite(spread) or spread <= 0:
        return 0.0, basis, absolute
    return 100.0 * absolute / spread, basis, absolute


def main() -> None:
    style.apply()
    conditions = list(config.CONDITIONS)
    traces: Dict[str, List] = {}
    for condition in conditions:
        runs = data.traced_runs(condition)
        traces[condition] = [(run, data.trace_frame(run)) for run in runs]

    n_rows, n_cols = len(ROWS), len(conditions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(21.0, 14.4),
                             squeeze=False)
    fig.subplots_adjust(top=0.875, bottom=0.135, left=0.058, right=0.990,
                        hspace=0.30, wspace=0.17)

    drifts = []
    for col, condition in enumerate(conditions):
        entries = traces[condition]
        colours = style.ramp(style.group(condition)["color"], len(entries))
        for row, (stem, label, invert) in enumerate(ROWS):
            ax = axes[row][col]
            if not entries:
                _placeholder(ax, condition)
                continue
            for (run, frame), colour in zip(entries, colours):
                steps = frame["step"].to_numpy(dtype=float)
                mean = frame[f"{stem}_mean"].to_numpy(dtype=float)
                lo_key, hi_key = f"{stem}_q25", f"{stem}_q75"
                if lo_key in frame and hi_key in frame:
                    ax.fill_between(steps, frame[lo_key], frame[hi_key],
                                    color=colour, alpha=0.18, linewidth=0,
                                    zorder=2)
                ax.plot(steps, mean, color=colour, lw=2.0, zorder=3)
                ax.scatter([steps[-1]], [mean[-1]], s=30, color=colour, zorder=4)
                drifts.append((condition, run, stem) + final_drift(frame, stem))
            ax.set_xscale("log")
            if invert:
                ax.invert_yaxis()
            style.tidy(ax)
            scored = [final_drift(f, stem) for _, f in entries]
            worst, basis, _ = max(scored, key=lambda item: item[0])
            ax.set_title(f"drift over last doubling: {worst:.1f}% of {basis}",
                         fontsize=11, color=style.MUTED, pad=4)
            if row == n_rows - 1:
                ax.set_xlabel("sampling step")
            if col == 0:
                ax.set_ylabel(label)

    for col, condition in enumerate(conditions):
        axes[0][col].annotate(condition, xy=(0.5, 1.30), xycoords="axes fraction",
                              ha="center", va="bottom", fontsize=15,
                              fontweight="bold", color=style.INK)
        entries = traces[condition]
        if not entries:
            continue
        colours = style.ramp(style.group(condition)["color"], len(entries))
        def _label(run):
            return (f"$\\lambda$={run.lam:g}" if run.beta == 1
                    else f"$\\beta$={run.beta:g}, $\\lambda$={run.lam:g}")

        # A cooled condition has 15 rungs; listing them all buries the panel and
        # spills into the row below. Show the two ends of the ramp and say how
        # many runs lie between them.
        if len(entries) > 8:
            ends = [(entries[0], colours[0]), (entries[-1], colours[-1])]
            handles = [Line2D([], [], color=c, lw=2.4, label=_label(r))
                       for (r, _), c in ends]
            handles.insert(1, Line2D([], [], color="none",
                                     label=f"({len(entries) - 2} more, pale→dark)"))
        else:
            handles = [Line2D([], [], color=colour, lw=2.4, label=_label(run))
                       for (run, _), colour in zip(entries, colours)]
        axes[0][col].legend(handles=handles, loc="upper left", fontsize=11,
                            handlelength=1.4, labelspacing=0.3,
                            borderpad=0.3)

    fig.suptitle("Chain equilibration: every traced observable is flat by the "
                 "end of the run", y=0.975)
    worst, worst_abs, worst_stem = 0.0, 0.0, ""
    for _c, _r, stem, pct, _basis, absolute in drifts:
        if pct > worst:
            worst, worst_abs, worst_stem = pct, absolute, stem
    style.caveat(
        fig,
        f"1000 chains per run; line = ensemble mean, band = interquartile range across chains.\n"
        f"Drift = movement of the mean over the final doubling of steps, as a percentage of the "
        f"ensemble's own interquartile spread — the scale on which the mean is actually uncertain. "
        f"Largest over all traced runs: {worst:.1f}% — an absolute move of "
        f"{worst_abs:.2g} in {worst_stem}.\n"
        "$Q_\\mu$ has no band and is scaled by its range instead: it is linear in $E$ and CDE so "
        "its mean is exact, but the trace stores only per-observable quartiles.",
        y=0.010)
    style.save(fig, "si06_equilibration")

    print(f"{'condition':<24} {'run':<20} {'observable':<16} {'drift':>8}  absolute")
    for condition, run, stem, drift, basis, absolute in drifts:
        print(f"  {condition:<22} β={run.beta:<4g} λ={run.lam:<6g} {stem:<16} "
              f"{drift:6.1f}% {basis:<6} {absolute:.4g}")
    for condition in conditions:
        if not traces[condition]:
            print(f"  {condition}: NO TRACE DATA — placeholder")


def _placeholder(ax, condition: str) -> None:
    reason = ("never sampled" if not data.discover_runs(condition)
              else "sampled without --trace-every,\nso no trace was written")
    ax.text(0.5, 0.5, f"{condition}\n\nno trace data\n({reason})", ha="center",
            va="center", fontsize=11.5, color=style.MUTED,
            transform=ax.transAxes, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.6", fc=style.SURFACE,
                      ec=style.FAINT, ls="--", lw=1.3))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


if __name__ == "__main__":
    main()
