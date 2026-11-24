"""H1 -- has every observable stopped moving?

This figure comes first because every other figure depends on it. A result may
only be read off a run in which EVERY quantity it is read off has saturated:
the energy, the quality Q the designs are ranked by, the Hamming distance the
restraint acts on and its indel part, P(S), the log-odds, and the context
entropy. Flattening in one of them says nothing about the others -- Q can still
be drifting while P(S) has long since saturated, because Q is dominated by the
energy and the energy keeps falling.

Two panels per block:

  (a) the traced mean of each observable, one line per run, each normalised to
      its own final value's IQR so that seven quantities on different scales can
      share one axis. A settled run is a flat line on the right-hand half.
  (b) the drift test itself: residual drift over the second half as a fraction
      of the ensemble's own IQR, per observable per run, against the 5 %
      threshold. Anything to the right of the rule is still moving.

A run that is flat because it is frozen is not settled, so the accepted moves
per mutable site are printed next to each run and anything under 20 is marked
as stuck -- the pathology CLAMP_CONTROL.md caveat 1 describes, where the beta=2
and beta=5 ensembles differ by two columns out of 265 and are not equilibrium
results at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
import data
import saturation
import style


def _run_colours(runs: list[str]) -> dict:
    """A single-hue ramp ordered by t: an ordered variable gets a ramp, never
    categorical hues."""
    ordered = sorted(runs, key=data._parse_run_name)
    colours = style.ramp(style.BLUE, len(ordered))
    return dict(zip(ordered, colours))


def panel_traces(ax, block: str, runs: list[str], observable: str) -> None:
    """The traced mean of one observable, one line per run."""
    colours = _run_colours(runs)
    for run in runs:
        frame = data.trace(block, run)
        column = f"{observable}_mean"
        if column not in frame:
            continue
        steps = frame["step"].to_numpy(dtype=float)
        values = frame[column].to_numpy(dtype=float)
        ax.plot(steps, values, color=colours[run], lw=1.6, solid_capstyle="round")
    ax.set_xscale("log")
    ax.set_ylabel(style.LABELS.get(observable, observable), fontsize=13)
    style.tidy(ax, grid="y")


def figure_traces(block: str) -> Path:
    """Every observable of every run of one block, as a column of traces."""
    runs = data.runs(block)
    observables = [
        o for o in saturation.OBSERVABLES
        if f"{o}_mean" in data.trace(block, runs[0]).columns
    ]
    fig, axes = plt.subplots(
        len(observables), 1, figsize=(9.0, 2.0 * len(observables)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, observable in zip(axes, observables):
        panel_traces(ax, block, runs, observable)
        # The second half, where the drift test is taken.
        last = data.trace(block, runs[0])["step"].max()
        ax.axvspan(last / 2, last, color=style.FAINT, alpha=0.45, zorder=0, lw=0)
    axes[-1].set_xlabel("proposed moves per chain")

    verdicts = [data.verdict(block, run) for run in runs]
    n_ok = sum(1 for v in verdicts if v.saturated)
    where, what = config.BLOCKS.get(block, (block, ""))
    fig.suptitle(
        f"{block}  ·  {where}  ·  {n_ok}/{len(runs)} runs saturated on every observable",
        y=0.995,
    )
    style.caveat(
        fig,
        "shaded: the second half, over which the drift is measured.  "
        "one line per run, light to dark with increasing t.",
    )
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    return style.save(fig, f"h01_saturation_traces_{block}")


def figure_verdicts(blocks: list[str]) -> Path:
    """The drift test for every run of every block, against its threshold."""
    rows = []
    for block in blocks:
        for run in data.runs(block):
            verdict = data.verdict(block, run)
            for observable in verdict.observables:
                rows.append((f"{block}/{run}", observable.name,
                             observable.drift_over_spread, verdict.stuck))
    if not rows:
        raise SystemExit("no traced runs")

    names = sorted({r[0] for r in rows}, key=lambda s: (s.split("/")[0],
                                                        data._parse_run_name(s.split("/")[1])))
    observables = [o for o in saturation.OBSERVABLES if any(r[1] == o for r in rows)]
    index = {name: i for i, name in enumerate(names)}

    fig, ax = plt.subplots(figsize=(10.5, 0.33 * len(names) + 2.4))
    colours = style.ramp(style.BLUE, len(observables))
    for offset, (observable, colour) in enumerate(zip(observables, colours)):
        y = [index[r[0]] + (offset - len(observables) / 2) * 0.1 for r in rows if r[1] == observable]
        x = [max(r[2], 1e-4) for r in rows if r[1] == observable]
        ax.scatter(x, y, s=22, color=colour, label=observable, zorder=3,
                   edgecolor="none")
    ax.axvline(saturation.DRIFT_OVER_SPREAD, color=style.RED, lw=1.6, zorder=2)
    ax.text(saturation.DRIFT_OVER_SPREAD * 1.1, -1.4,
            f"threshold {saturation.DRIFT_OVER_SPREAD:g}", color=style.RED, fontsize=12)
    ax.set_xscale("log")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(
        [n + ("  (stuck)" if any(r[0] == n and r[3] for r in rows) else "") for n in names],
        fontsize=10,
    )
    ax.set_xlabel("residual drift over the second half, as a fraction of the ensemble IQR")
    ax.invert_yaxis()
    ax.legend(ncol=len(observables), loc="upper center", bbox_to_anchor=(0.5, 1.09),
              fontsize=11, handletextpad=0.3, columnspacing=1.1)
    style.tidy(ax, grid="x")
    style.caveat(
        fig,
        "a run passes only if EVERY observable is left of the rule.  "
        "'stuck' = fewer than "
        f"{saturation.MIN_ACCEPTED_PER_SITE:g} accepted moves per mutable site, "
        "which is flat because it is frozen, not because it is settled.",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return style.save(fig, "h01_saturation_verdicts")


def main(blocks: list[str] | None = None) -> None:
    style.apply()
    config.ensure_dirs()
    available = [b for b in (blocks or config.BLOCKS)
                 if (config.RESULTS_ROOT / b).is_dir() and data.runs(b)]
    if not available:
        raise SystemExit(f"no sampled block under {config.RESULTS_ROOT}")
    for block in available:
        figure_traces(block)
    figure_verdicts(available)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
