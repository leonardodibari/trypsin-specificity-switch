"""Has every observable of a run stopped moving?

A result may only be read off a run in which **every** quantity it is read off
has saturated -- not the energy alone, not P(S) alone. The quality Q can still
be drifting while P(S) has flattened, and a cold chain can look settled in all
of them simply because it has stopped moving at all. So this module checks each
traced observable separately and a run passes only if all of them pass.

The observables, all traced per chain and summarised as mean and quartiles in
``trace.csv`` (see :func:`ensemble._record`):

    E_DCA           the Potts energy
    Q               E_DCA - mu * CDE, the score designs are ranked by
    hamming_to_wt   the distance the restraint acts on
    sub_to_wt       its substitution part, which t_sub prices
    indel_to_wt     its indel part, which t_indel prices
    P_S             P(S | context) at the critic site
    LOR             log P(S)/P(D) there, which keeps resolving after P_S saturates
    CDE             the context-dependent entropy

THE CRITERION
-------------
For each observable, over the **second half** of the run:

    drift   = |mean over the last quarter - mean over the third quarter|
    spread  = the ensemble's own IQR, averaged over the second half
    range   = the full excursion of the mean over the whole run

and the observable is saturated when

    ( drift < NEGLIGIBLE_DRIFT[observable]  OR  drift < DRIFT_OVER_SPREAD * spread )
    AND drift < DRIFT_OVER_RANGE * range

The second clause of the first line is the usual one: the residual drift is
small against the spread of the ensemble it would have to be read against. The
first clause beside it is an absolute floor in the observable's own units, and
it is there because a RELATIVE criterion cannot express "this movement is too
small to matter" for a quantity the ensemble has no spread in: ``indel_to_wt``
is 0 for most chains, so its IQR is exactly 0 and a wobble of 0.016 indels per
chain -- which could not change any claim -- reads as infinite relative drift.

The last clause guards both: the chain must not still be making progress of the
same order as the progress it has already made.

The z-score of the drift against the trace's own point-to-point noise is also
computed and reported, as a diagnostic for reading a borderline run. It is NOT
a criterion: the traced points are the same chains at different times, so they
are correlated and that standard error is optimistic.

STUCK IS NOT SETTLED
--------------------
A chain with acceptance 0.0015 produces a flat trace because it is frozen, not
because it has equilibrated (see docs/CLAMP_CONTROL.md caveat 1). So the report
also carries the acceptance rate and the accepted moves per chain, and flags a
run with fewer than :data:`MIN_ACCEPTED_PER_SITE` accepted moves per mutable
site as *stuck* however flat its trace is. A stuck run is reported as NOT
saturated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

#: The observables a result is ever read off, in reporting order.
OBSERVABLES = ["E_DCA", "Q", "hamming_to_wt", "sub_to_wt", "indel_to_wt", "P_S", "LOR",
               "CDE"]

#: Residual drift allowed, as a fraction of the ensemble's own IQR.
DRIFT_OVER_SPREAD = 0.05
#: Residual drift allowed, as a fraction of the run's total excursion.
DRIFT_OVER_RANGE = 0.10
#: A drift below this, in the observable's own units, cannot move any number
#: this study or the lambda ladder reports, so it is not a drift. Each entry is
#: roughly a tenth of the SMALLEST effect ever claimed on that observable --
#: the ladder separates conditions by ~9 in Q, by 3-5 substitutions, by 0.1 in
#: P(S), and reports the indel count to two decimals (RESULTS.md sections 4-5).
#: This is what answers an observable the ensemble has no spread in, such as
#: indel_to_wt, whose IQR is exactly 0 because most chains carry no indel at
#: all and against which any wobble would read as infinite relative drift.
NEGLIGIBLE_DRIFT = {
    "E_DCA": 0.5,
    "Q": 0.5,
    "hamming_to_wt": 0.05,
    # The two parts the restraint now prices separately. Each is judged on its
    # own: t_sub acts on sub_to_wt and t_indel on indel_to_wt, so a run whose
    # total d has settled while weight is still moving between the two has not
    # settled for either price. A trace written before the split carries no
    # sub_to_wt column and check_observable returns None for it, so old runs
    # keep exactly the verdicts they had.
    "sub_to_wt": 0.05,
    "indel_to_wt": 0.05,
    "P_S": 0.005,
    "LOR": 0.05,
    "CDE": 0.002,
}
#: Below this many accepted moves per mutable site the chain is frozen rather
#: than settled, and its flat trace means nothing. Calibrated against the
#: pathology it exists to catch, NOT against what looks comfortable: the cold
#: rungs of CLAMP_CONTROL.md caveat 1 -- acceptance 0.008 at beta = 2, roughly
#: 500 accepted moves over 78 sites, i.e. 6 per site -- are the runs whose
#: beta = 2 and beta = 5 ensembles differ by two columns out of 265 and which
#: are explicitly not equilibrium results. A run above this floor is not thereby
#: proved equilibrated; it is only not disproved. The drift test is the verdict,
#: and the two-start check of h04 is what actually settles a borderline case.
MIN_ACCEPTED_PER_SITE = 5.0


@dataclass
class ObservableVerdict:
    """The saturation of one observable of one run."""

    name: str
    final: float
    drift: float
    spread: float
    range: float
    drift_over_spread: float
    drift_over_range: float
    z: float
    negligible: bool
    saturated: bool


def worst_relative_drift(verdicts: List["ObservableVerdict"]) -> float:
    """Largest relative drift among the observables that actually moved.

    An observable whose absolute drift is negligible is left out: its relative
    drift is meaningless and often infinite, and printing ``inf`` next to a run
    that is settled on every count is how a report gets ignored.
    """
    moved = [v.drift_over_spread for v in verdicts if not v.negligible]
    return max(moved) if moved else 0.0


@dataclass
class RunVerdict:
    """The saturation of one run: the AND over its observables."""

    run: str
    n_steps: int
    n_points: int
    acceptance_rate: float
    accepted_per_site: float
    stuck: bool
    saturated: bool
    failures: List[str]
    observables: List[ObservableVerdict]

    def to_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "run": self.run,
            "n_steps": self.n_steps,
            "n_points": self.n_points,
            "acceptance_rate": self.acceptance_rate,
            "accepted_per_site": self.accepted_per_site,
            "stuck": self.stuck,
            "saturated": self.saturated,
            "failures": ",".join(self.failures),
        }
        for verdict in self.observables:
            row[f"{verdict.name}_drift_over_spread"] = verdict.drift_over_spread
            row[f"{verdict.name}_final"] = verdict.final
        return row


def check_observable(frame: pd.DataFrame, name: str) -> Optional[ObservableVerdict]:
    """Saturation of one observable, or None if the trace does not carry it."""
    column = f"{name}_mean"
    if column not in frame:
        return None
    steps = frame["step"].to_numpy(dtype=float)
    values = frame[column].to_numpy(dtype=float)
    if values.size < 4 or not np.all(np.isfinite(values)):
        return None

    half = steps[-1] / 2.0
    quarter = steps[-1] * 3.0 / 4.0
    third_quarter = values[(steps >= half) & (steps < quarter)]
    last_quarter = values[steps >= quarter]
    if third_quarter.size == 0 or last_quarter.size == 0:
        # Too few traced points in the second half to say anything. Fall back to
        # the last two points, and let the thinness show up as n_points.
        third_quarter, last_quarter = values[-2:-1], values[-1:]
    drift = abs(float(last_quarter.mean() - third_quarter.mean()))

    # How much the trace wobbles from point to point in the second half, with
    # each quarter's own mean removed, so a step between the quarters is not
    # counted as noise. This is the scale a drift is "small" against for an
    # observable the ensemble has no spread in.
    noise = float(
        np.sqrt(
            (third_quarter.var(ddof=0) * third_quarter.size
             + last_quarter.var(ddof=0) * last_quarter.size)
            / max(third_quarter.size + last_quarter.size, 1)
        )
    )
    standard_error = noise * np.sqrt(1 / max(third_quarter.size, 1)
                                     + 1 / max(last_quarter.size, 1))
    # Reported, not judged on: these are the same chains at different times, so
    # the points are correlated and this standard error is optimistic. It is a
    # useful diagnostic for reading a borderline run, not a criterion.
    z = drift / standard_error if standard_error > 0 else (0.0 if drift == 0 else float("inf"))

    if f"{name}_q25" in frame and f"{name}_q75" in frame:
        iqr = (frame[f"{name}_q75"] - frame[f"{name}_q25"]).to_numpy(dtype=float)
        spread = float(np.mean(iqr[steps >= half]))
    else:
        spread = float("nan")
    excursion = float(values.max() - values.min())

    # An observable that never moves at all -- indel_to_wt when gaps are inert,
    # for instance -- has zero drift and zero spread. That is saturated, not
    # infinitely drifting, so the zero drift is answered before the zero spread.
    if drift == 0.0:
        over_spread = 0.0
    elif spread > 0:
        over_spread = drift / spread
    else:
        over_spread = float("inf")
    over_range = drift / excursion if excursion > 0 else 0.0
    return ObservableVerdict(
        name=name,
        final=float(values[-1]),
        drift=drift,
        spread=spread,
        range=excursion,
        drift_over_spread=over_spread,
        drift_over_range=over_range,
        z=z,
        negligible=bool(drift < NEGLIGIBLE_DRIFT.get(name, 0.0)),
        saturated=bool(
            (drift < NEGLIGIBLE_DRIFT.get(name, 0.0) or over_spread < DRIFT_OVER_SPREAD)
            and over_range < DRIFT_OVER_RANGE
        ),
    )


def check_run(directory: str | Path, name: Optional[str] = None) -> RunVerdict:
    """Saturation of one ensemble directory (the one holding ``trace.csv``)."""
    directory = Path(directory)
    frame = pd.read_csv(directory / "trace.csv")
    with open(directory / "summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)

    verdicts = [v for v in (check_observable(frame, o) for o in OBSERVABLES) if v is not None]
    n_sites = int(summary.get("n_allowed_sites", 1)) or 1
    accepted_per_site = (
        float(frame["accepted_per_chain"].iloc[-1]) / n_sites
        if "accepted_per_chain" in frame
        else float("nan")
    )
    stuck = bool(accepted_per_site < MIN_ACCEPTED_PER_SITE)
    failures = [v.name for v in verdicts if not v.saturated]
    return RunVerdict(
        run=name or directory.name,
        n_steps=int(summary.get("n_steps", frame["step"].iloc[-1])),
        n_points=len(frame),
        acceptance_rate=float(summary.get("acceptance_rate", float("nan"))),
        accepted_per_site=accepted_per_site,
        stuck=stuck,
        saturated=bool(not failures and not stuck),
        failures=failures + (["stuck"] if stuck else []),
        observables=verdicts,
    )


def check_tree(root: str | Path) -> List[RunVerdict]:
    """Every run under ``root`` that has a ``trace.csv``, named by relative path."""
    root = Path(root)
    out = []
    for trace in sorted(root.rglob("trace.csv")):
        directory = trace.parent
        if not (directory / "summary.json").is_file():
            continue
        out.append(check_run(directory, name=str(directory.relative_to(root))))
    return out


def report(verdicts: List[RunVerdict]) -> pd.DataFrame:
    """The verdicts as a table, worst first."""
    frame = pd.DataFrame([v.to_row() for v in verdicts])
    if frame.empty:
        return frame
    frame["worst_drift_over_spread"] = [
        worst_relative_drift(v.observables) for v in verdicts
    ]
    return frame.sort_values(["saturated", "worst_drift_over_spread"], ascending=[True, False])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="Directory holding the runs (searched recursively)")
    parser.add_argument("--csv", default=None, help="Write the full table here")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any run has not saturated, so a driver can stop on it",
    )
    args = parser.parse_args()

    verdicts = check_tree(args.root)
    if not verdicts:
        raise SystemExit(f"No traced run under {args.root}")
    frame = report(verdicts)
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)

    print(f"{'run':38s} {'acc':>7s} {'acc/site':>9s} {'worst':>7s}  verdict")
    print("-" * 80)
    for verdict in sorted(verdicts, key=lambda v: (v.saturated, v.run)):
        worst = worst_relative_drift(verdict.observables)
        status = "SATURATED" if verdict.saturated else "NOT SATURATED: " + ", ".join(verdict.failures)
        print(
            f"{verdict.run:38s} {verdict.acceptance_rate:7.4f} "
            f"{verdict.accepted_per_site:9.1f} {worst:7.3f}  {status}"
        )
    n_bad = sum(1 for v in verdicts if not v.saturated)
    print("-" * 80)
    print(f"{len(verdicts) - n_bad}/{len(verdicts)} runs saturated on every observable")
    if args.csv:
        print(f"table -> {args.csv}")
    if n_bad and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
