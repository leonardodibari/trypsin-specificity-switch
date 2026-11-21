"""Which runs go in the selection table — and the per-column "best" marking.

There is deliberately NO scoring rule here. A single formula over five competing
desiderata (high P(S), low Q, few mutations, diverse library, high log-odds)
hides which criterion it traded away, and the weights would be arbitrary. So:

* ``full_pool()`` scores every run and goes in the SI table;
* ``SELECTED`` below is a hand-picked list, edited by a human who has looked at
  that table;
* ``best_in_column()`` marks the best value per column in BOTH tables, which is
  what makes the pool readable enough to pick from.

To change the selection, edit ``SELECTED``. Nothing else needs touching.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config
import data

#: (condition, beta, lambda). Hand-picked after reading the full pool (SI S2).
#: These are a starting point, not a verdict -- edit freely.
#:
#: Why these: for each condition, one run that buys the switch most cheaply and
#: one that keeps the library diverse, since those two goals pull apart hardest.
#: Hand-picked rows, per model. Keyed by model because the conditions, the
#: lambda ladders and the consensus set sizes all differ between them.
#: Edit freely -- this is the whole point of the design: no scoring rule, just a
#: human choosing from the full pool in SI S2.
_SELECTED: dict = {
    # Provisional for the serine-protease model: one mid and one strong rung of
    # each condition, so every condition is represented. Refine from SI S2.
    "serine_protease": [
        ("grafting, beta=1",        1.0,   30.0),
        ("grafting, beta=1",        1.0,  300.0),
        ("grafting + cooling",      2.0,   60.0),
        ("grafting + cooling",      5.0,  500.0),
        # One rung only for the tight set: with 8 sites the rungs differ least,
        # and it keeps the table at 9 rows.
        ("consensus-8, beta=1",     1.0,  300.0),
        ("consensus-22, beta=1",    1.0,   30.0),
        ("consensus-22, beta=1",    1.0,  300.0),
        ("consensus-22 + cooling",  2.0,   60.0),
        ("consensus-22 + cooling",  5.0,  500.0),
    ],
}
SELECTED: List[Tuple[str, float, float]] = [
    row for row in _SELECTED.get(config.MODEL_NAME, [])
    if row[0] in config.CONDITIONS
]

#: How each column is read. True = higher is better, False = lower is better.
#: Columns absent from this map are identifiers and are never marked.
BETTER_HIGH: Dict[str, bool] = {
    "n_allowed_sites": False,      # a smaller library is cheaper to build
    "hamming_to_wt": False,        # fewer mutations to make
    "subst_to_wt": False,          # point mutations: fewer is cheaper
    "indel_to_wt": False,          # indels are harder to build than substitutions
    "pct_pure_subst": True,        # share of the library with no indel at all
    "Q": False,                    # Q_mu: lower is more plausible
    "P_S": True,
    "log_odds": True,
    "pct_P_S_over_080": True,
    "pct_unique": True,
    "mean_pair_hamming": True,
    "pair_ratio": True,
}


def candidate_pool(condition: str) -> pd.DataFrame:
    """Every run under a condition, scored. Empty if the condition was never run."""
    runs = data.discover_runs(condition)
    if not runs:
        return pd.DataFrame()
    return pd.DataFrame([data.run_metrics(run) for run in runs])


def placeholder_row(condition: str, n: int = 1) -> pd.DataFrame:
    """A row for a condition that was never sampled. NaN, never invented."""
    blank: Dict[str, object] = {key: np.nan for key in BETTER_HIGH}
    blank.update({"condition": condition, "beta": np.nan, "lam": np.nan,
                  "E_DCA": np.nan, "n_chains": 0, "placeholder": True})
    return pd.DataFrame([blank] * n)


def _normalise(table: pd.DataFrame) -> pd.DataFrame:
    """Make ``placeholder`` a real boolean.

    pd.concat fills a column missing from one frame with NaN, and ``bool(nan)``
    is True -- which once labelled every real run a placeholder.
    """
    if "placeholder" not in table:
        table["placeholder"] = False
    table["placeholder"] = table["placeholder"].fillna(False).astype(bool)
    return table


def full_pool() -> pd.DataFrame:
    """Every candidate under every condition, in scan order."""
    frames = []
    for condition in config.CONDITIONS:
        pool = candidate_pool(condition)
        if pool.empty:
            pool = placeholder_row(condition)
        frames.append(pool.sort_values(["beta", "lam"], kind="mergesort"))
    return _normalise(pd.concat(frames, ignore_index=True))


def select_all() -> pd.DataFrame:
    """The hand-picked rows of ``SELECTED``, plus a placeholder per empty condition."""
    pool = full_pool()
    chosen = []
    for condition, beta, lam in SELECTED:
        match = pool[(pool["condition"] == condition)
                     & np.isclose(pool["beta"], beta)
                     & np.isclose(pool["lam"], lam)]
        if match.empty:
            raise SystemExit(
                f"SELECTED names a run that does not exist: {condition!r} "
                f"beta={beta:g}, lam={lam:g}.\nAvailable under that condition:\n"
                + pool[pool["condition"] == condition][["beta", "lam"]]
                .to_string(index=False))
        chosen.append(match.iloc[[0]])
    for condition in config.CONDITIONS:
        if not any(c == condition for c, _, _ in SELECTED):
            chosen.append(placeholder_row(condition))
    table = _normalise(pd.concat(chosen, ignore_index=True))
    table["rank_in_condition"] = table.groupby("condition").cumcount()
    return table


def best_in_column(table: pd.DataFrame, column: str) -> np.ndarray:
    """Boolean mask of the single best cell in ``column``.

    Ties mark every tied row -- claiming one arbitrary winner would be a lie
    about the data. Placeholder rows never win.
    """
    if column not in BETTER_HIGH or column not in table:
        return np.zeros(len(table), dtype=bool)
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
    if "placeholder" in table:
        values = np.where(table["placeholder"].to_numpy(dtype=bool), np.nan, values)
    if np.all(np.isnan(values)):
        return np.zeros(len(table), dtype=bool)
    target = np.nanmax(values) if BETTER_HIGH[column] else np.nanmin(values)
    return np.isclose(values, target, equal_nan=False)


def describe_selection() -> str:
    """Caption line: what the bold means, and that the picking was manual."""
    return ("bold = best value in that column;  "
            "rows are chosen by hand from the full pool (SI), not by a formula")


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 60)
    show = ["condition", "beta", "lam", "n_allowed_sites", "hamming_to_wt", "Q",
            "P_S", "log_odds", "pct_P_S_over_080", "pct_unique",
            "mean_pair_hamming", "pair_ratio"]
    pool = full_pool()
    print("=== FULL POOL ===")
    print(pool[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nbest per column:")
    for column in BETTER_HIGH:
        mask = best_in_column(pool, column)
        if mask.any():
            row = pool[mask].iloc[0]
            print(f"  {column:<20} {'max' if BETTER_HIGH[column] else 'min'}"
                  f" = {row[column]:>9.3f}   "
                  f"{row['condition']}  β={row['beta']:g}, λ={row['lam']:g}")
