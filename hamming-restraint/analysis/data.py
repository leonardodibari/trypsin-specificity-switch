"""Loading the restraint runs, and the ladder runs they are measured against.

One rule governs everything here: **a number is only loaded from a run that has
saturated on every observable**. :func:`load_block` therefore calls the drift
test of :mod:`saturation` on every run it reads and marks each row with its
verdict; :func:`designs` drops the runs that failed unless it is told otherwise.
Nothing downstream has to remember to check, and a figure cannot silently come
to rest on a chain that was still moving.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

import config
import saturation


def _parse_run_name(name: str) -> tuple[float, float]:
    """``'lam100_t0.5' -> (100.0, 0.5)``; ``'lam10_tm1' -> (10.0, -1.0)``."""
    lam_part, _, t_part = name.partition("_t")
    lam = float(lam_part.removeprefix("lam"))
    if not t_part:
        return lam, 0.0
    return lam, -float(t_part[1:]) if t_part.startswith("m") else float(t_part)


@lru_cache(maxsize=None)
def _verdicts(block_dir: str) -> Dict[str, saturation.RunVerdict]:
    """The saturation verdict of every traced run of a block, by run name."""
    return {v.run: v for v in saturation.check_tree(Path(block_dir))}


def load_block(
    name: str,
    *,
    serine: bool = True,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    """Per-chain observables of every run of one sampled block.

    ``serine=True`` reads ``<block>_serine``, which ``postprocess.sh`` writes:
    the critic column stamped with S and everything rescored at beta = 1. That
    is the form every ladder figure reads, and the only form in which a beta = 1
    run and a cooled run can be compared.

    Columns: ``block, lam, t, chain, E_DCA, hamming_to_wt, P_S, CDE, n_gaps,
    LOR, Q, saturated, stuck``.
    """
    root = Path(root or config.RESULTS_ROOT)
    directory = root / (f"{name}_serine" if serine else name)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{directory} does not exist"
            + ("  -- run hamming-restraint/postprocess.sh" if serine else "")
        )
    # The traces live with the unsubstituted run; the verdict is a property of
    # the chain, not of the rescoring.
    verdicts = _verdicts(str(root / name))

    frames = []
    for path in sorted(directory.glob("lam*/observables.csv")):
        run = path.parent.name
        lam, t = _parse_run_name(run)
        frame = pd.read_csv(path)
        frame["block"] = name
        frame["run"] = run
        frame["lam"] = lam
        frame["t"] = t
        verdict = verdicts.get(run)
        frame["saturated"] = bool(verdict.saturated) if verdict else False
        frame["stuck"] = bool(verdict.stuck) if verdict else False
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No lam*/observables.csv under {directory}")

    out = pd.concat(frames, ignore_index=True)
    out["Q"] = out["E_DCA"] - config.mu() * out["CDE"]
    if "LOR" not in out:
        out["LOR"] = np.nan
    return out


def load_ladder(
    directory: Path, beta: float, *, label: str
) -> pd.DataFrame:
    """One of the ladder's own ensembles, in the same shape as :func:`load_block`.

    These runs predate the restraint, so ``t = 0`` throughout; ``beta`` is what
    varies. They carry no LOR column -- it was added with the restraint -- so it
    is recomputed only where a figure needs it.
    """
    frames = []
    for path in sorted(Path(directory).glob("lam*/observables.csv")):
        frame = pd.read_csv(path)
        frame["block"] = label
        frame["run"] = path.parent.name
        frame["lam"] = float(path.parent.name.removeprefix("lam"))
        frame["t"] = 0.0
        frame["beta"] = beta
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No lam*/observables.csv under {directory}")
    out = pd.concat(frames, ignore_index=True)
    out["Q"] = out["E_DCA"] - config.mu() * out["CDE"]
    # MEASURED, not asserted. This used to read ``out["saturated"] = True`` with
    # a note deferring to the ladder's own SI S6. That is an assertion about runs
    # this subproject then reads numbers off, and it was wrong for two of the 21:
    # the beta = 1 lambda = 1000 run still drifts in E_DCA and hamming_to_wt. The
    # traces are right there in ensembles/, so the gate is run on them like any
    # other run in the study.
    verdicts = _ladder_saturation(Path(directory).parent / "ensembles")
    out["saturated"] = out["run"].map(verdicts).fillna(False)
    out["stuck"] = beta >= 2.0  # CLAMP_CONTROL caveat 1: cold is stuck, not settled
    return out


@lru_cache(maxsize=None)
def _ladder_saturation(sampled: Path) -> Dict[str, bool]:
    """Drift test on each ladder run's own trace, keyed by run directory name.

    The ladder predates the restraint, so its traces carry ``E_DCA``,
    ``hamming_to_wt``, ``P_S`` and ``CDE`` but neither ``Q`` nor ``LOR``.
    ``Q = E_DCA - mu*CDE`` is reconstructed here so the quality the designs are
    actually ranked by is gated too; ``check_observable`` returns ``None`` for
    the columns that are simply absent, so those are skipped rather than
    silently passed.

    THE ONE APPROXIMATION, stated because it is load-bearing. The mean of
    ``Q`` is exactly ``mean(E) - mu*mean(CDE)``, so the DRIFT -- which is what
    the verdict turns on -- is exact. Its quartiles are not the quartiles of the
    difference, so the IQR used as the drift's denominator is a bound (widest
    case: E's q25 against CDE's q75). A wider denominator can only make a run
    look MORE settled, so a run this gate fails has genuinely failed; a run it
    passes marginally on Q alone deserves a second look.
    """
    if not sampled.is_dir():
        return {}
    mu = config.mu()
    out: Dict[str, bool] = {}
    for trace_path in sorted(sampled.glob("lam*/trace.csv")):
        frame = pd.read_csv(trace_path)
        if "CDE_mean" in frame and "E_DCA_mean" in frame:
            frame["Q_mean"] = frame["E_DCA_mean"] - mu * frame["CDE_mean"]
            if "E_DCA_q25" in frame and "CDE_q25" in frame:
                frame["Q_q25"] = frame["E_DCA_q25"] - mu * frame["CDE_q75"]
                frame["Q_q75"] = frame["E_DCA_q75"] - mu * frame["CDE_q25"]
        checks = [saturation.check_observable(frame, name)
                  for name in saturation.OBSERVABLES]
        checks = [c for c in checks if c is not None]
        out[trace_path.parent.name] = bool(checks) and all(c.saturated for c in checks)
    return out


def cooling_ladder() -> pd.DataFrame:
    """Every cooled grafting run of the ladder, the competitor t is measured
    against. beta = 1 is the ladder's own unrestrained grafting run."""
    frames = [load_ladder(config.LADDER_GRAFTING, 1.0, label="ladder beta=1")]
    for beta, directory in config.LADDER_COOLING.items():
        if Path(directory).is_dir():
            frames.append(load_ladder(directory, beta, label=f"ladder beta={beta:g}"))
    return pd.concat(frames, ignore_index=True)


def summarise(frame: pd.DataFrame, by: Iterable[str] = ("block", "lam", "t")) -> pd.DataFrame:
    """One row per run: the ensemble means, the susceptibility and the diversity.

    ``var_hamming`` is the quantity conjugate to t -- ``-d<d>/dt = Var(d)`` -- so
    it is reported next to the mean it is the derivative of, and ``unique_pct``
    is the share of distinct designs, which is the axis cooling destroys.
    """
    by = [c for c in by if c in frame.columns]
    rows = []
    for key, group in frame.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row.update(
            n_chains=len(group),
            E_DCA=group["E_DCA"].mean(),
            Q=group["Q"].mean(),
            CDE=group["CDE"].mean(),
            hamming=group["hamming_to_wt"].mean(),
            var_hamming=group["hamming_to_wt"].var(ddof=1),
            hamming_q25=group["hamming_to_wt"].quantile(0.25),
            hamming_q75=group["hamming_to_wt"].quantile(0.75),
            P_S=group["P_S"].mean(),
            P_S_median=group["P_S"].median(),
            frac_switched=float((group["P_S"] >= 0.80).mean()),
            n_gaps=group["n_gaps"].mean(),
            saturated=bool(group["saturated"].all()),
            stuck=bool(group["stuck"].any()),
        )
        row["LOR"] = group["LOR"].mean() if "LOR" in group else np.nan
        if "beta" in group:
            row["beta"] = group["beta"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def designs(name: str, *, only_saturated: bool = True, **kwargs) -> pd.DataFrame:
    """:func:`load_block`, with the runs that failed the drift test dropped."""
    frame = load_block(name, **kwargs)
    if not only_saturated:
        return frame
    kept = frame[frame["saturated"]]
    dropped = sorted(set(frame.loc[~frame["saturated"], "run"]))
    if dropped:
        print(f"  {name}: dropping {len(dropped)} run(s) that have not saturated: "
              f"{', '.join(dropped)}")
    return kept


def unique_fraction(block: str, run: str, *, root: Optional[Path] = None) -> float:
    """Share of distinct sequences in one ensemble, read from its FASTA.

    The library-size axis: cooling buys quality by collapsing this to 8 %, and
    whether the restraint does the same is the question of h01.
    """
    from encoding import read_sequences

    root = Path(root or config.RESULTS_ROOT)
    path = root / f"{block}_serine" / run / "final_sequences.fasta"
    if not path.is_file():
        path = root / block / run / "final_sequences.fasta"
    sequences = [s for _, s in read_sequences(path)]
    return len(set(sequences)) / len(sequences) if sequences else float("nan")


def trace(block: str, run: str, *, root: Optional[Path] = None) -> pd.DataFrame:
    """The traced observables of one run, for the saturation figure."""
    root = Path(root or config.RESULTS_ROOT)
    return pd.read_csv(root / block / run / "trace.csv")


def verdict(block: str, run: str, *, root: Optional[Path] = None) -> saturation.RunVerdict:
    """The saturation verdict of one run."""
    root = Path(root or config.RESULTS_ROOT)
    return _verdicts(str(root / block))[run]


def runs(block: str, *, root: Optional[Path] = None) -> List[str]:
    """Every run directory of a block, in sampling order."""
    root = Path(root or config.RESULTS_ROOT)
    names = [p.name for p in sorted((root / block).glob("lam*")) if p.is_dir()]
    return sorted(names, key=_parse_run_name)
