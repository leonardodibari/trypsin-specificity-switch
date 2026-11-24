"""Loading, with the gate attached.

One rule governs this package, the same one the rest of the study runs on: **a
number is only read off a run in which every quantity it is read off has
saturated**. Here that means ``E_DCA``, ``Q``, ``hamming_to_wt`` and ``P_S``
(``config.GATED``), each tested separately by the drift test in
``hamming-restraint/analysis/saturation.py``.

Everything is read through the ``_serine`` copy -- the critic column stamped
with S and rescored at ``beta = 1``. That is not cosmetic here: this grid varies
``beta``, and ``run_ensemble`` scores each ensemble at its own ``beta``, where a
cooled run's ``P(S)`` is inflated (``docs/RESULTS.md`` section 5). The rescored
copy is the only form in which a ``beta = 1`` and a ``beta = 1.5`` run are
comparable at all.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

import config
import saturation
from encoding import Alphabet


@lru_cache(maxsize=None)
def _verdicts(block_dir: str) -> Dict[str, "saturation.RunVerdict"]:
    """Saturation verdict of every traced run of one block, by run name.

    Read from the SAMPLED block, never the rescored one: the trace is a property
    of the chain, and ``substitute_critic`` does not write one.
    """
    return {v.run: v for v in saturation.check_tree(Path(block_dir))}


def gated_verdict(verdict: "saturation.RunVerdict") -> tuple[bool, List[str]]:
    """``(passes, which of config.GATED failed)`` for one run.

    ``saturation.py``'s own verdict covers every observable it can find, which
    is a superset of what is read off here. This narrows it to the four
    quantities this study quotes, so that a run dropped for drifting ``CDE``
    alone is distinguishable from one drifting in ``Q``.
    """
    failed = [
        o.name for o in verdict.observables
        if o.name in config.GATED and not o.saturated
    ]
    return (not failed), failed


def load_run(
    set_name: str, beta: float, run: str, *, serine: bool = True
) -> pd.DataFrame:
    """Per-chain observables of one run, with its grid coordinates attached."""
    from ensemble import parse_run_dirname_full

    directory = config.block_dir(set_name, beta, serine=serine) / run
    frame = pd.read_csv(directory / "observables.csv")
    lam, t_sub, t_indel = parse_run_dirname_full(run)
    frame["set"] = set_name
    frame["beta"] = beta
    frame["lam"] = lam
    frame["t"] = t_sub
    frame["t_indel"] = t_indel
    frame["run"] = run
    frame["n_cols"] = config.SETS[set_name][1]
    if "Q" not in frame:
        frame["Q"] = frame["E_DCA"] - config.mu() * frame["CDE"]
    return frame


def load_all(*, serine: bool = True, only_saturated: bool = True,
             grid_only: bool = True,
             betas: Optional[Iterable[float]] = None) -> pd.DataFrame:
    """Every run of the grid that exists on disk, one row per chain.

    With ``only_saturated`` the runs that fail the gate are dropped and named.
    Nothing in this package calls it with ``False`` except the gate report
    itself, which has to see the failures in order to report them.

    ``betas`` defaults to ``config.BETAS``, the factorial's own two. A ladder
    sampled at some other beta -- ``beta = 3``, to measure what cooling buys --
    lives in a block this loader would otherwise never look in, so ``s05`` passes
    its own beta explicitly. Keeping the default at ``config.BETAS`` is what
    stops such a block from leaking into the factorial.

    ``grid_only`` keeps the FACTORIAL exact: only the ``t`` values declared in
    ``config.T_VALUES`` are loaded. The ladder of ``s05`` samples an extra rung
    into the same block, and without this filter that rung would silently become
    a 49th cell of a 48-cell design -- every marginal in ``s02`` would then be a
    mean over an unbalanced grid. ``s05`` is the one caller that passes
    ``grid_only=False``, and it selects its own rungs by hand.
    """
    frames, dropped = [], []
    for set_name in config.SETS:
        for beta in (config.BETAS if betas is None else betas):
            sampled = config.block_dir(set_name, beta)
            if not sampled.is_dir():
                continue
            verdicts = _verdicts(str(sampled))
            source = config.block_dir(set_name, beta, serine=serine)
            if not source.is_dir():
                continue
            for path in sorted(source.glob("lam*/observables.csv")):
                run = path.parent.name
                if grid_only:
                    from ensemble import parse_run_dirname_full

                    _, t_sub, _ = parse_run_dirname_full(run)
                    if not any(np.isclose(t_sub, v) for v in config.T_VALUES):
                        continue
                verdict = verdicts.get(run)
                passes, failed = (True, [])
                if verdict is not None:
                    passes, failed = gated_verdict(verdict)
                elif only_saturated:
                    # No trace means the gate cannot be applied. That is a
                    # failure, not a pass: an ungated run is exactly what this
                    # package exists to refuse.
                    passes, failed = False, ["no trace"]
                frame = load_run(set_name, beta, run, serine=serine)
                frame["saturated"] = passes
                frame["failed"] = ",".join(failed) or "-"
                if verdict is not None:
                    frame["accepted_per_site"] = verdict.accepted_per_site
                if only_saturated and not passes:
                    dropped.append(f"{set_name}/beta{beta:g}/{run} ({','.join(failed)})")
                    continue
                frames.append(frame)
    if dropped:
        print(f"  GATE: dropping {len(dropped)} run(s) that failed on "
              f"{'/'.join(config.GATED)}:")
        for name in dropped:
            print(f"    - {name}")
    if not frames:
        raise FileNotFoundError(
            f"no runs under {config.RESULTS_ROOT} -- run t_steering/run_steering.sh"
        )
    return pd.concat(frames, ignore_index=True)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per run: the ensemble means and the two spread measures.

    ``var_hamming`` sits next to the mean it is the derivative of
    (``-d<d>/dt = Var(d)``), and ``unique_pct`` is the library size, which is
    the axis cooling destroys and the reason beta >= 2 is not in this grid.
    """
    rows = []
    keys = ["set", "beta", "lam", "t"]
    for key, group in frame.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row.update(
            run=group["run"].iloc[0],
            n_cols=int(group["n_cols"].iloc[0]),
            n_chains=len(group),
            E_DCA=group["E_DCA"].mean(),
            Q=group["Q"].mean(),
            CDE=group["CDE"].mean(),
            hamming=group["hamming_to_wt"].mean(),
            hamming_sd=group["hamming_to_wt"].std(ddof=1),
            var_hamming=group["hamming_to_wt"].var(ddof=1),
            hamming_q25=group["hamming_to_wt"].quantile(0.25),
            hamming_q75=group["hamming_to_wt"].quantile(0.75),
            P_S=group["P_S"].mean(),
            P_S_median=group["P_S"].median(),
            frac_switched=float((group["P_S"] >= 0.80).mean()),
            n_gaps=group["n_gaps"].mean(),
            unique_pct=100 * unique_fraction(
                group["set"].iloc[0], float(group["beta"].iloc[0]),
                group["run"].iloc[0]
            ),
        )
        n_unique, pairwise, ratio = batch_diversity(
            group["set"].iloc[0], float(group["beta"].iloc[0]), group["run"].iloc[0]
        )
        row.update(n_unique=n_unique, pairwise_hamming=pairwise,
                   divergence_ratio=ratio)
        row["LOR"] = group["LOR"].mean() if "LOR" in group else np.nan
        if "accepted_per_site" in group:
            row["accepted_per_site"] = group["accepted_per_site"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys)


@lru_cache(maxsize=None)
def batch_diversity(set_name: str, beta: float, run: str) -> tuple[int, float, float]:
    """``(n_unique, mean_pairwise_hamming, divergence_ratio)`` of one ensemble.

    ``mean_pairwise_hamming`` is the mean Hamming distance between two DISTINCT
    chains of the batch -- how different the designs are from EACH OTHER, as
    opposed to from the wild type. Computed exactly from the per-column state
    counts rather than by comparing 500,000 pairs: for column i with counts
    ``n_ia`` over ``N`` chains, the number of unordered disagreeing pairs is
    ``(N^2 - sum_a n_ia^2) / 2``, so

        mean pairwise = sum_i (N^2 - sum_a n_ia^2) / (N (N-1))

    which is O(L*q) and has no sampling error of its own.

    ``divergence_ratio`` is that divided by the mean distance to the wild type.
    It is the scale-free form, and the one worth thresholding: it asks whether
    the chains differ from one another by a decent fraction of how far they have
    travelled. A batch that found one answer and stayed there has ratio ~0
    however far from the wild type it sits; a batch drawn independently from a
    broad distribution approaches ~2 (two chains differ at a site roughly twice
    as often as either differs from a fixed reference, when the reference state
    is not itself dominant).
    """
    from encoding import Alphabet, read_sequences

    alphabet = Alphabet.from_name("protein")
    for serine in (True, False):
        path = config.block_dir(set_name, beta, serine=serine) / run / "final_sequences.fasta"
        if not path.is_file():
            continue
        sequences = [s for _, s in read_sequences(path)]
        if not sequences:
            break
        encoded = np.stack([alphabet.encode(s) for s in sequences])
        n, length = encoded.shape
        if n < 2:
            break
        counts = np.zeros((length, alphabet.q), dtype=np.int64)
        for state in range(alphabet.q):
            counts[:, state] = (encoded == state).sum(axis=0)
        disagreeing = (n * n - (counts.astype(np.float64) ** 2).sum(axis=1))
        pairwise = float(disagreeing.sum() / (n * (n - 1)))
        reference = wild_type()
        to_wt = float((encoded != reference[None, :]).sum(axis=1).mean())
        ratio = pairwise / to_wt if to_wt > 0 else float("nan")
        return len(set(sequences)), pairwise, ratio
    return 0, float("nan"), float("nan")


@lru_cache(maxsize=None)
def unique_fraction(set_name: str, beta: float, run: str) -> float:
    """Share of distinct sequences in one ensemble, read from its FASTA.

    The library-size axis. A batch of 1000 chains that collapsed onto three
    sequences is not a batch anyone can screen, and no summary of means shows it.
    """
    from encoding import read_sequences

    for serine in (True, False):
        path = config.block_dir(set_name, beta, serine=serine) / run / "final_sequences.fasta"
        if path.is_file():
            sequences = [s for _, s in read_sequences(path)]
            return len(set(sequences)) / len(sequences) if sequences else float("nan")
    return float("nan")


def trace(set_name: str, beta: float, run: str) -> pd.DataFrame:
    """The traced observables of one run, for the gate figure."""
    return pd.read_csv(config.block_dir(set_name, beta) / run / "trace.csv")


@lru_cache(maxsize=1)
def model():
    """The fitted Potts model, loaded once.

    ``str(path)`` and ``cache=True`` both matter: the adabmDCA loader keys its
    gzip detection off the path's suffix as a STRING, so handing it a ``Path``
    makes it read ``params.dat.gz`` as utf-8 text and fail on the gzip magic
    byte. This mirrors ``figures_ladder/data.py:model()`` rather than inventing
    a second way of loading the same file.
    """
    from potts import PottsModel

    path = config.REPO.parent / "DCAmodel-serine_protease" / "params.dat.gz"
    if not path.is_file():
        raise SystemExit(f"Model parameters not found: {path}  (see docs/DATA.md)")
    return PottsModel.load(str(path), alphabet=Alphabet.from_name("protein"),
                           dtype="float32", cache=True)


def wild_type() -> np.ndarray:
    """The encoded wild type, for the reference lines every figure carries."""
    from encoding import Alphabet, read_single_sequence

    _, string = read_single_sequence(config.REPO / "data" / "wt_trypsin.fasta")
    return Alphabet.from_name("protein").encode(string)
