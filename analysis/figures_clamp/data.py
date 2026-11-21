"""Loaders and derived quantities. Every figure reads its data through here.

Two things are centralised on purpose:

* **The critic conditional.** ``P(. | sigma_{\\i*})`` at column 201 does not depend
  on the residue sitting at 201, so scoring a sequence is a leave-one-site-out
  prediction. Both ``P(S|.)`` and ``P(D|.)`` come from one call, because the
  log-odds needs both and ``observables.describe()`` returns only ``P_S``.

* **Run metrics.** The three summary CSVs in the results tree have three different
  schemas and none carries the log-odds, so every table column is recomputed from
  ``final_sequences.fasta``. One code path, one definition of each column.
"""

from __future__ import annotations

import functools
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

import config

config.add_src_to_path()

from encoding import Alphabet, read_sequences, read_single_sequence  # noqa: E402
from observables import (  # noqa: E402
    min_hamming_to_set,
    sample_quality,
    site_probabilities,
)
from potts import PottsModel  # noqa: E402


def _save_npz(path: Path, **arrays) -> None:
    """Write a cache file atomically.

    Several figure scripts may run in parallel and warm the same cache. A plain
    np.savez leaves a half-written file visible under its final name if two
    processes overlap; writing to a unique temporary and renaming does not,
    because rename is atomic within a filesystem.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# --- primitives ----------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def alphabet() -> Alphabet:
    return Alphabet.from_name(config.ALPHABET_NAME)


@functools.lru_cache(maxsize=1)
def model() -> PottsModel:
    config.model_fingerprint()  # fails loudly if the parameter file is missing
    return PottsModel.load(str(config.MODEL_PARAMS), alphabet=alphabet(),
                           device=config.DEVICE, dtype=config.DTYPE, cache=True)


def state(residue: str) -> int:
    return alphabet().index(residue)


S_STATE = lambda: state(config.TARGET_RESIDUE)   # noqa: E731
D_STATE = lambda: state(config.WT_RESIDUE)       # noqa: E731


def encode(text: str) -> np.ndarray:
    return alphabet().encode(text)


@functools.lru_cache(maxsize=8)
def single(path_str: str) -> np.ndarray:
    return encode(read_single_sequence(path_str)[1])


def wild_type() -> np.ndarray:
    return single(str(config.WT_FASTA))


def chymotrypsin_rat() -> np.ndarray:
    return single(str(config.CHYMO_RAT))


def chymotrypsin_bovine() -> np.ndarray:
    return single(str(config.CHYMO_BOVINE))


# --- the natural alignment ------------------------------------------------------
@functools.lru_cache(maxsize=1)
def alignment() -> tuple[List[str], np.ndarray]:
    """``(names, encoded)`` for the whole natural alignment, cached on disk."""
    cache = config.CACHE_DIR / "alignment.npz"
    if cache.is_file():
        blob = np.load(cache, allow_pickle=True)
        return list(blob["names"]), blob["encoded"]
    records = read_sequences(str(config.ALIGNMENT))
    names = [n for n, _ in records]
    encoded = np.asarray([encode(s) for _, s in records])
    config.ensure_dirs()
    _save_npz(cache, names=np.array(names, dtype=object), encoded=encoded)
    return names, encoded


def critic_conditional(sequences: np.ndarray, chunk: int = 512) -> np.ndarray:
    """``P(. | sigma_{\\i*})`` for each sequence, shape ``(N, q)``."""
    sequences = np.atleast_2d(np.asarray(sequences))
    return site_probabilities(model(), sequences, config.CRITIC_COL, chunk=chunk)


@functools.lru_cache(maxsize=1)
def natural_conditional() -> tuple[np.ndarray, np.ndarray]:
    """``(P, residue_at_i_star)`` over the whole alignment. Cached per model."""
    cache = config.cache_path("natural_conditional")
    if cache.is_file():
        blob = np.load(cache, allow_pickle=True)
        return blob["P"], blob["residue"]
    _, encoded = alignment()
    P = critic_conditional(encoded)
    residue = np.array([alphabet().tokens[i] for i in encoded[:, config.CRITIC_COL]])
    _save_npz(cache, P=P, residue=np.array(residue, dtype=object))
    return P, np.asarray(residue)


@functools.lru_cache(maxsize=1)
def split_mask() -> np.ndarray | None:
    """Boolean mask over the alignment: True where a sequence is HELD OUT.

    None when the split files are not available, in which case callers must
    fall back to the whole alignment and say so on the figure. The split is
    never reconstructed or guessed -- an invented split would turn the one
    figure that answers the circularity objection into a false claim.

    Matching is by SEQUENCE, not by name. The split files were written with
    renamed headers (``UPI002B1E7D56_UPI002B1`` against the alignment's
    ``UPI002DBD1C23``), and the alignment's own names are not unique -- 85,069
    distinct names over 89,439 rows -- so name matching both misses rows (328 of
    the test set) and is ambiguous where it hits. The sequences themselves ARE
    unique: all 89,439 rows are distinct, so matching on them is exact, and it
    recovers 17,888 of 17,888.
    """
    if not config.SPLIT_TEST.is_file():
        return None
    names, encoded = alignment()
    tokens = alphabet().tokens
    index = {"".join(tokens[c] for c in row): i for i, row in enumerate(encoded)}
    mask = np.zeros(len(names), dtype=bool)
    missed = 0
    for _, sequence in read_sequences(str(config.SPLIT_TEST)):
        row = index.get(sequence)
        if row is None:
            missed += 1
        else:
            mask[row] = True
    if missed:
        raise SystemExit(
            f"{config.SPLIT_TEST}: {missed:,} test sequences are not present in "
            f"{config.ALIGNMENT}. The split and the alignment do not correspond."
        )
    expected = config.SPLIT_EXPECTED["test"]
    if abs(int(mask.sum()) - expected) > 0.02 * expected:
        raise SystemExit(
            f"{config.SPLIT_TEST} marks {int(mask.sum()):,} of {len(names):,} "
            f"alignment rows, but the model's log says the test split held "
            f"{expected:,}. The file and the alignment do not correspond; "
            "fix the split rather than evaluating on the wrong set."
        )
    return mask


def split_status() -> str:
    """One line for a caption: held-out evaluation, or why not."""
    mask = split_mask()
    if mask is None:
        return ("no held-out split available locally — evaluated on the whole "
                "alignment, so this is a consistency check, not prediction")
    return (f"evaluated on the {int(mask.sum()):,} HELD-OUT sequences the model "
            "never saw during training")


def natural_classes() -> Dict[str, np.ndarray]:
    """Boolean masks over the alignment, by the residue carried at i*."""
    _, residue = natural_conditional()
    return {
        "trypsin": residue == config.WT_RESIDUE,
        "chymotrypsin": residue == config.TARGET_RESIDUE,
        "other": ~np.isin(residue, [config.WT_RESIDUE, config.TARGET_RESIDUE]),
    }


def log_odds(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """``log P(S|.) / P(D|.)``, the discriminant the sampler does NOT steer on."""
    return np.log(P[:, S_STATE()] + eps) - np.log(P[:, D_STATE()] + eps)


# --- the 1992 construct ladder ---------------------------------------------------
def graft(columns: Iterable[int], target: np.ndarray | None = None) -> np.ndarray:
    """The wild type carrying ``target``'s residue at every listed column.

    Gaps are copied like any other state: the 1992 loop swap shortens both loops,
    and skipping gap columns would silently drop the deletions.
    """
    target = chymotrypsin_rat() if target is None else target
    out = np.array(wild_type())
    for column in columns:
        out[column] = target[column]
    return out


def construct_series() -> List[tuple[str, np.ndarray]]:
    """The published ladder, grafted from rat chymotrypsin into rat trypsin.

    RECONSTRUCTED from the column positions the literature implicates -- not the
    sequences reported in the paper. Label it as such wherever it is plotted.
    """
    c = config.CRITIC_COL
    base = [c] + config.LOOP_1 + config.LOOP_2
    return [
        ("wild-type trypsin", np.array(wild_type())),
        ("+ loops 185–188, 221–225", graft(base)),
        ("+ 172", graft(base + [config.POS_172])),
        ("+ 216, 226", graft(base + [config.POS_172, config.POS_216, config.POS_226])),
    ]


# --- designed ensembles -----------------------------------------------------------
_BETA_DIR = re.compile(r"^b([0-9.]+)$")
_LAM_DIR = re.compile(r"^lam([0-9.]+)$")


@dataclass(frozen=True)
class Run:
    """One sampled ensemble: a (condition, beta, lambda) point."""
    condition: str
    path: Path
    beta: float
    lam: float

    @property
    def label(self) -> str:
        return f"β={self.beta:g}, λ={self.lam:g}"


def discover_runs(condition: str) -> List[Run]:
    """Every ``lam*/`` ensemble under a condition, at either nesting depth.

    The cooling runs live at ``<root>/b<beta>/lam<lam>/``; the others at
    ``<root>/lam<lam>/``. CAREFUL: the *serine-substituted* summary.json always
    records ``beta: 1.0``, because substitute_critic.py rescores at beta = 1 --
    the sampling beta only survives in the directory name. Reading beta from the
    JSON would silently collapse the whole cooling scan onto beta = 1.
    """
    root = config.CONDITIONS.get(condition)
    if root is None:
        return []
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Missing ensembles for {condition!r}: {root}\n"
                         "Has STAGE 1 (sampling) been run? See README.md")
    runs: List[Run] = []
    for lam_dir in sorted(root.glob("**/lam*")):
        if not (lam_dir / "final_sequences.fasta").is_file():
            continue
        lam_match = _LAM_DIR.match(lam_dir.name)
        if not lam_match:
            continue
        # Only the serine-substituted copies. The two models nest differently --
        # <cond>/ensembles_serine/b5/lam500 for the first, b5/ensembles_serine/lam500
        # for the second -- and a bare **/lam* glob picks up the UNsubstituted
        # ensembles too, which would silently double every condition.
        # Checked on the FULL path: a condition root can itself sit *inside*
        # ensembles_serine (the first model's consensus runs do), in which case
        # the marker never appears in the path relative to the root.
        if "ensembles_serine" not in lam_dir.parts:
            continue
        parts = lam_dir.relative_to(root).parts
        # beta lives in a b<x> path component under either layout; it is NOT in
        # summary.json, which records the beta = 1 rescoring (see below).
        beta_part = next((p for p in parts if _BETA_DIR.match(p)), None)
        beta = (float(_BETA_DIR.match(beta_part).group(1)) if beta_part
                else _beta_from_json(lam_dir))
        runs.append(Run(condition, lam_dir, beta, float(lam_match.group(1))))
    keep = config.CONDITION_FILTERS.get(condition)
    if keep is not None:
        runs = [run for run in runs if keep(run)]
    config.check_run(root)
    return sorted(runs, key=lambda r: (r.beta, r.lam))


def _beta_from_json(lam_dir: Path) -> float:
    summary = lam_dir / "summary.json"
    if summary.is_file():
        return float(json.loads(summary.read_text()).get("beta", 1.0))
    return 1.0


@functools.lru_cache(maxsize=64)
def run_sequences(path_str: str) -> np.ndarray:
    records = read_sequences(str(Path(path_str) / "final_sequences.fasta"))
    return np.asarray([encode(s) for _, s in records])


def run_frame(run: Run) -> pd.DataFrame:
    """Per-chain observables for one run, with Q and the log-odds added.

    ``observables.csv`` is used where it exists (it already holds E_DCA, CDE and
    the natural-neighbour distance), but P(S), P(D) and the log-odds are always
    recomputed from the sequences so that every figure uses one definition.
    """
    frame_path = run.path / "observables.csv"
    sequences = run_sequences(str(run.path))
    if frame_path.is_file():
        frame = pd.read_csv(frame_path)
    else:
        frame = pd.DataFrame(index=range(len(sequences)))
    P = critic_conditional(sequences)
    frame = frame.iloc[: len(sequences)].reset_index(drop=True)
    frame["P_S"] = P[:, S_STATE()]
    frame["P_D"] = P[:, D_STATE()]
    frame["log_odds"] = log_odds(P)
    if "hamming_to_wt" not in frame:
        frame["hamming_to_wt"] = (sequences != wild_type()[None, :]).sum(axis=1)
    if "Q" not in frame and {"E_DCA", "CDE"} <= set(frame.columns):
        frame["Q"] = sample_quality(frame["E_DCA"].to_numpy(),
                                    frame["CDE"].to_numpy(), config.mu())
    frame["beta"] = run.beta
    frame["lam"] = run.lam
    frame["condition"] = run.condition
    return frame


def trace_path(run: Run) -> Path | None:
    """The sampler's per-step trace for a run, if one was recorded.

    Traces are written next to the *unsubstituted* ensemble, while Run.path
    points at the serine-substituted copy, so the location is recovered from
    summary.json's ``source`` field. A trace exists only where the sampling
    script asked for one: run_analysis.sh passes
    ``--trace-every 1000 --trace-schedule log --trace-lambdas 0 100 1000``,
    and run_site_budget*.sh passes nothing -- which is why the consensus-19
    runs have no traces.
    """
    summary = run.path / "summary.json"
    if summary.is_file():
        source = json.loads(summary.read_text()).get("source")
        if source:
            candidate = Path(source).parent / "trace.csv"
            if candidate.is_file():
                return candidate
    # The ``source`` field is an ABSOLUTE path recorded by the sampler at run
    # time, so it dies the moment the tree is moved, cloned or renamed -- which
    # is exactly what happened in the Nov 2026 reorganisation, and it silently
    # turned SI S6 into five "no trace data" placeholders while 54 perfectly
    # good trace.csv files sat on disk.
    #
    # So fall back to the STRUCTURAL relationship instead of the recorded path:
    # a trace is written beside the unsubstituted ensemble, and the substituted
    # copy differs only by the "ensembles_serine" path component.
    parts = ["ensembles" if p == "ensembles_serine" else p for p in run.path.parts]
    candidate = Path(*parts) / "trace.csv"
    if candidate.is_file():
        return candidate
    candidate = run.path / "trace.csv"
    return candidate if candidate.is_file() else None


def trace_frame(run: Run) -> pd.DataFrame | None:
    """``(step, mean, q25, q75)`` per observable, plus Q, or None if untraced.

    Q is linear in E and CDE, so its MEAN is exact from theirs. Its quartiles
    are not -- the quartile of a sum is not the sum of the quartiles -- so no
    band is offered for Q and the figure says so.
    """
    path = trace_path(run)
    if path is None:
        return None
    frame = pd.read_csv(path)
    if {"E_DCA_mean", "CDE_mean"} <= set(frame.columns):
        frame["Q_mean"] = sample_quality(frame["E_DCA_mean"].to_numpy(dtype=float),
                                         frame["CDE_mean"].to_numpy(dtype=float),
                                         config.mu())
    frame["beta"] = run.beta
    frame["lam"] = run.lam
    frame["condition"] = run.condition
    return frame


def traced_runs(condition: str) -> List[Run]:
    """The runs of a condition that recorded a trace."""
    return [run for run in discover_runs(condition) if trace_path(run) is not None]


def diversity(sequences: np.ndarray) -> tuple[float, float]:
    """``(percent_unique, mean_pairwise_hamming)``.

    Pairwise distance via the one-hot Gram matrix: matches = X Xᵀ, so the Hamming
    distance is L - matches. For 1000 chains that is a 1000x5565 matmul, which is
    far cheaper than materialising the 1000x1000x265 comparison tensor.
    """
    n, length = sequences.shape
    unique = len(np.unique(sequences, axis=0))
    q = model().q
    one_hot = np.zeros((n, length * q), dtype=np.float32)
    rows = np.repeat(np.arange(n), length)
    cols = np.tile(np.arange(length), n) * q + sequences.ravel()
    one_hot[rows, cols] = 1.0
    matches = one_hot @ one_hot.T
    distance = length - matches
    iu = np.triu_indices(n, k=1)
    return 100.0 * unique / n, float(distance[iu].mean())


def run_metrics(run: Run) -> Dict[str, float]:
    """The F5 table row for one run. Every column computed here, from sequences."""
    sequences = run_sequences(str(run.path))
    frame = run_frame(run)
    summary = json.loads((run.path / "summary.json").read_text())
    pct_unique, mean_pair = diversity(sequences)
    mean_hamming = float(frame["hamming_to_wt"].mean())
    # Split the mutation count. --allow-gap-moves makes the gap an ordinary
    # state, so a "mutation" may be a deletion (wild-type residue -> gap) or an
    # insertion (wild-type gap -> residue). Those are not substitutions and an
    # experimentalist cannot build them the same way, so they are counted apart.
    # Indels are a small share of MUTATIONS but can affect a large share of
    # CHAINS -- 60% of the warm grafting library carries one -- which is why
    # pct_pure_subst is reported next to the means.
    gap_state = alphabet().gap_index
    wt = wild_type()
    changed = sequences != wt[None, :]
    is_indel = changed & ((sequences == gap_state) | (wt[None, :] == gap_state))
    indel_per_chain = is_indel.sum(axis=1)
    mean_indel = float(indel_per_chain.mean())
    mean_subst = mean_hamming - mean_indel
    pct_pure = 100.0 * float((indel_per_chain == 0).mean())
    # Raw pairwise distance is not comparable across runs that travelled
    # different distances from the wild type: 3 mutations apart is a tight
    # library at 20 mutations out and a scattered one at 4. The ratio is the
    # spread relative to the distance travelled -- 0 means every chain is the
    # same sequence, ~1 means chains differ from each other about as much as
    # they differ from the wild type.
    pair_ratio = mean_pair / mean_hamming if mean_hamming > 0 else np.nan
    return {
        "condition": run.condition,
        "beta": run.beta,
        "lam": run.lam,
        "n_allowed_sites": int(summary.get("n_allowed_sites", np.nan)),
        "hamming_to_wt": mean_hamming,
        "subst_to_wt": mean_subst,
        "indel_to_wt": mean_indel,
        "pct_pure_subst": pct_pure,
        "pair_ratio": pair_ratio,
        "P_S": float(frame["P_S"].mean()),
        "log_odds": float(frame["log_odds"].mean()),
        "pct_P_S_over_080": 100.0 * float((frame["P_S"] > 0.80).mean()),
        "pct_unique": pct_unique,
        "mean_pair_hamming": mean_pair,
        "Q": float(frame["Q"].mean()) if "Q" in frame else np.nan,
        "E_DCA": float(frame["E_DCA"].mean()) if "E_DCA" in frame else np.nan,
        "n_chains": len(sequences),
    }


# --- the wild type as a reference point -------------------------------------------
@functools.lru_cache(maxsize=1)
def wild_type_reference() -> Dict[str, float]:
    """E, CDE, Q, P(S) and the log-odds of the wild type, for the F5 gate."""
    from observables import context_dependent_entropy, energies
    wt = wild_type()[None, :]
    energy = float(energies(model(), wt)[0])
    cde = float(context_dependent_entropy(model(), wt)[0])
    P = critic_conditional(wt)
    return {
        "E_DCA": energy,
        "CDE": cde,
        "Q": float(sample_quality(np.array([energy]), np.array([cde]), config.mu())[0]),
        "P_S": float(P[0, S_STATE()]),
        "P_D": float(P[0, D_STATE()]),
        "log_odds": float(log_odds(P)[0]),
    }


@functools.lru_cache(maxsize=1)
def natural_observables() -> pd.DataFrame:
    """The calibration sample of natural sequences, with their class at i*.

    ``calibration/natural_observables.csv`` already holds E_DCA, CDE, Q, the
    Hamming distance to the wild type and the natural-neighbour distance for a
    20,000-sequence sample -- the same observables the design ensembles carry,
    so the two can share a pair of axes. The csv has no residue at i*, so the
    class is recovered by matching names back to the alignment.
    """
    if not config.NATURAL_OBSERVABLES.is_file():
        raise SystemExit(f"{config.NATURAL_OBSERVABLES} not found -- run the "
                         "calibration stage of STAGE 1 first")
    frame = pd.read_csv(config.NATURAL_OBSERVABLES)
    names, encoded = alignment()
    residue_by_name: Dict[str, str] = {}
    for name, row in zip(names, encoded[:, config.CRITIC_COL]):
        residue_by_name.setdefault(name, alphabet().tokens[int(row)])
    frame["residue_at_critic"] = frame["name"].map(residue_by_name)
    frame = frame[frame["residue_at_critic"].notna()].copy()
    frame["natural_class"] = np.where(
        frame["residue_at_critic"] == config.TARGET_RESIDUE, "chymotrypsin",
        np.where(frame["residue_at_critic"] == config.WT_RESIDUE, "trypsin", "other"))

    # hamming_to_wt is in the csv only when calibrate_mu.py was given --wt. It is
    # trivial to recompute from the alignment, so do that rather than demand a
    # re-run; the natural-neighbour column is NOT recomputed here because it
    # costs a full 20k x 89k distance sweep, and callers drop it if absent.
    if "hamming_to_wt" not in frame.columns:
        row_of = {}
        for index, name in enumerate(names):
            row_of.setdefault(name, index)
        rows = frame["name"].map(row_of)
        wt = wild_type()
        distances = (encoded[rows.to_numpy()] != wt[None, :]).sum(axis=1)
        frame["hamming_to_wt"] = distances
    return frame


def observables_of(sequences: np.ndarray) -> pd.DataFrame:
    """The full observable set for arbitrary sequences, on the design's axes.

    Same columns as the ensembles and as ``natural_observables()``, so reference
    points can be overlaid on any corner-plot panel.
    """
    from observables import context_dependent_entropy, energies

    sequences = np.atleast_2d(np.asarray(sequences))
    energy = energies(model(), sequences)
    cde = context_dependent_entropy(model(), sequences)
    P = critic_conditional(sequences)
    _, chymo_rows = alignment()
    s_carriers = chymo_rows[chymo_rows[:, config.CRITIC_COL] == S_STATE()]
    distance, _ = nearest_natural(sequences, s_carriers)
    return pd.DataFrame({
        "E_DCA": energy,
        "CDE": cde,
        "Q": sample_quality(energy, cde, config.mu()),
        "hamming_to_wt": (sequences != wild_type()[None, :]).sum(axis=1),
        "P_S": P[:, S_STATE()],
        "P_D": P[:, D_STATE()],
        "log_odds": log_odds(P),
        "min_hamming_natural_S_pct": 100.0 * distance / sequences.shape[1],
    })


@functools.lru_cache(maxsize=1)
def reference_points() -> pd.DataFrame:
    """Wild type and the reconstructed literature series, on the design's axes.

    Row 0 is the wild type; rows 1..3 are the published ladder (loops, then 172,
    then 216 and 226). ``label`` is the short name, ``index`` the number used to
    mark it on a figure.
    """
    series = construct_series()
    frame = observables_of(np.stack([seq for _, seq in series]))
    frame.insert(0, "label", [name for name, _ in series])
    frame.insert(1, "index", np.arange(len(series)) + 1)
    frame.insert(2, "is_wild_type", [True] + [False] * (len(series) - 1))
    return frame


def nearest_natural(sequences: np.ndarray, reference: np.ndarray | None = None,
                    exclude: np.ndarray | None = None,
                    chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """``(distance, index)`` to the closest sequence of ``reference``.

    ``exclude[i]`` is a reference row that sequence ``i`` may not match -- pass a
    query's own index when scoring alignment members against the alignment, or
    every sequence is its own nearest neighbour at distance 0.

    NOTE: ``min_hamming_to_set`` takes ``(sequences, reference_set, device, ...)``
    and does NOT take the model. An earlier version of this wrapper passed
    ``model()`` as the first argument, which pushed the reference array into the
    ``device`` slot and raised deep inside torch.
    """
    if reference is None:
        _, reference = alignment()
    return min_hamming_to_set(sequences, reference, device=config.DEVICE,
                              exclude=exclude, chunk=chunk)


if __name__ == "__main__":
    names, encoded = alignment()
    classes = natural_classes()
    print(f"alignment      : {encoded.shape}")
    for key, mask in classes.items():
        print(f"  {key:<14} {int(mask.sum()):>7,}")
    print(f"wild type      : {wild_type_reference()}")
    for condition in config.CONDITIONS:
        runs = discover_runs(condition)
        print(f"{condition:<26} {len(runs)} runs"
              + ("  [PLACEHOLDER: never sampled]" if not runs else ""))
