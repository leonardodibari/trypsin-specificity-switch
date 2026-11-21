"""Every path, constant and the model fingerprint. Nothing else hard-codes a path.

To point the pipeline at a new model, change ``MODEL_PARAMS`` here and nothing
else -- then re-run STAGE 1 (the sampling scripts) before STAGE 2 (the figures).
See README.md. The fingerprint machinery below exists to make it impossible to
draw a new model's figures from an old model's ensembles.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

# --- roots --------------------------------------------------------------------
# Every path is derived from this file's own location, so the tree works wherever
# it is cloned and nothing has to be edited after `git clone`.
#
# The ONE exception is heavy data -- the 24 MB alignment and the 137 MB parameter
# file -- which is far too large for git and therefore lives OUTSIDE the
# repository. It is found through TRYP_DATA_ROOT, which defaults to the
# repository's parent directory. See docs/DATA.md for what belongs there.
REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TRYP_DATA_ROOT", REPO.parent))
FIGDIR = Path(__file__).resolve().parent
CACHE_DIR = FIGDIR / ".cache"

# --- what the sampler was asked to do -------------------------------------------
# This tree is CLAMP-AND-RELAX, not a steering scheme. The critic column is fixed
# at S and frozen, lambda is 0, and the chains sample H = beta * E_DCA over the
# remaining columns. There is no bias term, so there is nothing for a bias to
# game, and P(S | sigma_\i*) becomes a DIAGNOSTIC rather than a target: it says
# how well the context the model relaxes into supports the serine imposed on it.
#
# This is the zero-parameter null the lambda ladder under work/figures/ should be
# read against. It gets its own results root and its own out/ because the runs
# answer a different question, not the same question differently.
STEERING = "clamp"
STEERING_MATH = (r"$\mathcal{H} = \beta E_{\mathrm{DCA}}$,  "
                 r"$\sigma_{i^*} \equiv \mathrm{S}$")

# --- the model ----------------------------------------------------------------
# Each model gets its own results tree and its own output folder, because a model
# is not just a parameter file: mu, the lambda grid and the beta grid are all
# calibrated against its energy scale, and the designed ensembles belong to it.
# Mixing them is the failure mode the fingerprint check below exists to stop.
#
#   python make_all.py                       # the only model, below
#
# The earlier "pcd_v2" model was dropped in the Nov 2026 cleanup along with its
# results; it had no held-out set and every figure it produced was superseded.
MODELS = {
    "serine_protease": {
        "params": DATA_ROOT / "DCAmodel-serine_protease" / "params.dat.gz",
        "results": REPO / "results" / "clamp",
        "label": "bmDCA serine protease (trained on the 71,551-sequence train split)",
        # Clamp-and-relax layout: <condition>/b<beta>/ensembles_serine/lam0.
        # Same nesting as the cooling runs of the lambda pipeline, so
        # data.discover_runs reads beta from the b<x> path component.
        #
        # There is only ONE lambda (0) in every one of these runs. The scan here
        # is over beta, not over a bias strength, because there is no bias.
        "conditions": {
            # The experiment: S imposed at i*, the grafting window relaxed.
            "clamp, grafting window": ("clamp_graft", "clamp_graft"),
            # S imposed, the WHOLE protein relaxed. The question SI S7 raised:
            # given a serine at 189, does the model pull the rest of the chain
            # toward the chymotrypsins, or does it stay a trypsin?
            "clamp, all sites free": ("clamp_free", "clamp_free"),
            # The control. Same cold sampling, wild-type D left in place. Any
            # shift seen in the two clamped conditions has to be measured
            # against this one, or it is just the effect of lowering beta.
            "no clamp (control)": ("relax_graft", "relax_control"),
            # The control for "all sites free". Cold sampling of the whole chain
            # drifts toward the model's global consensus by itself, so the free
            # clamped run closing 23% of the chymotrypsin gap says nothing until
            # the same relaxation WITHOUT a serine has been measured.
            "no clamp, all sites free": ("relax_free", "relax_free_control"),
        },
    },
}
MODEL_NAME = os.environ.get("FIG_MODEL", "serine_protease")
if MODEL_NAME not in MODELS:
    raise SystemExit(f"Unknown FIG_MODEL={MODEL_NAME!r}; choose one of {list(MODELS)}")
_ACTIVE = MODELS[MODEL_NAME]
MODEL_PARAMS = _ACTIVE["params"]
MODEL_LABEL = _ACTIVE["label"]

# --- sequence data ------------------------------------------------------------
ALIGNMENT = DATA_ROOT / "iter_aln_v2.faa"
WT_FASTA = REPO / "data" / "wt_trypsin.fasta"
CHYMO_RAT = REPO / "data" / "chymotrypsin_rat.fasta"
CHYMO_BOVINE = REPO / "data" / "chymotrypsin_bovine.fasta"
ELASTASE = REPO / "data" / "elastase_pig.fasta"
MAPPING_CSV = REPO / "data" / "mapping_iter_aln_v2.csv"
SECTORS_CSV = REPO / "data" / "sectors_iter_aln_v2.csv"

#: The train/test split the serine-protease model was fitted on, IF it is
#: available locally. The model's own log names
#: ``data/split/iter_aln_v2.{train,test}.fasta`` (71,551 / 17,888 sequences,
#: which together are exactly the 89,439 rows of iter_aln_v2.faa), but those
#: files were produced on the training machine and are not in this repo.
#:
#: When the test split IS present, figure F1 and SI S1 evaluate on held-out
#: sequences and become genuine prediction rather than a consistency check.
#: The split is NEVER guessed: without the file the figures fall back to the
#: whole alignment and say so, because a fabricated split would make the one
#: figure meant to answer the circularity objection quietly dishonest.
SPLIT_DIR = DATA_ROOT / "DCAmodel-serine_protease"
SPLIT_TRAIN = SPLIT_DIR / "iter_aln_v2.train.fasta"
SPLIT_TEST = SPLIT_DIR / "iter_aln_v2.test.fasta"
SPLIT_EXPECTED = {"train": 71551, "test": 17888}

# --- results ------------------------------------------------------------------
RESULTS_ROOT = _ACTIVE["results"]
OUT_DIR = REPO / "figures" / "clamp"
# mu and the natural-sequence observables are properties of the MODEL, not of a
# steering scheme, so they are read from the log P(S) tree rather than
# recalibrated. Recalibrating would give the same number with a different seed
# and make the two pipelines gratuitously incomparable.
CALIBRATION_DIR = REPO / "results" / "ladder" / "calibration"
MU_JSON = CALIBRATION_DIR / "mu.json"
NATURAL_OBSERVABLES = CALIBRATION_DIR / "natural_observables.csv"

# --- model constants ----------------------------------------------------------
CRITIC_COL = 201              # MSA column of the critic site i* (3TGI residue 189)
CRITIC_LABEL = "189"          # its 3TGI label
TARGET_RESIDUE = "S"          # chymotrypsin
WT_RESIDUE = "D"              # trypsin
GRAFTING_SEGMENT = (165, 244)  # [start, stop) -- cols 165..243, residues 160-228
ALPHABET_NAME = "protein"
DEVICE = os.environ.get("FIG_DEVICE", "mps")
DTYPE = "float32"

# --- site sets (0-based MSA columns) -------------------------------------------
# The construct that was actually built and assayed: Hedstrom, Szilagyi & Rutter,
# Science 1992, 255:1249-1253 -- D189S plus surface loops 185-188 and 221-225.
# Insertion-code columns are included because the loop swap changes length
# (188a, 188b, 188A, 223a). Always called "1992 construct", never "Hedstrom 17".
#: Model-DERIVED site sets, recomputed per model. Nothing here may be inherited:
#: the coupling ranking and the design consensus are properties of the model that
#: produced them, and carrying them over from an earlier model would quietly make
#: every downstream figure describe a model we no longer use.
_COUPLING = {
    # The 20 columns of the grafting window with the largest mean |J| to the
    # critic site, computed from each model's own couplings.
    "serine_protease": [165, 175, 176, 184, 186, 194, 195, 197, 200, 202, 204,
                        212, 223, 229, 230, 233, 235, 236, 239, 241],
}
#: Where derive_consensus.py writes this model's own consensus columns.
#: Two sets: the tight one from the cold run, the wide one from the warm
#: lambda = 1000 run (see run_ladder.sh for why they come from different runs).
CONSENSUS_FILE = RESULTS_ROOT / "consensus_columns.txt"
CONSENSUS_LARGE_FILE = RESULTS_ROOT / "consensus_large_columns.txt"
#: Fallback for a model whose consensus was fixed before this machinery existed.
#: Empty now that the only such model (pcd_v2) has been removed.
_CONSENSUS_STATIC: dict[str, list[int]] = {}


def consensus_columns(kind: str = "tight") -> list[int] | None:
    """This model's own consensus set, or None if it has not been derived yet.

    ``kind`` is "tight" (the cold run, >= 95 %) or "wide" (the warm lambda = 1000
    run, >= 50 %). Models that predate the wide set have only the tight one.
    """
    path = CONSENSUS_FILE if kind == "tight" else CONSENSUS_LARGE_FILE
    if path.is_file():
        text = path.read_text().strip()
        if text:
            return [int(c) for c in text.split(",")]
    return _CONSENSUS_STATIC.get(MODEL_NAME) if kind == "tight" else None


def site_sets() -> dict:
    """The candidate mutation sets, with everything model-derived recomputed.

    CAREFUL: the red sector has SEVENTEEN columns in sectors_iter_aln_v2.csv.
    Column 2 (3TGI residue 18) lies outside the grafting segment, so the set
    below is the red sector INTERSECTED WITH the segment -- which is what the
    design runs actually allowed. The all-sites-free sector analysis scores all
    17, so its red-sector denominator differs; see fig08 and the README.
    """
    sets = {
        "full segment": list(range(*GRAFTING_SEGMENT)),
        "red sector": [165, 177, 189, 192, 193, 197, 198, 201, 204, 228, 229,
                       233, 235, 241, 242, 243],
        # Hedstrom, Szilagyi & Rutter, Science 1992, 255:1249-1253: D189S plus
        # surface loops 185-188 and 221-225. Insertion-code columns included
        # because the loop swap changes length. Literature, not model-derived.
        "1992 construct": [194, 195, 196, 197, 198, 199, 200, 201, 235, 236,
                           237, 238, 239, 240],
        "coupling-20": _COUPLING[MODEL_NAME],
    }
    # Both consensus sets, each named by its own derived size, so a changed
    # threshold renames the set instead of silently redefining it.
    for kind in ("tight", "wide"):
        columns = consensus_columns(kind)
        if columns:
            sets[f"consensus-{len(columns)}"] = columns
    return sets


def site_set_order() -> list[str]:
    """Site sets in a stable drawing order: context, literature, then derived."""
    sets = site_sets()
    fixed = ["full segment", "red sector", "1992 construct"]
    consensus = sorted((n for n in sets if n.startswith("consensus-")),
                       key=lambda n: len(sets[n]))
    return [n for n in fixed if n in sets] + consensus + ["coupling-20"]


# The 1992 construct in pieces, for the ladder in F1 / F7.
LOOP_1 = [194, 195, 196, 197, 198, 199, 200]   # 185-188 + insertion codes
LOOP_2 = [235, 236, 237, 238, 239, 240]        # 221-225 + 223a
POS_172, POS_216, POS_226 = 177, 229, 241

# --- the four conditions behind the F5 table ------------------------------------
# Each maps to a directory of lam*/ runs, or None where the run does not exist.
#: Condition name -> ensemble root, or None where that condition was never
#: sampled for this model. The NAMES are stable across models so the palette and
#: every figure can key off them; the size of the consensus set is not baked into
#: the name, because it is derived per model and appears in the tables' "allowed
#: sites" column anyway.
def _consensus_size(path: Path) -> str:
    """Size of a derived consensus set, for naming. '?' until it is derived."""
    if path.is_file():
        text = path.read_text().strip()
        if text:
            return str(len(text.split(",")))
    return "?"


_SIZES = {
    "tight": _consensus_size(RESULTS_ROOT / "consensus_columns.txt"),
    "wide": _consensus_size(RESULTS_ROOT / "consensus_large_columns.txt"),
}

#: Condition display name -> ensemble root (None where never sampled).
#: The consensus names carry the DERIVED size, so they follow the threshold
#: rather than asserting a number that may no longer hold.
CONDITIONS = {}
#: Display name -> stable role, which is what the palette is keyed on.
CONDITION_ROLE = {}
for _name, (_rel, _role) in _ACTIVE["conditions"].items():
    _label = _name.format(**_SIZES)
    CONDITIONS[_label] = RESULTS_ROOT / _rel if _rel else None
    CONDITION_ROLE[_label] = _role

#: Per-condition filters on the discovered runs.
#:
#: Empty here. The lambda pipeline needed one because its cooling condition's
#: beta = 1 rung was byte-identical to the uncooled grafting condition, which
#: would have put the same ensembles in two "different" conditions. The clamp
#: runs have no such overlap: the three conditions differ in WHAT is clamped and
#: which columns are free, not in beta, and every rung is a distinct run.
CONDITION_FILTERS = {}

SECTOR_ENRICHMENT = {
    "grafting segment": RESULTS_ROOT / "grafting" / "sectors" / "sector_enrichment.csv",
    "all sites free": RESULTS_ROOT / "free" / "sectors" / "sector_enrichment.csv",
}


# --- source tree ----------------------------------------------------------------
def add_src_to_path() -> None:
    """Put the project's ``src/`` on sys.path so ``import observables`` works."""
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


# --- the reproducibility contract -----------------------------------------------
_FINGERPRINT_FILE = "_model_fingerprint.json"
_warned: set[str] = set()


def model_fingerprint(path: Path | None = None) -> str:
    """A cheap, stable fingerprint of the parameter file.

    Size plus the sha256 of the first and last 1 MB. Deliberately NOT mtime:
    copying or syncing a file changes mtime without changing the model, and a
    fingerprint that cries wolf gets switched off.
    """
    path = Path(path or MODEL_PARAMS)
    if not path.is_file():
        raise SystemExit(f"Model parameters not found: {path}\nCheck MODEL_PARAMS in config.py")
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode())
    with open(path, "rb") as handle:
        digest.update(handle.read(1 << 20))
        if size > (1 << 21):
            handle.seek(-(1 << 20), os.SEEK_END)
            digest.update(handle.read(1 << 20))
    return digest.hexdigest()[:16]


def check_run(run_dir: Path, *, record: bool = True) -> None:
    """Assert that ``run_dir`` was sampled from the model config.py points at.

    Drawing a new model's figures from an old model's ensembles is the one
    failure mode that produces plausible, wrong figures in silence. So a
    mismatch is fatal, not a warning.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Missing ensemble directory: {run_dir}\n"
                         "Has STAGE 1 (sampling) been run for this model? See README.md")
    stamp = run_dir / _FINGERPRINT_FILE
    current = model_fingerprint()
    if stamp.is_file():
        recorded = json.loads(stamp.read_text()).get("fingerprint")
        if recorded != current:
            raise SystemExit(
                f"STALE ENSEMBLE: {run_dir}\n"
                f"  sampled with model fingerprint {recorded}\n"
                f"  config.py now points at     {current}  ({MODEL_PARAMS})\n"
                "Re-run STAGE 1 (the sampling scripts) for this model, or point\n"
                "MODEL_PARAMS back at the model these ensembles came from."
            )
        return
    key = str(run_dir)
    if key not in _warned:
        _warned.add(key)
        warnings.warn(
            f"{run_dir} carries no model fingerprint; assuming it came from "
            f"{MODEL_PARAMS.name} and recording {current}. If that is wrong, "
            "delete the ensembles and re-run STAGE 1.",
            stacklevel=2,
        )
    if record:
        try:
            stamp.write_text(json.dumps(
                {"fingerprint": current, "model": str(MODEL_PARAMS)}, indent=2))
        except OSError:
            pass


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_path(name: str) -> Path:
    """A cache file keyed by the model fingerprint, so a new model misses cleanly."""
    ensure_dirs()
    return CACHE_DIR / f"{name}.{model_fingerprint()}.npz"


def mu() -> float:
    """The calibrated energy/entropy trade-off, from calibrate_mu.py."""
    if not MU_JSON.is_file():
        raise SystemExit(f"{MU_JSON} not found -- run the calibration stage first")
    return float(json.loads(MU_JSON.read_text())["mu"])


if __name__ == "__main__":
    print(f"model       : {MODEL_PARAMS}")
    print(f"fingerprint : {model_fingerprint()}")
    print(f"mu          : {mu():.4f}")
    print(f"out         : {OUT_DIR}")
