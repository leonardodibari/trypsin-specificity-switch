"""Paths, the grid, and the one definition of each mutation set.

Nothing model-specific is re-derived here. The column sets and ``mu`` are read
through ``hamming-restraint/analysis/config.py``, which reads them from the
ladder's own calibration -- a second copy of "the 1992 construct" or of
``coupling-20`` would be a second thing to keep in step, and ``coupling-20`` is
model-DERIVED, so a stale copy would be silently wrong rather than loudly wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # t_steering/
REPO = ROOT.parent                      # trypsin-specificity-switch/

# The ladder's figure package is the project's design system; the
# hamming-restraint analysis package owns the site sets and the saturation gate.
# Both are imported, never copied.
LADDER_ANALYSIS = REPO / "analysis" / "figures_ladder"
HR_ANALYSIS = REPO / "hamming-restraint" / "analysis"
for path in (str(LADDER_ANALYSIS), str(HR_ANALYSIS), str(REPO / "src"),
             str(REPO / "scripts")):
    if path not in sys.path:
        sys.path.append(path)

RESULTS_ROOT = ROOT / "results"
OUT_DIR = ROOT / "figures"

# --- the grid, exactly as run_steering.sh samples it ----------------------------
#: set name -> (config name of its column set, how many columns, what it is)
SETS = {
    "segment": ("grafting", 79,
                "the grafting window, 165-243 (3TGI 160-228)"),
    # 16 columns, not the sector's 17: the shared config returns the red sector
    # INTERSECTED WITH the grafting segment, because column 2 (3TGI 18) lies
    # outside it. That is what the design runs allowed, and it is also what keeps
    # every set here a subset of the segment -- which is what lets s03_prepare.sh
    # score the natural-neighbour axis over one common window for all of them.
    # The red sector is NOT model-derived and carries between models.
    "redsector": ("red sector", 16,
                  "the red sector, intersected with the grafting segment"),
    "coupling20": ("coupling-20", 20,
                   "the 20 window columns most strongly coupled to the critic"),
}

BETAS = (1.0, 1.5)
LAMBDAS = (100.0, 300.0)
T_VALUES = (0.0, 0.5, 1.0, 2.0)

#: 3 x 2 x 2 x 4. Every one of them must pass the gate before anything is read.
N_RUNS = len(SETS) * len(BETAS) * len(LAMBDAS) * len(T_VALUES)

#: The quantities the gate is applied to. These are the ones a result is read
#: off here, and the drift test is run on each separately -- flattening in one
#: says nothing about the others.
GATED = ("E_DCA", "Q", "hamming_to_wt", "P_S")

#: ONE colour per mutation set, defined here and imported by every figure.
#: Previously s05/s06 ramped a fixed hue whatever the set while s02/s07/s08
#: coloured by set, so a reader comparing a ladder against the factorial saw the
#: same set in two different colours. These are slots 1-3 of the design system's
#: validated palette -- the only three that pass the all-pairs separation gate,
#: which is what a scatter or a corner plot needs (see figures_ladder/style.py).
#: Keyed by set name so adding a set forces a colour decision rather than
#: silently reusing one.
SET_COLOUR = {
    "segment": "#1baf7a",      # style.AQUA
    "redsector": "#2a78d6",    # style.BLUE
    "coupling20": "#eb6834",   # style.ORANGE
}

CRITIC_COL = 201
CRITIC_LABEL = "189"
TARGET_RESIDUE = "S"
WT_RESIDUE = "D"


def _hr_config():
    """The hamming-restraint config, loaded BY PATH under its own module name.

    Not ``import config``: this file is also called ``config`` and sits earlier
    on ``sys.path`` (a script's own directory always does), so a plain import
    would resolve to this module and silently recurse. Loading by path under a
    distinct name is the only form that cannot shadow.
    """
    global _HR_CONFIG
    if _HR_CONFIG is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "hr_config", HR_ANALYSIS / "config.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["hr_config"] = module
        spec.loader.exec_module(module)
        _HR_CONFIG = module
    return _HR_CONFIG


_HR_CONFIG = None


def column_spec(set_name: str) -> str:
    """The allowed-column spec of one set, through the shared definition."""
    return _hr_config().column_spec(SETS[set_name][0])


def mu() -> float:
    """mu of ``Q = E_DCA - mu*CDE``, from the ladder's calibration."""
    return _hr_config().mu()


def natural_quality_sd() -> float:
    """``sd(Q)`` over the natural alignment, from the ladder's own calibration.

    The reference SCALE for "similar to the wild type". Read from
    ``mu.json``'s ``std_quality`` rather than recomputed, so there is one number
    and it is the one mu itself was calibrated against.
    """
    import json

    path = _hr_config().MU_JSON
    return float(json.loads(path.read_text())["std_quality"])


def block_dir(set_name: str, beta: float, *, serine: bool = False) -> Path:
    """``results/<set>/beta<beta>[_serine]``, the directory holding its lam* runs."""
    name = f"beta{beta:g}" + ("_serine" if serine else "")
    return RESULTS_ROOT / set_name / name


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
