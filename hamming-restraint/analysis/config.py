"""Paths, constants and the run grid for the Hamming-restraint study.

Everything model-specific is read from the repository's own calibration, never
re-derived here: mu comes from ``results/ladder/calibration/mu.json`` and the
site sets come from ``analysis/figures_ladder/config.py``. A second copy of
either would be a second thing to keep in step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # hamming-restraint/
REPO = ROOT.parent                      # trypsin-specificity-switch/

# The ladder's figure package: its style module is the project's design system
# and its config holds the site sets. Both are imported, never copied.
LADDER_ANALYSIS = REPO / "analysis" / "figures_ladder"
for path in (str(LADDER_ANALYSIS), str(REPO / "src"), str(REPO / "scripts")):
    if path not in sys.path:
        sys.path.append(path)

RESULTS_ROOT = ROOT / "results"
OUT_DIR = ROOT / "figures"
LADDER_RESULTS = REPO / "results" / "ladder"
MU_JSON = LADDER_RESULTS / "calibration" / "mu.json"

# --- model constants, the same ones the ladder uses -----------------------------
CRITIC_COL = 201
CRITIC_LABEL = "189"
TARGET_RESIDUE = "S"
WT_RESIDUE = "D"
GRAFTING_SEGMENT = (165, 244)

# --- the blocks of phase A ------------------------------------------------------
#: block directory -> (what moves, what it answers)
BLOCKS = {
    "a1_graft": ("grafting window, 79 columns", "lambda x t Pareto against cooling"),
    "a2_free_lam100": ("all 265 columns", "t scan at lambda = 100: susceptibility"),
    "a3_free_lam0": ("all 265 columns", "the matched lambda = 0 control"),
}

#: Where the ladder's own runs live, for the t = 0 and the cooled comparisons.
#: Read through ensembles_serine, which is rescored at beta = 1 -- a cooled run's
#: own P(S) saturates at 1.0 and is not comparable (RESULTS.md section 5).
LADDER_GRAFTING = LADDER_RESULTS / "grafting" / "ensembles_serine"
LADDER_COOLING = {
    1.5: LADDER_RESULTS / "cooling" / "b1.5" / "ensembles_serine",
    2.0: LADDER_RESULTS / "cooling" / "b2" / "ensembles_serine",
    3.0: LADDER_RESULTS / "cooling" / "b3" / "ensembles_serine",
    5.0: LADDER_RESULTS / "cooling" / "b5" / "ensembles_serine",
    10.0: LADDER_RESULTS / "cooling" / "b10" / "ensembles_serine",
}


def mu() -> float:
    """The frozen quality scale: Q = E_DCA - mu * CDE. Calibrated once, on 20,000
    natural sequences, and never refitted."""
    with open(MU_JSON, encoding="utf-8") as handle:
        return float(json.load(handle)["mu"])


def site_sets() -> dict:
    """The candidate site sets, as 0-based MSA columns.

    All but ``all free`` come straight from the ladder's config -- the grafting
    window, the construct Hedstrom, Szilagyi & Rutter actually built, the 20
    columns most strongly coupled to the critic site, and the red sector. The
    red sector is NOT model-derived and carries between models; the coupling set
    is and does not.
    """
    # Loaded BY PATH, under a name of its own. The ladder's module is also called
    # ``config``, and importing it by name would either return this module or
    # evict it from sys.modules -- and style.save resolves OUT_DIR through
    # whatever ``config`` is there, so that would quietly redirect every figure
    # into the ladder's output directory.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ladder_config", LADDER_ANALYSIS / "config.py"
    )
    ladder_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder_config)
    sets = ladder_config.site_sets()
    return {
        "grafting": sets["full segment"],
        "1992 construct": sets["1992 construct"],
        "coupling-20": sets["coupling-20"],
        # 16 columns, NOT 17: the ladder's "red sector" is the sector
        # INTERSECTED WITH the grafting segment, because column 2 (3TGI 18) lies
        # outside it. That intersection is what the design runs actually allowed,
        # and keeping it means every set here is a subset of the segment. The
        # all-sites-free sector analysis scores all 17 and so has a different
        # denominator -- see figures_ladder/fig08 and its README.
        "red sector": sets["red sector"],
        "all free": list(range(265)),
    }


def column_spec(name: str) -> str:
    """A site set as the ``--allowed-cols`` string the runner takes."""
    columns = site_sets()[name]
    return ",".join(str(c) for c in columns)


def ensure_dirs() -> None:
    """Required by ``style.save``, which resolves OUT_DIR through this module."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
