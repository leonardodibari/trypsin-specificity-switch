"""Draw every clamp figure into out/. STAGE 2 only -- sampling is run_clamp.sh.

    python work/figures_clamp/make_all.py            # all of them
    python work/figures_clamp/make_all.py c02 c03    # just these
    CAVEATS=1 python work/figures_clamp/make_all.py  # with the text under each

The figures carry a title and nothing else by default, because they are shown
and spoken over. The caveats are still in the scripts and are printed here, so
the thing that must be said out loud next to each figure is never lost.
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback

import config

#: stem -> module. Order is the order they are shown in.
FIGURES = {
    "c01": "c01_clamp_result",
    "c02": "c02_family_gap",
    "c03": "c03_pocket",
}


def main() -> None:
    wanted = [a.lower() for a in sys.argv[1:]] or list(FIGURES)
    unknown = [w for w in wanted if w not in FIGURES]
    if unknown:
        raise SystemExit(f"Unknown figure(s) {unknown}; choose from {list(FIGURES)}")

    print(f"model  : {config.MODEL_PARAMS}")
    print(f"steering: {config.STEERING}  (lambda = 0 everywhere; nothing is optimised)")
    print(f"results: {config.RESULTS_ROOT}")
    print(f"out    : {config.OUT_DIR}\n")

    failures = []
    for key in wanted:
        module_name = FIGURES[key]
        print(f"--- {key}  {module_name}")
        start = time.time()
        try:
            module = importlib.import_module(module_name)
            module.main()
        except SystemExit as error:          # a missing-data guard, not a crash
            print(f"    SKIPPED: {error}")
            failures.append((key, str(error)))
        except Exception:                     # noqa: BLE001 -- report and continue
            traceback.print_exc()
            failures.append((key, "raised"))
        else:
            print(f"    {time.time() - start:.1f}s")

    print()
    if failures:
        # Loud, and a non-zero exit: a half-drawn deck that exits 0 is how a
        # stale figure ends up in a talk.
        print("INCOMPLETE:")
        for key, why in failures:
            print(f"  {key}: {why}")
        raise SystemExit(1)
    print(f"All {len(wanted)} figures written to {config.OUT_DIR}")


if __name__ == "__main__":
    main()
