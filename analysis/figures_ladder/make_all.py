"""Build every figure, in talk order. STAGE 2 of the pipeline.

    python make_all.py            # everything
    python make_all.py fig01 fig07   # just these

This is the cheap stage: it only reads ensembles that STAGE 1 already sampled.
If you have pointed config.MODEL_PARAMS at a new model, re-run STAGE 1 FIRST --
otherwise these scripts will happily draw the new model's scores over the old
model's designs. config.check_run refuses to do that, and a stale run stops the
build here rather than producing a plausible, wrong figure.
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback

import config

#: Talk order. The figure number is the slide order, so sorting by filename
#: sorts by the order they are presented in.
SCRIPTS = [
    "fig01_discriminator",
    "fig02_corner_grafting",
    "fig03_cooling",
    "fig04_set_overlap",
    "fig05_table",
    "fig06_corner_selected",
    "fig07_pca",
    "fig08_sector_enrichment",
    "si01_controls",
    "si02_table_full",
    "si03_gaps",
    "si04_logodds_rerank",
    "si05_naturals_vs_designs",
    "si06_equilibration",
    "si07_construct_context",
]


def main(argv: list[str]) -> int:
    wanted = argv or SCRIPTS
    chosen = [s for s in SCRIPTS if any(s.startswith(w) for w in wanted)]
    missing = [w for w in wanted if not any(s.startswith(w) for s in SCRIPTS)]
    if missing:
        print(f"unknown figure(s): {', '.join(missing)}", file=sys.stderr)
        print(f"available: {', '.join(SCRIPTS)}", file=sys.stderr)
        return 2

    print(f"model       : {config.MODEL_PARAMS}")
    print(f"fingerprint : {config.model_fingerprint()}")
    print(f"output      : {config.OUT_DIR}\n")

    failures = []
    for name in chosen:
        start = time.time()
        print(f"--- {name}")
        try:
            module = importlib.import_module(name)
            module.main()
        except SystemExit as exc:          # a loud, deliberate refusal
            failures.append((name, str(exc)))
            print(f"    REFUSED: {exc}", file=sys.stderr)
        except Exception:                  # noqa: BLE001 -- report and continue
            failures.append((name, traceback.format_exc(limit=3)))
            print(f"    FAILED:\n{traceback.format_exc(limit=3)}", file=sys.stderr)
        else:
            print(f"    ok ({time.time() - start:.1f}s)")

    print()
    if failures:
        print(f"{len(failures)} of {len(chosen)} figures did not build:")
        for name, why in failures:
            print(f"  {name}: {why.splitlines()[-1] if why else 'unknown'}")
        return 1
    print(f"all {len(chosen)} figures built into {config.OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
