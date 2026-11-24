#!/usr/bin/env bash
# PHASE B -- the four site sets, under whatever phase A decided about t.
#
# Phase A asks whether the restraint is worth having. Phase B takes the answer
# and applies it across the four candidate mutation sets:
#
#   b1_graft      79 columns, 165-243        the grafting window
#   b2_free       265 columns                everything
#   b3_1992       14 columns                 what Hedstrom, Szilagyi & Rutter
#                                            actually built and assayed
#   b4_coupling   20 columns                 the 20 columns of the window most
#                                            strongly coupled to the critic site
#
# The sets are NOT redefined here. They are read from the ladder's own config
# through hamming-restraint/analysis/config.py, so that there is exactly one
# definition of "the 1992 construct" in the repository. coupling-20 in
# particular is model-DERIVED and would be wrong if copied from an older model.
#
# THE t GRID DEPENDS ON PHASE A, and that is the point of the T_VALUES knob:
#
#   t is useful      T_VALUES="0 0.5 1 2"   -- the restraint is a knob worth
#                    scanning in every set, and the sets are compared along it.
#   t is NOT useful  T_VALUES="0"           -- the four sets are then just four
#                    unrestrained conditions, which is what the project asked
#                    for in that case.
#
# Pass it in rather than editing this file:
#   T_VALUES="0" bash hamming-restraint/run_phase_b.sh
#
# STEP COUNTS ARE PER SET, and they are not a style choice. A run is only usable
# if every observable has saturated (analysis/saturation.py), and how long that
# takes scales with the size of the state space:
#
#   79 columns   60,000 steps   all 20 phase-A runs saturated at this
#   14, 20 cols  60,000 steps   smaller space, so this is comfortable
#   265 columns  400,000 steps  measured, not guessed: the phase-A free runs
#                               relax with tau ~ 19,000-31,000 steps, so the
#                               120,000 of run_phase_a.sh was only 4-6 tau and
#                               SEVEN of its nine rungs failed the drift test.
#                               400,000 is ~16 tau. See docs in the README.
#
# Usage:  bash hamming-restraint/run_phase_b.sh [b1|b2|b3|b4|post|check]
#         with no argument: b1 b3 b4 b2 post check   (b2 last: it is the slow one)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../analysis/_paths.sh"

OUT=${OUT:-$HERE/results}
MU=$REPO/results/ladder/calibration/mu.json

CRITIC=201
NCHAINS=${NCHAINS:-1000}
DEVICE=${DEVICE:-mps}

NSTEPS=${NSTEPS:-60000}
NSTEPS_FREE=${NSTEPS_FREE:-400000}

LAMBDAS=${LAMBDAS:-"10 100 300"}
T_VALUES=${T_VALUES:-"0 0.5 1 2"}

# LINEAR, not log: the drift test needs points spread evenly over the second
# half of the run. 2000 on the long free runs keeps the traced CDE pass -- the
# most expensive thing in a run -- from dominating its cost, while still leaving
# 100 points in the second half.
TRACE=(--trace-every 1000 --trace-schedule linear)
TRACE_FREE=(--trace-every 2000 --trace-schedule linear)

run() { echo; echo "### $*"; "$@"; }

# The site sets, resolved through the one config that defines them.
cols_for() { $PY -c "
import sys; sys.path.insert(0, '$HERE/analysis')
import config
print(config.column_spec('$1'))
"; }

# $1 = output subdir, $2 = allowed cols, $3 = steps, $4.. = trace flags
sample() {
    local name=$1 cols=$2 steps=$3; shift 3
    run $PY "$REPO/scripts/run_ensemble.py" --wt "$WT" --model "$PARAMS" \
        --output "$OUT/$name" --critic-col $CRITIC --allowed-cols "$cols" \
        --allow-gap-moves --lambdas $LAMBDAS --t-values $T_VALUES --beta 1 --mu "$MU" \
        --n-chains $NCHAINS --n-steps "$steps" "$@" \
        --seed 0 --device "$DEVICE" --cache-model
}

stage_b1() { sample b1_graft    "$(cols_for grafting)"       "$NSTEPS"      "${TRACE[@]}"; }
stage_b3() { sample b3_1992     "$(cols_for '1992 construct')" "$NSTEPS"    "${TRACE[@]}"; }
stage_b4() { sample b4_coupling "$(cols_for coupling-20)"    "$NSTEPS"      "${TRACE[@]}"; }
stage_b2() { sample b2_free     "$(cols_for 'all free')"     "$NSTEPS_FREE" "${TRACE_FREE[@]}"; }

# Rescore at beta = 1 with S stamped at the critic column -- the door every
# ladder figure reads an ensemble through, and the only form in which two runs
# sampled at different beta are comparable.
stage_post() {
    for block in $(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name 'b[0-9]_*' -not -name '*_serine' | sort); do
        [[ -n $(find "$block" -mindepth 1 -maxdepth 1 -name 'lam*' -print -quit) ]] || continue
        run $PY "$REPO/scripts/substitute_critic.py" \
            --ensemble-dir "$block" --output "${block}_serine" \
            --wt "$WT" --model "$PARAMS" --critic-col $CRITIC --beta 1.0 \
            --against-residue D --device "$DEVICE" --cache-model
    done
}

# Nothing is read off a run that has not saturated on EVERY observable.
stage_check() {
    run $PY "$HERE/analysis/saturation.py" "$OUT" --csv "$OUT/saturation.csv"
}

STAGES=("$@")
# b2 last: the 265-column runs cost ~7x a windowed one, and the cheap sets
# should be on disk before the expensive one starts.
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(b1 b3 b4 b2 post check)
for stage in "${STAGES[@]}"; do
    echo; echo "================ phase B stage: $stage ================"
    "stage_$stage"
done
echo; echo "Done. Results under $OUT"
