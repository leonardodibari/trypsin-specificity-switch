#!/usr/bin/env bash
# PHASE A, completion -- everything run_phase_a.sh did not finish.
#
# WHAT WAS MISSING, and why each piece is here:
#
#   fix       postprocess.sh ran against a substitute_critic.py that still
#             parsed a run directory as float(name[3:]). That silently drops
#             EVERY restrained run -- a failure that looks exactly like a run
#             that was never sampled -- so a1_graft_serine/ and
#             a2_free_lam100_serine/ hold only the t = 0 rungs. The parser is
#             fixed (ensemble.parse_run_dirname); this re-runs the rescoring so
#             the _serine trees finally carry the t rungs.
#
#   a2_top    the a2 scan died on its last rung, t = 2, and its grid was too
#             coarse where it matters. <d>(t) from the finished rungs:
#               t    -1    -0.5  -0.25  -0.1    0    0.1   0.25  0.5   1
#               <d>  232   204   160    118    102   88    71    52    36
#               Var   39   140   689    515   307   150   147    55    26
#             Var(d) = -d<d>/dt peaks hard at t = -0.25. That peak is the whole
#             question of test 1, and three points do not locate it. This fills
#             in -0.4 -0.3 -0.2 -0.15 -0.05 0.05 around it, and runs t = 2.
#
#   a3        the matched lambda = 0 control, which run_phase_a.sh never
#             reached. NON-NEGOTIABLE: t < 0 alone walks the chains toward the
#             family average, and the family is 10.4 % serine at column 201, so
#             it raises P(S) with no bias at all. Without this arm nothing in a2
#             can be attributed to lambda. Same design as clamp-vs-relax.
#
# Usage:  bash hamming-restraint/run_phase_a2.sh [fix|a2_top|a3|post|check]
#         with no argument: fix a2_top a3 post check
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../analysis/_paths.sh"

OUT=${OUT:-$HERE/results}
MU=$REPO/results/ladder/calibration/mu.json

CRITIC=201
FREE=0-264

NCHAINS=${NCHAINS:-1000}
NSTEPS_FREE=${NSTEPS_FREE:-120000}
DEVICE=${DEVICE:-mps}

# LINEAR, not log: the drift test needs points spread evenly over the second
# half of the run, and a log schedule puts only three or four there.
TRACE=(--trace-every 1000 --trace-schedule linear)

run() { echo; echo "### $*"; "$@"; }

# $1 = output subdir, $2 = cols, $3 = steps, $4 = lambdas, $5 = t values,
# $6 = base seed. The seed base is offset per block so that a top-up run never
# reuses the seed of a rung already in the directory.
sample() {
    local name=$1 cols=$2 steps=$3 lams=$4 ts=$5 seed=$6
    run $PY "$REPO/scripts/run_ensemble.py" --wt "$WT" --model "$PARAMS" \
        --output "$OUT/$name" --critic-col $CRITIC --allowed-cols "$cols" \
        --allow-gap-moves --lambdas $lams --t-values $ts --beta 1 --mu "$MU" \
        --n-chains $NCHAINS --n-steps "$steps" "${TRACE[@]}" \
        --seed "$seed" --device "$DEVICE" --cache-model
}

# The full scan, as it will stand once a2_top has run. a3 samples all of it in
# one go so that its rungs match a2's one for one.
T_SCAN_FULL=${T_SCAN_FULL:-"-1 -0.5 -0.4 -0.3 -0.25 -0.2 -0.15 -0.1 -0.05 0 0.05 0.1 0.25 0.5 1 2"}
# What a2 still owes: the finished rungs are -1 -0.5 -0.25 -0.1 0 0.1 0.25 0.5 1.
T_SCAN_TOP=${T_SCAN_TOP:-"-0.4 -0.3 -0.2 -0.15 -0.05 0.05 2"}

stage_fix()    { stage_post; }
stage_a2_top() { sample a2_free_lam100 "$FREE" "$NSTEPS_FREE" "100" "$T_SCAN_TOP"  100; }
stage_a3()     { sample a3_free_lam0   "$FREE" "$NSTEPS_FREE" "0"   "$T_SCAN_FULL" 200; }

# Rescore every block at beta = 1 with S stamped at the critic column, which is
# the door every ladder figure reads an ensemble through.
stage_post() {
    for block in $(find "$OUT" -mindepth 1 -maxdepth 1 -type d -not -name '*_serine' | sort); do
        [[ -n $(find "$block" -mindepth 1 -maxdepth 1 -name 'lam*' -print -quit) ]] || continue
        run $PY "$REPO/scripts/substitute_critic.py" \
            --ensemble-dir "$block" --output "${block}_serine" \
            --wt "$WT" --model "$PARAMS" --critic-col $CRITIC --beta 1.0 \
            --device "$DEVICE" --cache-model
    done
}

# Nothing is read off a run that has not saturated on EVERY observable.
stage_check() {
    run $PY "$HERE/analysis/saturation.py" "$OUT" --csv "$OUT/saturation.csv"
}

STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(fix a2_top a3 post check)
for stage in "${STAGES[@]}"; do
    echo; echo "================ phase A completion: $stage ================"
    "stage_$stage"
done
echo; echo "Done. Results under $OUT"
