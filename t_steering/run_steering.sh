#!/usr/bin/env bash
# t-STEERING -- the full factorial: 3 mutation sets x 2 beta x 2 lambda x 4 t.
#
# THE QUESTION. Phase A/B asked whether the restraint is worth having and what it
# costs. This asks the applied version: across the three candidate mutation sets,
# which (beta, lambda, t) actually delivers a design an experimentalist would
# build -- few mutations, wild-type-like quality, a diverse batch, high P(S)?
#
# THE SETS are NOT redefined here. They come from the one config that defines
# them (hamming-restraint/analysis/config.py), so "the 1992 construct" and
# "coupling-20" mean here exactly what they mean everywhere else. coupling-20 is
# model-DERIVED and would be silently wrong if copied from another model.
#
#   segment      79 columns, 165-243    the grafting window
#   redsector    16 columns             the red sector INTERSECTED with the
#                                       segment (column 2 / 3TGI 18 lies outside
#                                       it). Not model-derived: it carries
#                                       between models, unlike coupling-20.
#   coupling20   20 columns             most strongly coupled to the critic site
#
# THE GRID, and why these values rather than a wider scan:
#
#   beta   1, 1.5    NOT 2 or above. CLAMP_CONTROL caveat 1: at beta >= 2 the
#                    chain is frozen rather than settled -- the library collapses
#                    to 89 % unique at beta = 2 and 7.9 % at beta = 10, and
#                    RESULTS.md cannot tell a converged cold run from a stuck
#                    one. beta = 1.5 is the coldest rung that is still warm.
#   lambda 100, 300  P(S) ~ 0.93 and ~0.97 on the grafting block. lambda = 10
#                    reaches only 0.74, which fails the "high P(S)" requirement
#                    before the restraint is even applied.
#   t      0, 0.5,   phase B's own grid. Every one of the 20 phase-A runs on the
#          1, 2      79-column block saturated on it, and it spans <d> ~ 17 -> 8
#                    there, which is the range a construct is commissioned in.
#
# STEPS. 60,000 for all three sets: that is the figure all 20 phase-A runs on the
# 79-column window saturated at, and the 14- and 20-column sets are a smaller
# state space, so it is comfortable there. No set here is the 265-column one, so
# the 400,000-step figure of run_phase_b.sh does not apply.
#
# NOTHING IS READ OFF A RUN THAT HAS NOT SATURATED. stage_check gates all 48 on
# E_DCA, Q, hamming_to_wt and P_S; the analysis refuses to proceed otherwise.
#
# Usage:  bash t_steering/run_steering.sh [segment|hedstrom14|coupling20|post|check]
#         with no argument: all three sets, then post, then check.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../analysis/_paths.sh"

OUT=${OUT:-$HERE/results}
MU=$REPO/results/ladder/calibration/mu.json
HR=$REPO/hamming-restraint/analysis

CRITIC=201
NCHAINS=${NCHAINS:-1000}
NSTEPS=${NSTEPS:-60000}
DEVICE=${DEVICE:-mps}

BETAS=${BETAS:-"1 1.5"}
LAMBDAS=${LAMBDAS:-"100 300"}
T_VALUES=${T_VALUES:-"0 0.5 1 2"}

# LINEAR, not log: the drift test needs points spread evenly over the SECOND HALF
# of the run, and a log schedule puts only three or four there.
TRACE=(--trace-every 1000 --trace-schedule linear)

run() { echo; echo "### $*"; "$@"; }

cols_for() { $PY -c "
import sys; sys.path.insert(0, '$HR')
import config
print(config.column_spec('$1'))
"; }

# $1 = set name, $2 = the config name of its column set.
# One invocation per (set, beta): run_ensemble.py loops the lambda x t grid
# internally, so the 137 MB parameter file is loaded once per eight runs rather
# than once per run. beta has to be a bash loop because it is a scalar flag.
sample_set() {
    local name=$1 spec=$2 cols
    cols=$(cols_for "$spec")
    for beta in $BETAS; do
        run $PY "$REPO/scripts/run_ensemble.py" --wt "$WT" --model "$PARAMS" \
            --output "$OUT/${name}/beta${beta}" --critic-col $CRITIC \
            --allowed-cols "$cols" --allow-gap-moves \
            --lambdas $LAMBDAS --t-values $T_VALUES --beta "$beta" --mu "$MU" \
            --n-chains $NCHAINS --n-steps "$NSTEPS" "${TRACE[@]}" \
            --seed 0 --device "$DEVICE" --cache-model
    done
}

stage_segment()    { sample_set segment    "grafting"; }
stage_redsector()  { sample_set redsector  "red sector"; }
stage_coupling20() { sample_set coupling20 "coupling-20"; }

# Rescore at beta = 1 with S stamped at the critic column. This is the ONLY form
# in which a beta = 1 and a beta = 1.5 run are comparable: run_ensemble scores
# each ensemble at its own beta, and a cooled run's P(S) is inflated there
# (RESULTS.md section 5). Every figure reads the _serine copy.
stage_post() {
    for block in $(find "$OUT" -mindepth 2 -maxdepth 2 -type d -name 'beta*' -not -name '*_serine' | sort); do
        [[ -n $(find "$block" -mindepth 1 -maxdepth 1 -name 'lam*' -print -quit) ]] || continue
        run $PY "$REPO/scripts/substitute_critic.py" \
            --ensemble-dir "$block" --output "${block}_serine" \
            --wt "$WT" --model "$PARAMS" --critic-col $CRITIC --beta 1.0 \
            --against-residue D --device "$DEVICE" --cache-model
    done
}

# The gate. saturation.py takes a directory holding lam* runs, which here is each
# (set, beta) block, so it is run per block and the verdicts are concatenated.
stage_check() {
    for block in $(find "$OUT" -mindepth 2 -maxdepth 2 -type d -name 'beta*' -not -name '*_serine' | sort); do
        local tag
        tag=$(basename "$(dirname "$block")")_$(basename "$block")
        run $PY "$HR/saturation.py" "$block" --csv "$OUT/saturation_${tag}.csv" || true
    done
}

STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(redsector coupling20 segment post check)
for stage in "${STAGES[@]}"; do
    echo; echo "================ t-steering: $stage ================"
    "stage_$stage"
done
echo; echo "Done. Results under $OUT"
