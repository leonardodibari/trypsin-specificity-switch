#!/usr/bin/env bash
# STAGE 1 for CLAMP-AND-RELAX on the serine-protease model.
#
# This is not a steering scheme. There is no lambda. The critic column is set to
# S and frozen, and the chains sample H = beta * E_DCA over the rest:
#
#     sigma_{i*} = S  (imposed, immutable)
#     H = beta * E_DCA
#
# It is the zero-parameter null the whole lambda ladder should be measured
# against, and it is the one protocol in this project with nothing in it for an
# objective to game -- which is exactly why it is worth running. The log-odds
# pilot (analysis/figures_clamp/README.md) showed what a gameable objective does:
# steering on log P(S)/P(D) turned the pocket into GLYCINE, not serine.
#
# THREE CONDITIONS, and the third is not optional:
#
#   A  clamp_graft   S clamped, cols 165-243 free     the experiment
#   B  clamp_free    S clamped, ALL columns free      does the model relax the
#                                                     WHOLE protein toward the
#                                                     chymotrypsins? (the
#                                                     82-residue gap of SI S7)
#   C  relax_graft   NO clamp (D stays), cols 165-243 free     the control
#
# Without C, any shift in the relaxed context could just be the effect of
# lowering the temperature. C is the same cold sampling with the wild type's D
# left in place, so the difference B/A minus C is what the serine did.
#
# Usage:  bash analysis/run_clamp.sh [clamp_graft|clamp_free|relax_graft|relax_free]
#         with no argument, all four run in order.
set -euo pipefail

source "$(dirname "$0")/_paths.sh"
OUT=${OUT:-$REPO/results/clamp}

CRITIC=201
GRAFT=165-243

NCHAINS=${NCHAINS:-1000}
# 60,000 for the grafting window, as in run_ladder.sh: the equilibration tests
# on this model put the slowest case at ~11,400 steps. The all-sites-free runs
# get double, because a 265-column window needs proportionally more single-site
# moves to mix -- that is a step-count argument, not a physics one, and SI
# equilibration re-checks it from the traces rather than taking it on trust.
NSTEPS=${NSTEPS:-60000}
NSTEPS_FREE=${NSTEPS_FREE:-120000}
DEVICE=${DEVICE:-mps}

TRACE=(--trace-every 1000 --trace-schedule log)

run() { echo; echo "### $*"; "$@"; }

# $1 = output subdir, $2 = allowed cols, $3 = beta, $4 = steps, $5 = clamp flag
sample() {
    local name=$1 cols=$2 beta=$3 steps=$4 clamp=$5
    # bash 3.2 (macOS) treats "${extra[@]}" on an EMPTY array as an unbound
    # variable under set -u, so the control condition died on expansion rather
    # than on anything real. ${extra[@]+...} is the portable guard.
    local extra=()
    [[ $clamp == clamp ]] && extra+=(--clamp-critic)
    local extra_safe=("${extra[@]+"${extra[@]}"}")

    run $PY "$REPO/scripts/run_ensemble.py" --wt "$WT" --model "$PARAMS" \
        --output "$OUT/$name/ensembles" --critic-col $CRITIC --allowed-cols "$cols" \
        --allow-gap-moves "${extra_safe[@]+"${extra_safe[@]}"}" --lambdas 0 --beta "$beta" \
        --n-chains $NCHAINS --n-steps "$steps" "${TRACE[@]}" \
        --seed 0 --device "$DEVICE" --cache-model

    # substitute_critic rescores at beta = 1 and writes ensembles_serine/, which
    # is the layout every figure script discovers. It substitutes S at the critic
    # -- already true under the clamp, so for A and B it is a rescore and nothing
    # more. For the control C it is what makes the comparison like-for-like:
    # without it, C would be scored with a D where A and B have an S.
    run $PY "$REPO/scripts/substitute_critic.py" \
        --ensemble-dir "$OUT/$name/ensembles" --output "$OUT/$name/ensembles_serine" \
        --wt "$WT" --model "$PARAMS" --critic-col $CRITIC --beta 1.0 \
        --device "$DEVICE" --cache-model

    run $PY "$REPO/scripts/add_natural_neighbour.py" \
        --ensemble-dir "$OUT/$name/ensembles_serine" --natural "$ALN" \
        --critic-col $CRITIC --allowed-cols "$cols" --wt "$WT" \
        --as-percentage --device "$DEVICE"
}

# A -- the experiment. Five rungs: beta = 1 is the model's own equilibrium in
# that window, and cooling asks how much of the relaxed context survives when
# the chains are pushed toward the energy minimum.
BETAS=${BETAS:-"1 1.5 2 3 5"}
# B and C get the short ladder; they are reference points, not a scan.
BETAS_REF=${BETAS_REF:-"1 2 5"}

stage_clamp_graft() {
    for beta in $BETAS; do sample "clamp_graft/b$beta" "$GRAFT" "$beta" $NSTEPS clamp; done
}

stage_clamp_free() {
    # "all columns" is written out rather than passed as a range so the figure
    # scripts record the same allowed_cols they would for any other condition.
    local length; length=$($PY -c "
import sys; sys.path.insert(0, '$REPO/src')
from encoding import read_single_sequence
print(len(read_single_sequence('$WT')[1]))")
    for beta in $BETAS_REF; do
        sample "clamp_free/b$beta" "0-$((length - 1))" "$beta" $NSTEPS_FREE clamp
    done
}

# D -- the control for B, and it is not optional either. Cold sampling of the
# WHOLE chain moves toward the model's global consensus on its own, so
# clamp_free closing 23% of the gap means nothing until the same free relaxation
# without a serine has been measured. This stage was added after clamp_free
# produced that number; it is the control that says whether the serine caused it.
stage_relax_free() {
    local length; length=$($PY -c "
import sys; sys.path.insert(0, '$REPO/src')
from encoding import read_single_sequence
print(len(read_single_sequence('$WT')[1]))")
    for beta in $BETAS_REF; do
        sample "relax_free/b$beta" "0-$((length - 1))" "$beta" $NSTEPS_FREE free
    done
}

stage_relax_graft() {
    for beta in $BETAS_REF; do sample "relax_graft/b$beta" "$GRAFT" "$beta" $NSTEPS free; done
}

STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(clamp_graft relax_graft clamp_free relax_free)
for stage in "${STAGES[@]}"; do
    echo; echo "================ stage: $stage ================"
    "stage_$stage"
done
echo; echo "Done. Results under $OUT"
