#!/usr/bin/env bash
# STAGE 1 for the GENERATIVE CHECK of docs/METHODS.md §7.
#
# This is not a design run. lambda = 0, beta = 1, every column mutable, the gap
# an ordinary state: just the model resampling itself. The question is whether
# the sampler reaches the statistics the model was FITTED on, which is a
# property of the model and the sampler, not of the design objective.
#
# THREE STARTS, and the third is the one that makes it an argument:
#
#   wt       chains start at the wild type        far from the family mean
#   natural  chains start at natural sequences    already at the target
#   random   chains start at random sequences     far on the opposite side
#
# A run started at natural sequences can only tell you the chains have not
# wandered off. A run started at the wild type can only tell you they moved.
# Started from opposite sides, agreement at the end is evidence of
# equilibration rather than of a lucky initial condition -- and disagreement is
# evidence the chains are still travelling, which no single run would reveal.
#
# TWO CHOICES THAT MATTER, both forced by how this model was trained
# (docs/TRAINING.md):
#
#   --natural is the TRAIN SPLIT, not the whole alignment. The check asks
#     whether resampling reproduces the statistics the fit targeted; the 17,888
#     held-out sequences were never a target and including them would make the
#     comparison neither a fit check nor a generalisation test.
#
#   NO --weights. The earlier model was fitted with reweighting; this one was
#     trained with `no_reweighting: True`, so its target frequencies are
#     unweighted. Passing a weights file here would score the model against
#     statistics it was never asked to match.
#
# Read every Pearson against the SAMPLING CEILING printed beside it: with a
# finite batch of chains, even a perfect sampler cannot reach 1.0, and for the
# connected correlations the ceiling is far below 1 (~0.82 at 1000 chains).
# A raw C_ij Pearson means nothing without it.
#
# Usage:  bash analysis/run_generative.sh [wt|natural|random]
#         with no argument, all three run in order.
set -euo pipefail

source "$(dirname "$0")/_paths.sh"
OUT=${OUT:-$REPO/results/generative}

# The split the model was fitted on. Not iter_aln_v2.faa -- see above.
NATURAL=${NATURAL:-$DATA_ROOT/DCAmodel-serine_protease/iter_aln_v2.train.fasta}

CRITIC=201
NCHAINS=${NCHAINS:-1000}
# 4,000 sweeps over all 265 columns. A 100-sweep probe reaches C_ij Pearson
# 0.17 of a 0.82 ceiling, so this needs sweeps, not chains; the trace is what
# decides whether 4,000 was enough.
NSTEPS=${NSTEPS:-1060000}
DEVICE=${DEVICE:-mps}
STARTS_DEFAULT=(wt natural random)

if [[ ! -f $NATURAL ]]; then
    echo "MISSING: $NATURAL" >&2
    echo "The train split ships inside the model directory; see docs/DATA.md." >&2
    exit 1
fi

run() { echo; echo "### $*"; "$@"; }

sample() {
    local start=$1
    run "$PY" "$REPO/scripts/run_generative_check.py" \
        --wt "$WT" --model "$PARAMS" --natural "$NATURAL" \
        --output "$OUT/$start" --critic-col $CRITIC --allowed-cols "0-264" \
        --sampler gibbs --start "$start" \
        --n-chains "$NCHAINS" --n-steps "$NSTEPS" \
        --trace-every 8000 --trace-schedule log \
        --seed 0 --device "$DEVICE" --cache-model
}

STARTS=("$@")
[[ ${#STARTS[@]} -eq 0 ]] && STARTS=("${STARTS_DEFAULT[@]}")
for s in "${STARTS[@]}"; do
    echo; echo "================ start: $s ================"
    sample "$s"
done
echo; echo "Done. Results under $OUT"
