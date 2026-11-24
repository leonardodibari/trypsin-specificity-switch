#!/usr/bin/env bash
# THE TWO-START BRACKET -- the replacement for the all-265-columns-free blocks
# of phase A, which did not equilibrate.
#
# WHY THIS REPLACES a2/a3 RATHER THAN TOPPING THEM UP.
# results/EQUILIBRATION_NOTE.md: a free-column pilot at the susceptibility peak,
# sampled at 400,000 steps (3.3x the first pass), STILL failed the drift test on
# hamming_to_wt -- while every energetic observable had settled. The chains were
# not stuck (173 accepted moves per site); d was simply still climbing, and the
# drift fell only 2.1x for a 3.3x longer run. d is not relaxing exponentially
# from a wild-type start, it is DIFFUSING outward from d = 0. Brute force would
# need several million steps per rung, so the fix is not more steps:
#
#   1. anneal in t instead of restarting at the wild type every rung
#      (hamming-restraint/scripts/run_ensemble_annealed.py). <d>(t) is monotone
#      because -d<d>/dt = Var(d) >= 0, so neighbouring rungs are neighbours in
#      <d>: each rung pays only the increment to its neighbour.
#   2. bracket the d axis from BOTH sides, and treat agreement as the verdict.
#      One arm starts at the wild type (d = 0) and anneals DOWN in t, i.e.
#      outward in d. The other starts at bovine chymotrypsinogen B (d = 145) and
#      anneals UP in t, i.e. inward. The two arms approach every rung's
#      equilibrium from opposite directions, so where they agree the answer is
#      equilibrium, and where they disagree the gap IS the error bar. That is a
#      far sharper test than any single-run drift criterion, and it is the thing
#      the main study has never had for a free-column run.
#
# THE FOUR ARMS. lambda = 100 is the biased ensemble; lambda = 0 is the matched
# control, and it is non-negotiable: t < 0 alone drags chains toward the family
# average, and the family carries S at column 201 in 10.4 % of its members, so
# negative t raises P(S) with no bias at all. Both lambdas get both starts.
#
#   c1_lam100_from_wt     lambda=100  start d=0    anneal t descending
#   c1_lam100_from_chymo  lambda=100  start d=145  anneal t ascending
#   c2_lam0_from_wt       lambda=0    start d=0    anneal t descending
#   c2_lam0_from_chymo    lambda=0    start d=145  anneal t ascending
#
# Usage:  bash hamming-restraint/run_bracket.sh [c1_wt|c1_chymo|c2_wt|c2_chymo|post|check]
#         with no argument: all four arms, then post, then check.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
DATA_ROOT=${TRYP_DATA_ROOT:-$(dirname "$REPO")}
PARAMS=${PARAMS:-$DATA_ROOT/DCAmodel-serine_protease/params.dat.gz}
WT=$REPO/data/wt_trypsin.fasta
PY=${PY:-$REPO/.venv/bin/python}

# The far end of the bracket. Bovine chymotrypsinogen B, the aligned row of
# iter_aln_v2: a REAL chymotrypsin, 145 substitutions from the wild type, which
# is inside the 81-196 range the natural chymotrypsins span. Starting there is
# what makes the far arm informative: if the biased ensemble is genuinely
# chymotrypsin-like, this is the side it should be reachable from.
FAR=${FAR:-$REPO/data/chymotrypsin_bovine.fasta}

for f in "$PARAMS" "$WT" "$FAR"; do
    [[ -f $f ]] || { echo "MISSING: $f" >&2; exit 1; }
done

OUT=${OUT:-$HERE/results}
MU=$REPO/results/ladder/calibration/mu.json

CRITIC=201
FREE=0-264
NCHAINS=${NCHAINS:-1000}
DEVICE=${DEVICE:-mps}

# Per RUNG, not per ladder. With annealing a rung only has to cover the step in
# <d> to its neighbour, so this buys far more than the same number spent from a
# wild-type start -- and the two-start gap, not this number, is what says
# whether it was enough.
NSTEPS=${NSTEPS:-150000}

# The grid, centred on the susceptibility peak at t = -0.25 that the first pass
# located (Var(d): 39, 140, 689, 515, 307 at t = -1, -0.5, -0.25, -0.1, 0).
# Identical for all four arms: the control is only a control if its rungs match
# one for one.
T_SCAN=${T_SCAN:-"-1 -0.5 -0.4 -0.3 -0.25 -0.2 -0.15 -0.1 -0.05 0 0.05 0.1 0.25 0.5 1 2"}

# LINEAR, not log: the drift test needs points spread evenly over the second
# half of the run, and a log schedule puts only three or four there.
TRACE=(--trace-every 2000 --trace-schedule linear)

run() { echo; echo "### $*"; "$@"; }

# $1 = output subdir, $2 = lambda, $3 = base seed, $4.. = extra flags
arm() {
    local name=$1 lam=$2 seed=$3; shift 3
    run $PY "$HERE/scripts/run_ensemble_annealed.py" --wt "$WT" --model "$PARAMS" \
        --output "$OUT/$name" --critic-col $CRITIC --allowed-cols "$FREE" \
        --allow-gap-moves --lam "$lam" --t-values $T_SCAN --beta 1 --mu "$MU" \
        --n-chains $NCHAINS --n-steps "$NSTEPS" "${TRACE[@]}" \
        --seed "$seed" --device "$DEVICE" --cache-model "$@"
}

# Seeds differ per arm so that agreement between the two arms is agreement
# between independent samples, not an artefact of a shared random stream.
stage_c1_wt()    { arm c1_lam100_from_wt    100 1000; }
stage_c1_chymo() { arm c1_lam100_from_chymo 100 2000 --start-fasta "$FAR"; }
stage_c2_wt()    { arm c2_lam0_from_wt        0 3000; }
stage_c2_chymo() { arm c2_lam0_from_chymo     0 4000 --start-fasta "$FAR"; }

# Rescore at beta = 1 with S stamped at the critic column -- the door every
# ladder figure reads an ensemble through.
stage_post() {
    for block in $(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name 'c[0-9]_*' -not -name '*_serine' | sort); do
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
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(c1_wt c1_chymo c2_wt c2_chymo post check)
for stage in "${STAGES[@]}"; do
    echo; echo "================ bracket: $stage ================"
    "stage_$stage"
done
echo; echo "Done. Results under $OUT"
