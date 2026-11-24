#!/usr/bin/env bash
# PHASE A -- is the Hamming restraint useful at all?
#
# The sampled Hamiltonian gains one term:
#
#     H = beta*E_DCA  -  lambda*log P(S | sigma_\i*)  +  t*d_H(sigma, WT)
#
# d_H is a sum of single-site indicators, so +t*d_H is exactly a uniform field of
# +t on every non-wild-type state: the sampled distribution is still a Potts
# model, the move costs the same, and t is the chemical potential conjugate to
# the Hamming distance. t > 0 holds the chains near wild-type trypsin, t < 0 pays
# them to leave it, t = 0 is the lambda ladder bit for bit.
#
# Three blocks, and between them they answer "is t useful?":
#
#   A1  grafting window, beta = 1, lambda x t
#       The Pareto test. The ladder restrains movement by COOLING (beta), which
#       works by freezing the chain -- acceptance falls from 0.034 to 0.0015 and
#       RESULTS.md cannot tell converged from stuck. t restrains by tilting the
#       equilibrium instead. At matched P(S) and matched <d>, which protocol
#       gives the better Q and the more diverse library?
#
#   A2  all 265 columns free, beta = 1, lambda = 100, t scanned through zero
#       The susceptibility test. -d<d>/dt = Var(d), so a peak in Var(d) along the
#       scan with a bimodal d-histogram means the trypsin island is a genuine
#       basin with a barrier; a smooth ramp means it is only a cost gradient.
#       Either answer is a quantitative statement where RESULTS.md section 6 has
#       a qualitative one.
#
#   A3  the same scan at lambda = 0 -- the matched control
#       The t term alone moves chains away from the wild type and toward the
#       family average, which raises P(S) on its own (the family is 10.4 % serine
#       at column 201). Without this control nothing in A2 can be attributed to
#       the bias. Same design as clamp-vs-relax in CLAMP_CONTROL.md.
#
# Usage:  bash hamming-restraint/run_phase_a.sh [a1|a2|a3|check]
#         with no argument: a1 a2 a3 check
#
# Every run is traced linearly, so the drift test in analysis/saturation.py has
# half a run's worth of points to work with. NOTHING here may be read off a run
# that analysis/saturation.py does not pass on EVERY observable.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../analysis/_paths.sh"

OUT=${OUT:-$HERE/results}
MU=$REPO/results/ladder/calibration/mu.json

CRITIC=201
GRAFT=165-243
FREE=0-264

NCHAINS=${NCHAINS:-1000}
# 60k on the 79-column window is the ladder's own figure, with a >5x margin over
# its slowest measured equilibration. The all-sites-free runs get double, as in
# run_clamp.sh: 265 columns take longer to sweep.
NSTEPS=${NSTEPS:-60000}
NSTEPS_FREE=${NSTEPS_FREE:-120000}
DEVICE=${DEVICE:-mps}

# LINEAR, not log. The log schedule resolves the approach to equilibrium; the
# drift test needs points spread evenly over the SECOND HALF of the run, and a
# log schedule puts only three or four there.
TRACE=(--trace-every 1000 --trace-schedule linear)

run() { echo; echo "### $*"; "$@"; }

# $1 = output subdir, $2 = allowed cols, $3 = steps, $4 = lambdas (quoted),
# $5 = t values (quoted)
sample() {
    local name=$1 cols=$2 steps=$3 lams=$4 ts=$5
    run $PY "$REPO/scripts/run_ensemble.py" --wt "$WT" --model "$PARAMS" \
        --output "$OUT/$name" --critic-col $CRITIC --allowed-cols "$cols" \
        --allow-gap-moves --lambdas $lams --t-values $ts --beta 1 --mu "$MU" \
        --n-chains $NCHAINS --n-steps "$steps" "${TRACE[@]}" \
        --seed 0 --device "$DEVICE" --cache-model
}

# A1: the Pareto grid. lambda spans the ladder's own range; t spans from no
# restraint to one that roughly halves the mutation count.
stage_a1() { sample a1_graft "$GRAFT" "$NSTEPS" "10 30 100 300" "0 0.25 0.5 1 2"; }

# A2 / A3: the t scan through zero, all columns free. The negative rungs are the
# ones that ask whether the designs can leave the trypsins at all.
T_SCAN=${T_SCAN:-"-1 -0.5 -0.25 -0.1 0 0.1 0.25 0.5 1 2"}
stage_a2() { sample a2_free_lam100 "$FREE" "$NSTEPS_FREE" "100" "$T_SCAN"; }
stage_a3() { sample a3_free_lam0   "$FREE" "$NSTEPS_FREE" "0"   "$T_SCAN"; }

# Nothing is read off a run that has not saturated on every observable.
stage_check() {
    run $PY "$HERE/analysis/saturation.py" "$OUT" --csv "$OUT/saturation_phase_a.csv"
}

STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(a1 a2 a3 check)
for stage in "${STAGES[@]}"; do
    echo; echo "================ phase A stage: $stage ================"
    "stage_$stage"
done
echo; echo "Done. Results under $OUT"
