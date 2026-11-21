#!/usr/bin/env bash
# STAGE 1 for the serine-protease model: every design ensemble the deck needs.
#
# Four conditions, all traced, all rescored at beta = 1:
#   A  grafting, beta = 1        cols 165-243, lambda ladder
#   B  grafting + cooling        cols 165-243, beta ladder at fixed lam_tilde
#   C  consensus-19, beta = 1    19 cols, lambda ladder
#   D  consensus-19 + cooling    19 cols, beta ladder     <- new; never run before
#
# WHY THE GRIDS ARE WHAT THEY ARE
# They were shifted down from the ladders used on the project's earlier model
# (deleted in the Nov 2026 cleanup, along with the drivers that scanned it).
# The serine-protease model has roughly half the energy scale of the earlier one
# (wild-type E 82.9 vs 160.6, CDE 0.445 vs 0.795), so the bias wins much more
# easily: lambda = 10 already reaches P(S) = 0.75 where the old model needed 100.
# mu is recalibrated (422.62 vs 331.42) and the lambda and lam_tilde ladders are
# shifted down accordingly. See analysis/figures_ladder/README.md.
#
# lam_tilde = lambda / beta is what is optimised; beta says only how cold.
#
# Usage:  bash analysis/run_ladder.sh [graft|derive|consensus|derive_large|consensus_large]
#         with no argument, all five run in order.
#
# The stages are NOT independent: `derive` reads this model's consensus columns
# out of the cooled grafting run, so `consensus` cannot be sampled until `graft`
# and `derive` have both completed.
set -euo pipefail

source "$(dirname "$0")/_paths.sh"
OUT=${OUT:-$REPO/results/ladder}

CRITIC=201
GRAFT=165-243
# The consensus set is DERIVED FROM THIS MODEL, not inherited. Nothing here may
# depend on the earlier model: its consensus columns and its coupling ranking
# were properties of a model we no longer trust. stage_consensus reads the set
# that stage_derive writes after the cooled grafting runs exist.
CONSENSUS_FILE=$OUT/consensus_columns.txt
# A second, wider set. The cold run concentrates on 8 columns and nothing
# between 13 and ~20 is defensible from it -- below 44% its frequencies fall
# into the 14%-and-under noise. The wider set is therefore read from the most
# strongly biased WARM run (beta = 1, lambda = 1000, 22 mutations per chain),
# where there is a clean 12-point gap at 50% (51.3% -> 39.1%). The 8 columns are
# fully contained in the 22, so the two are a nested budget comparison.
CONSENSUS_LARGE_FILE=$OUT/consensus_large_columns.txt

NCHAINS=${NCHAINS:-1000}
# 60,000 steps. The equilibration tests on this model put the slowest case --
# the cold beta = 5 chains, acceptance 0.0025 -- at ~11,400 steps, and beta = 1
# (lambda = 0, the slowest-mixing warm case) at ~8,500. 60k is a >5x margin, and
# every run is traced so SI S6 re-checks it rather than taking it on trust.
NSTEPS=${NSTEPS:-60000}
DEVICE=${DEVICE:-mps}

# Every run is traced. The earlier pipeline traced only three lambdas of one
# condition, which is why SI S6 could not show equilibration for consensus-19.
TRACE=(--trace-every 1000 --trace-schedule log)

run() { echo; echo "### $*"; "$@"; }

# $1 = output subdir, $2 = allowed cols, $3 = beta, rest = lambdas
sample() {
    local name=$1 cols=$2 beta=$3; shift 3
    run $PY "$REPO/scripts/run_ensemble.py" --wt "$WT" --model "$PARAMS" \
        --output "$OUT/$name/ensembles" --critic-col $CRITIC --allowed-cols "$cols" \
        --allow-gap-moves --lambdas "$@" --beta "$beta" \
        --n-chains $NCHAINS --n-steps $NSTEPS "${TRACE[@]}" \
        --seed 0 --device "$DEVICE" --cache-model

    run $PY "$REPO/scripts/substitute_critic.py" \
        --ensemble-dir "$OUT/$name/ensembles" --output "$OUT/$name/ensembles_serine" \
        --wt "$WT" --model "$PARAMS" --critic-col $CRITIC --beta 1.0 \
        --device "$DEVICE" --cache-model

    run $PY "$REPO/scripts/add_natural_neighbour.py" \
        --ensemble-dir "$OUT/$name/ensembles_serine" --natural "$ALN" \
        --critic-col $CRITIC --allowed-cols "$cols" --wt "$WT" \
        --as-percentage --device "$DEVICE"
}

# A + B: everything on the grafting segment. Must run before stage_derive.
stage_graft() {
    sample grafting "$GRAFT" 1 $GRAFT_LAMBDAS
    for spec in "${COOL_SPECS[@]}"; do
        local beta=${spec%%:*} lams=${spec#*:}
        sample "cooling/b$beta" "$GRAFT" "$beta" $lams
    done
}

# Derive this model's own consensus set from its cold run, the same way the set
# was originally defined: the columns rewritten in >= 95 % of the chains. The
# SIZE is whatever the threshold gives -- it is not forced to 19.
stage_derive() {
    run $PY "$REPO/analysis/derive_consensus.py" --results "$OUT" --wt "$WT" \
        --threshold 0.95 --output "$CONSENSUS_FILE"
}

# C + D: the same two protocols on this model's own consensus columns.
# C: the tight consensus set, beta = 1 only (no cooling -- an 8-site library is
# already nearly deterministic; cooling it has nothing left to remove).
stage_consensus() {
    [[ -f $CONSENSUS_FILE ]] || { echo "run stage_derive first" >&2; exit 1; }
    local cols; cols=$(cat "$CONSENSUS_FILE")
    echo "tight consensus columns: $cols"
    sample consensus "$cols" 1 $CONS_LAMBDAS
}

# Derive the wider set from the warm lambda = 1000 run at a 50% threshold.
stage_derive_large() {
    run $PY "$REPO/analysis/derive_consensus.py" --results "$OUT" --wt "$WT" \
        --cold-run grafting/ensembles/lam1000 --threshold 0.50 \
        --output "$CONSENSUS_LARGE_FILE"
}

# D + E: the wider consensus set, with and without cooling.
stage_consensus_large() {
    [[ -f $CONSENSUS_LARGE_FILE ]] || { echo "run stage_derive_large first" >&2; exit 1; }
    local cols; cols=$(cat "$CONSENSUS_LARGE_FILE")
    echo "wide consensus columns: $cols"
    sample consensus_large "$cols" 1 $CONS_LAMBDAS
    for spec in "${COOL_SPECS[@]}"; do
        local beta=${spec%%:*} lams=${spec#*:}
        sample "consensus_large_cooling/b$beta" "$cols" "$beta" $lams
    done
}

# --- the ladders ----------------------------------------------------------------
# Chosen from pilot scans on THIS model; see analysis/figures_ladder/README.md.
#
# CAREFUL: run_ensemble's printed "<P(S)>" and summary.json's mean_P_S are the
# fraction of chains carrying S at the critic, which saturates at 1.0 for any
# cooled run. The quantity the figures use is the beta = 1 conditional
# P(S | sigma_\i*) recomputed from the sequences, which at beta = 5, lam_tilde = 10
# is 0.756, not 1.0. The ladders below are set from the recomputed value.
GRAFT_LAMBDAS="${GRAFT_LAMBDAS:-0 10 30 100 300 1000}"
CONS_LAMBDAS="${CONS_LAMBDAS:-0 10 30 100 300 1000}"
# lam_tilde = lambda / beta held at 10, 30, 100 down each beta rung.
# Override for a smoke test with e.g. COOL_SPECS_STR="5:500".
if [[ -n "${COOL_SPECS_STR:-}" ]]; then
    IFS=';' read -ra COOL_SPECS <<< "$COOL_SPECS_STR"
else
    COOL_SPECS=("1.5:15 45 150" "2:20 60 200" "3:30 90 300" "5:50 150 500" "10:100 300 1000")
fi

STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(graft derive consensus derive_large consensus_large)
for stage in "${STAGES[@]}"; do
    echo; echo "================ stage: $stage ================"
    "stage_$stage"
done
echo; echo "Done. Results under $OUT"
