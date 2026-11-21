#!/usr/bin/env bash
# STAGE 1b for the serine-protease model: the all-sites-free run and the sector
# annotation that figure F8 needs.
#
# F8 asks whether the design concentrates its mutations on the red sector, and
# whether freeing every column dilutes that. It needs sector_enrichment.csv for
# two runs: the grafting segment (produced by run_ladder.sh) and an
# unrestricted run over every column, produced here.
#
# Kept separate from run_ladder.sh so it can be run after it without editing a
# script that is already executing.
#
# Usage:  bash analysis/run_ladder_sectors.sh [free|annotate]
set -euo pipefail

source "$(dirname "$0")/_paths.sh"
OUT=${OUT:-$REPO/results/ladder}
SECTORS=$REPO/data/sectors_iter_aln_v2.csv

CRITIC=201
ALL=0-264
NCHAINS=${NCHAINS:-1000}
NSTEPS=${NSTEPS:-60000}
DEVICE=${DEVICE:-mps}
FREE_LAMBDAS=${FREE_LAMBDAS:-"0 10 30 100 300 1000"}

run() { echo; echo "### $*"; "$@"; }

stage_free() {
    run $PY "$REPO/scripts/run_ensemble.py" --wt "$WT" --model "$PARAMS" \
        --output "$OUT/free/ensembles" --critic-col $CRITIC --allowed-cols "$ALL" \
        --allow-gap-moves --lambdas $FREE_LAMBDAS --beta 1 \
        --n-chains $NCHAINS --n-steps $NSTEPS \
        --trace-every 1000 --trace-schedule log \
        --seed 0 --device "$DEVICE" --cache-model

    run $PY "$REPO/scripts/substitute_critic.py" \
        --ensemble-dir "$OUT/free/ensembles" --output "$OUT/free/ensembles_serine" \
        --wt "$WT" --model "$PARAMS" --critic-col $CRITIC --beta 1.0 \
        --device "$DEVICE" --cache-model

    # NO add_natural_neighbour here. It compares every design against all 89,439
    # naturals to find the nearest S-carrier, which costs ~7 min per lambda and
    # dominates the whole run. F8 -- the only figure this run feeds -- reads
    # sector_enrichment.csv and never touches that column.
}

# Sector annotation for both runs. This writes sector_enrichment.csv, which is
# the only thing F8 reads, and also adds n_mut_red / n_mut_grafting / n_mut_other
# into each lambda's observables.csv.
# Sector annotation. Runs for whichever of the two exist: F8 draws the
# all-sites-free curve only if that run is present.
stage_annotate() {
    for name in ${ANNOTATE:-grafting free}; do
        [[ -d $OUT/$name/ensembles ]] || { echo "skipping $name: not sampled"; continue; }
        run $PY "$REPO/scripts/annotate_sectors.py" --sectors "$SECTORS" --wt "$WT" \
            --ensemble-dir "$OUT/$name/ensembles" --output "$OUT/$name/sectors"
    done
}

STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(free annotate)
for stage in "${STAGES[@]}"; do
    echo; echo "================ stage: $stage ================"
    "stage_$stage"
done
echo; echo "Done. Results under $OUT"
