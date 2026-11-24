#!/usr/bin/env bash
# Rescore every sampled block the way the ladder's figures read it, so the
# restraint runs and the ladder runs are comparable number for number.
#
# WHY THIS EXISTS. run_ensemble.py scores each ensemble at the run's own beta,
# and a cooled run's P(S) saturates at 1.0 there (RESULTS.md section 5 flags
# exactly this). Every ladder figure therefore reads ensembles_serine/, which
# substitute_critic.py writes: the critic column stamped with S and everything
# rescored at beta = 1. Comparing our beta = 1 runs against the ladder's cooled
# runs means reading both through that same door.
#
# Stamping S at the critic changes nothing in the conditional -- P(S | sigma_\i*)
# does not depend on the residue at i* -- but it does fix the critic column so
# the Hamming distances of the two sets are counted the same way.
#
# Usage:  bash hamming-restraint/postprocess.sh [serine|neighbour|check]
#         with no argument: serine check   (neighbour is slow; ask for it)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../analysis/_paths.sh"

OUT=${OUT:-$HERE/results}
CRITIC=201
DEVICE=${DEVICE:-mps}

run() { echo; echo "### $*"; "$@"; }

blocks() { find "$OUT" -mindepth 1 -maxdepth 1 -type d -not -name '*_serine' | sort; }

# Rescore at beta = 1 with S stamped at the critic column.
stage_serine() {
    for block in $(blocks); do
        [[ -n $(find "$block" -mindepth 1 -maxdepth 1 -name 'lam*' -print -quit) ]] || continue
        run $PY "$REPO/scripts/substitute_critic.py" \
            --ensemble-dir "$block" --output "${block}_serine" \
            --wt "$WT" --model "$PARAMS" --critic-col $CRITIC --beta 1.0 \
            --device "$DEVICE" --cache-model
    done
}

# Distance to the nearest natural sequence carrying S at the critic: the novelty
# guard. A design 100 substitutions from the wild type that is a copy of some
# natural chymotrypsin is retrieval, not design, and this is what says so.
# CAREFUL: it is scored over each block's own --allowed-cols, so it is NOT
# comparable between a 79-column block and an all-sites-free one (CLAMP_CONTROL
# caveat 3). Compare it within a block only.
stage_neighbour() {
    for block in $(blocks); do
        local cols
        case "$(basename "$block")" in
            a1_graft|*_graft*) cols=165-243 ;;
            *free*) cols=0-264 ;;
            *) echo "no column spec for $block; skipping" >&2; continue ;;
        esac
        [[ -d ${block}_serine ]] || continue
        run $PY "$REPO/scripts/add_natural_neighbour.py" \
            --ensemble-dir "${block}_serine" --natural "$ALN" \
            --critic-col $CRITIC --allowed-cols "$cols" --wt "$WT" \
            --as-percentage --device "$DEVICE"
    done
}

stage_check() {
    run $PY "$HERE/analysis/saturation.py" "$OUT" --csv "$OUT/saturation.csv"
}

STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(serine check)
for stage in "${STAGES[@]}"; do
    echo; echo "================ postprocess: $stage ================"
    "stage_$stage"
done
