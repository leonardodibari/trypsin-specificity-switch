#!/usr/bin/env bash
# Add the natural-neighbour column to the selected blocks, over the ONE common
# column window that makes it comparable between mutation sets.
#
# WHY A COMMON WINDOW. add_natural_neighbour.py scores the distance over the
# columns the sampler was allowed to move, and --as-percentage normalises by
# that count. Per set that is the right choice and it is what the ladder does;
# ACROSS sets it is not, because "% of 14 columns" and "% of 79 columns" are not
# the same quantity -- which is exactly why figures_ladder/fig06 drops the axis
# when it compares site sets (CLAMP_CONTROL caveat 3).
#
# That caveat is about the scoring, not the observable. All three sets here are
# subsets of the grafting window, so scoring every selected run over that one
# 79-column window makes the axis mean the same thing for all of them. The
# window is therefore hard-coded here and must NOT be changed to each set's own
# columns -- that is the thing this script exists to avoid.
#
# Usage:  bash t_steering/s03_prepare.sh
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../analysis/_paths.sh"

CRITIC=201
COMMON_WINDOW=165-243          # the grafting window; see above. Do not localise.
DEVICE=${DEVICE:-mps}

[[ -f $ALN ]] || { echo "MISSING alignment: $ALN (see docs/DATA.md)" >&2; exit 1; }

# The blocks holding a selected run. Printed by the selection itself, so this
# script and the figure can never disagree about which runs were chosen.
blocks=$(cd "$HERE" && $PY - <<'PYEOF'
import sys
sys.path.insert(0, "analysis")   # cwd is $HERE, set by the cd above
import config, data, selection
table = data.summarise(data.load_all(serine=True, only_saturated=True))
chosen = selection.select(table, selection.wild_type_Q())
seen = []
for row in chosen.itertuples():
    block = str(config.block_dir(row.set, row.beta, serine=True))
    if block not in seen:
        seen.append(block)
print("\n".join(seen))
PYEOF
)

[[ -n $blocks ]] || { echo "no selected runs -- nothing to prepare" >&2; exit 1; }

for block in $blocks; do
    echo; echo "### $block  (window $COMMON_WINDOW)"
    $PY "$REPO/scripts/add_natural_neighbour.py" \
        --ensemble-dir "$block" --natural "$ALN" \
        --critic-col $CRITIC --allowed-cols "$COMMON_WINDOW" --wt "$WT" \
        --as-percentage --device "$DEVICE"
done
echo; echo "Done. s03_corner.py will now draw the 5x5."
