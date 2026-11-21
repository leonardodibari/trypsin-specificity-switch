# Shared path resolution for every driver in analysis/.  Sourced, never run.
#
# Defines: REPO  DATA_ROOT  PARAMS  ALN  WT  PY
#
# Nothing here is machine-specific.  The repository locates itself from this
# file, and the heavy data -- the 24 MB alignment and the 137 MB parameter file,
# both far too large for git -- is looked up under $TRYP_DATA_ROOT, which
# defaults to the repository's parent directory.  See docs/DATA.md.
#
# Override any of these from the environment:
#   TRYP_DATA_ROOT=/mnt/scratch/tryp  bash analysis/run_clamp.sh
#   PARAMS=/path/to/other_params.dat.gz bash analysis/run_ladder.sh

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${TRYP_DATA_ROOT:-$(dirname "$REPO")}

PARAMS=${PARAMS:-$DATA_ROOT/DCAmodel-serine_protease/params.dat.gz}
ALN=${ALN:-$DATA_ROOT/iter_aln_v2.faa}
WT=$REPO/data/wt_trypsin.fasta

# Prefer the repository's own virtualenv; fall back to whatever python3 is on
# PATH so the scripts still work under conda, uv, or a system interpreter.
if [[ -z ${PY:-} ]]; then
    if [[ -x $REPO/.venv/bin/python ]]; then PY=$REPO/.venv/bin/python
    else PY=$(command -v python3); fi
fi

# --- preflight ---------------------------------------------------------------
# Fail here, with the reason, rather than 40 minutes into a run or -- worse --
# with a silently empty ensemble.
_missing=0
for _f in "$PARAMS" "$ALN" "$WT"; do
    if [[ ! -f $_f ]]; then echo "MISSING: $_f" >&2; _missing=1; fi
done
if (( _missing )); then
    cat >&2 <<EOF

The heavy data files are not where this script expects them.

  repository : $REPO
  data root  : $DATA_ROOT   (override with TRYP_DATA_ROOT)

The data root must contain:
  iter_aln_v2.faa                            the 89,439-sequence alignment
  DCAmodel-serine_protease/params.dat.gz     the fitted Potts model

Neither is in git -- they are 24 MB and 137 MB.  See docs/DATA.md for what they
are and where to get them.
EOF
    exit 1
fi
if [[ ! -x $PY ]]; then
    echo "No usable python: set PY=/path/to/python, or create the venv (see README)." >&2
    exit 1
fi
unset _missing _f
