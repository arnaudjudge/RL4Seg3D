#!/usr/bin/env bash
# Submit a prediction sweep. Everything except which case list to work on lives in
# rl4seg3d/config/predict3d.yaml (paths, tta, tto, limit, ordering) and
# config/launcher/predict_nibi.yaml (walltime, gres, cpus) -- set those per cluster once.
#
#   ./run_predict.sh 3                              csv_split/maple_part3of6.csv, 100 tasks
#   ./run_predict.sh 3 200                          ... 200 tasks
#   ./run_predict.sh maple.csv                      an explicit list instead of a part number
#   ./run_predict.sh 3 100 limit=5 run_time_min=60  anything further is passed to hydra
#
# Re-run the same line freely: cases are claimed atomically, so tasks never collide and each
# takes the next ones still available.
#
# Between runs, with nothing queued or running:
#   find <output>/csv -name '*.csv' ! -empty | wc -l   # done
#   find <output>/csv -type f -empty -delete           # release claims from killed tasks
set -euo pipefail

# ---- set once per cluster ----
DATA=$HOME/scratch/data/maple_norm     # where maple.csv and csv_split/ live
LAUNCHER=predict_nibi                  # config/launcher/<name>.yaml
# ------------------------------

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }
[ $# -ge 1 ] || { usage; exit 1; }

CSV=$1; shift
[[ "$CSV" =~ ^[0-9]+$ ]] && CSV="$DATA/csv_split/maple_part${CSV}of6.csv"
[[ "$CSV" == /* ]]      || CSV="$DATA/$CSV"
[ -f "$CSV" ] || { echo "no such case list: $CSV" >&2; exit 1; }

N=100
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then N=$1; shift; fi

# array, so overrides needing nested quotes survive intact
CMD=(python rl4seg3d/predict_3d.py --multirun "+launcher=$LAUNCHER"
     "shard_id=range(0,$N)" "case_list=$CSV" "view_mapping=$CSV" "$@")
printf '%q ' "${CMD[@]}"; echo
exec "${CMD[@]}"
