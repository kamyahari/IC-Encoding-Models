#!/bin/bash
# Submit preprocessing jobs for all subjects.
# Usage:
#   bash submit_preprocessing.sh ica       # ICA preprocessing (smoothing + bandpass)
#   bash submit_preprocessing.sh encoding  # Encoding preprocessing (no smoothing/bandpass)

MODE=${1:-ica}
if [[ "$MODE" != "ica" && "$MODE" != "encoding" ]]; then
    echo "Usage: bash submit_preprocessing.sh [ica|encoding]"
    exit 1
fi

SUBJECTS=("sub-UTS01" "sub-UTS02" "sub-UTS03" "sub-UTS04"
          "sub-UTS05" "sub-UTS06" "sub-UTS07" "sub-UTS08")

for SUB in "${SUBJECTS[@]}"; do
    sbatch preprocess_array.sbatch "$SUB" "$MODE"
done
