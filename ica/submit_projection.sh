#!/bin/bash
# Submit IC projection jobs for all subjects as a SLURM array.
# Each array index maps to one subject in project_components.py's SUBJECTS list.
#
# Usage: bash submit_projection.sh

#SBATCH --job-name=ic_projection
#SBATCH --array=0-7
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/projection_%A_%a.out
#SBATCH --error=logs/projection_%A_%a.err
#SBATCH --mail-type=FAIL
# TODO: set your email
#SBATCH --mail-user=your_email@institution.edu

mkdir -p logs

module load anaconda3
conda activate parcellate

# TODO: set path to repo
REPO_DIR="/path/to/ic-ems/"

python "${REPO_DIR}/ica/project_components.py" "${SLURM_ARRAY_TASK_ID}"
