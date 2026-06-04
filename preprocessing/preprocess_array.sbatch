#!/bin/bash
#SBATCH --job-name=fmri_preproc
#SBATCH --array=0-38
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/preproc_%A_%a.out
#SBATCH --error=logs/preproc_%A_%a.err
#SBATCH --mail-type=FAIL
# TODO: set your email
#SBATCH --mail-user=your_email@institution.edu

# ============================================================
# CONFIGURATION — set these paths before running
# ============================================================
SUBJECT=${1}   # e.g. sub-UTS01 (passed from submit_preprocessing.sh)
MODE=${2:-ica} # "ica" or "encoding"

# TODO: set your paths
FMRIPREP_DIR="/path/to/fmriprep/derivatives/"
OUTPUT_BASE="/path/to/output/"
REPO_DIR="/path/to/ic-ems/"

# ============================================================
if [ "$MODE" == "ica" ]; then
    PYTHON_SCRIPT="${REPO_DIR}/preprocessing/preprocess_ica.py"
    OUTPUT_DIR="${OUTPUT_BASE}/PreprocessedMNI/${SUBJECT}/"
else
    PYTHON_SCRIPT="${REPO_DIR}/preprocessing/preprocess_encoding.py"
    OUTPUT_DIR="${OUTPUT_BASE}/PreprocessedMNIEncoding/${SUBJECT}/"
fi

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

# Find all MNI152 BOLD files for this subject
FUNC_FILES=($(find "${FMRIPREP_DIR}/${SUBJECT}/ses-*/func/" \
    -name "*space-MNI152*_desc-preproc_bold.nii.gz" 2>/dev/null | sort))

if [ ${#FUNC_FILES[@]} -eq 0 ]; then
    echo "ERROR: No MNI152 BOLD files found for ${SUBJECT}"
    exit 1
fi

FUNC_FILE="${FUNC_FILES[$SLURM_ARRAY_TASK_ID]}"
if [ ! -f "$FUNC_FILE" ]; then
    echo "ERROR: File not found at index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# TODO: set path to GM probability segmentation mask from fMRIPrep
MASK_FILE="${FMRIPREP_DIR}/${SUBJECT}/ses-1/anat/${SUBJECT}_ses-1_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz"

if [ ! -f "$MASK_FILE" ]; then
    echo "ERROR: Mask not found: $MASK_FILE"
    exit 1
fi

echo "Subject: ${SUBJECT} | Mode: ${MODE} | Task: ${SLURM_ARRAY_TASK_ID}"
echo "File: $(basename $FUNC_FILE)"

module load anaconda3
conda activate parcellate

python "${PYTHON_SCRIPT}" "${FUNC_FILE}" "${MASK_FILE}" "${OUTPUT_DIR}"
