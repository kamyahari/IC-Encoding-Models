#!/usr/bin/env python3
"""
Project fMRI data into IC space using the ICA mixing matrix from parcellate.

For each preprocessed BOLD file, applies the pseudoinverse of the ICA spatial
source matrix (S†) to produce component timecourses: A_new = X_new @ S†

Output: one .npy file per story, shape (n_components, n_timepoints).

Usage:
    python project_components.py <subject_index>
    e.g.: python project_components.py 0   # processes sub-UTS01

Submit all subjects:
    bash submit_projection.sh
"""

import sys
import re
import numpy as np
import nibabel as nib
from pathlib import Path

# ============================================================
# CONFIGURATION — set your paths here
# ============================================================

# TODO: directory containing preprocessed encoding BOLD files (one subdir per subject)
PREPROCESSED_BASE = Path("/path/to/PreprocessedMNIEncoding/")

# TODO: directory containing parcellate output (ica_mixing_main.npy, mask.nii.gz)
PARCELLATE_OUT = Path("/path/to/parcellate/output/")

# TODO: output directory for projected component timecourses
OUTPUT_BASE = Path("/path/to/ProjectedComponents/")

# ============================================================

SUBJECTS = [
    "sub-UTS01", "sub-UTS02", "sub-UTS03", "sub-UTS04",
    "sub-UTS05", "sub-UTS06", "sub-UTS07", "sub-UTS08",
]

# Stories used for encoding model training + test story
STORIES_TO_KEEP = [
    'adollshouse', 'adventuresinsayingyes', 'avatar', 'buck',
    'eyespy', 'fromboyhoodtofatherhood', 'hangtime', 'haveyoumethimyet',
    'inamoment', 'itsabox', 'legacy', 'naked', 'odetostepfather',
    'sloth', 'souls', 'stagefright', 'swimmingwithastronauts',
    'thatthingonmyarm', 'theclosetthatateeverything', 'tildeath',
    'undertheinfluence', 'wheretheressmoke',
]


def extract_task_name(filename: str):
    m = re.search(r'task-([A-Za-z0-9]+)', filename)
    return m.group(1) if m else None


def is_averaged_wheretheressmoke(filename: str) -> bool:
    return 'wheretheressmoke' in filename and 'averaged' in filename


def project_to_component_space(data_file, weight_file, mask_file):
    """
    Project preprocessed BOLD data into component space.

    Uses the pseudoinverse of the ICA source matrix S (shape: n_voxels x n_components)
    to compute component timecourses: A_new = X_new @ S†

    Returns z-score normalised timecourses, shape (n_components, n_timepoints).
    """
    W_2d = np.load(str(weight_file))
    if W_2d.ndim > 2:
        W_2d = np.squeeze(W_2d)
    if W_2d.ndim != 2:
        raise ValueError(f"Weight matrix must be 2D, got shape {W_2d.shape}")
    n_voxels_w, n_components = W_2d.shape
    print(f"    Weight matrix: {W_2d.shape}  (voxels x components)")

    img = nib.load(str(data_file))
    data = img.get_fdata()
    mask_data = nib.load(str(mask_file)).get_fdata().astype(bool)

    data_2d = data[mask_data]           # (n_voxels, n_timepoints)
    n_voxels_d, n_volumes = data_2d.shape
    print(f"    Data shape: {data_2d.shape}")

    if n_voxels_w != n_voxels_d:
        raise ValueError(
            f"Voxel mismatch: weights={n_voxels_w}, data={n_voxels_d}. "
            "Ensure the same mask was used during ICA and preprocessing.")

    W_pinv = np.linalg.pinv(W_2d)
    components = W_pinv @ data_2d      # (n_components, n_timepoints)

    # Z-score normalise per component
    components = ((components - components.mean(axis=1, keepdims=True))
                  / components.std(axis=1, keepdims=True))
    print(f"    Output shape: {components.shape}")
    return components


def collect_files(subject_id: str):
    preproc_dir = PREPROCESSED_BASE / subject_id
    if not preproc_dir.is_dir():
        print(f"  WARNING: directory not found: {preproc_dir}")
        return []

    selected = []
    for f in sorted(preproc_dir.glob("*.nii.gz")):
        task = extract_task_name(f.name)
        if task is None:
            if is_averaged_wheretheressmoke(f.name):
                selected.append(f)
        elif task == 'wheretheressmoke':
            if is_averaged_wheretheressmoke(f.name):
                selected.append(f)
        elif task in STORIES_TO_KEEP and 'bold' in f.name:
            selected.append(f)
    return selected


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python project_components.py <subject_index>  (0–{len(SUBJECTS)-1})")
        sys.exit(1)

    idx = int(sys.argv[1])
    if not 0 <= idx < len(SUBJECTS):
        print(f"ERROR: index {idx} out of range")
        sys.exit(1)

    subject_id = SUBJECTS[idx]
    print("=" * 60)
    print(f"  Subject: {subject_id}  (index {idx})")
    print("=" * 60)

    weight_file = PARCELLATE_OUT / subject_id / "sample" / "main" / "ica_mixing_main.npy"
    mask_file   = PARCELLATE_OUT / subject_id / "mask.nii.gz"
    out_dir     = OUTPUT_BASE / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for f, label in [(weight_file, "Weight"), (mask_file, "Mask")]:
        if not f.is_file():
            print(f"ERROR: {label} file not found: {f}")
            sys.exit(1)

    files = collect_files(subject_id)
    print(f"  Found {len(files)} file(s) to process")

    n_ok = 0
    for data_file in files:
        print(f"\n  --- {data_file.name} ---")
        try:
            components = project_to_component_space(data_file, weight_file, mask_file)
            task = extract_task_name(data_file.name) or "unknown"
            out_npy = out_dir / f"task-{task}_components.npy"
            np.save(str(out_npy), components)
            print(f"  Saved -> {out_npy}")
            n_ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n  Done: {n_ok}/{len(files)} stories projected for {subject_id}")


if __name__ == "__main__":
    main()
