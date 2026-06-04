#!/usr/bin/env python3
"""
Prepare inputs for ICA-AROMA from parcellate ICA output.

ICA-AROMA expects a specific directory structure with melodic-format files.
This script:
  1. Concatenates motion confounds across runs and saves a .par file.
  2. Extracts the mixing matrix and FT mixing matrix from the saved ICA model.
  3. Copies spatial maps to the expected melodic.ica directory.

Run once per subject before calling ICA_AROMA.py.

Usage:
    python prepare_aroma.py <subject_id>
    e.g.: python prepare_aroma.py sub-UTS01
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import joblib
from scipy.fft import fft
from pathlib import Path

# ============================================================
# CONFIGURATION — set your paths here
# ============================================================

# TODO: path to fMRIPrep derivatives (for confound .tsv files)
FMRIPREP_DIR = Path("/path/to/fmriprep/derivatives/")

# TODO: path to parcellate output (contains ica_model.joblib, spatial_maps_4d_resampled.nii.gz)
PARCELLATE_OUT = Path("/path/to/parcellate/output/")

# TODO: output directory for AROMA inputs
AROMA_BASE = Path("/path/to/aroma/inputs/")

# Stories used for ICA estimation (sessions and task names)
# These are the 3 sessions used as IC Estimation Set
ICA_SESSIONS = [
    ("ses-2", "alternateithicatom"),
    ("ses-3", "howtodraw"),
    ("ses-4", "exorcism"),
]

# ============================================================


def make_confounds_par(subject_id: str):
    """
    Concatenate motion confounds across ICA estimation sessions and save as .par file.
    fMRIPrep confound TSVs -> concatenated -> 6-column .par (rot_x/y/z, trans_x/y/z).
    """
    out_dir = AROMA_BASE / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_list = []
    for ses, task in ICA_SESSIONS:
        # TODO: adjust confound filename pattern if your fMRIPrep output differs
        conf_path = (FMRIPREP_DIR / subject_id / ses / "func" /
                     f"{subject_id}_{ses}_task-{task}_desc-confounds_timeseries.tsv")
        if not conf_path.exists():
            raise FileNotFoundError(f"Confounds not found: {conf_path}")
        conf_list.append(pd.read_csv(conf_path, sep='\t'))

    concat = pd.concat(conf_list, axis=0).reset_index(drop=True)
    tsv_out = out_dir / f"{subject_id}_concat_confounds.tsv"
    concat.to_csv(tsv_out, sep='\t', index=False)
    print(f"  Saved concatenated confounds: {tsv_out}")

    # Save as .par (space-separated, 6 motion params, no header)
    par_cols = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    par_out = out_dir / f"{subject_id}_motion.par"
    concat[par_cols].to_csv(par_out, sep=' ', header=False, index=False, float_format='%.6f')
    print(f"  Saved motion .par: {par_out}")
    return par_out


def prepare_melodic_dir(subject_id: str):
    """
    Build the melodic.ica directory expected by ICA-AROMA:
      melodic.ica/
        melodic_mix       — mixing matrix (time x components)
        melodic_FTmix     — FFT of mixing matrix (positive freqs)
        melodic_IC.nii.gz — 4D spatial maps
    """
    sample_dir = PARCELLATE_OUT / subject_id / "sample" / "main"
    melodic_dir = AROMA_BASE / subject_id / "ICA_AROMA_out" / "melodic.ica"
    melodic_dir.mkdir(parents=True, exist_ok=True)

    # Load ICA model saved by parcellate (joblib) and extract mixing matrix
    ica_path = sample_dir / "ica_model.joblib"
    if not ica_path.exists():
        raise FileNotFoundError(f"ICA model not found: {ica_path}")
    print(f"  Loading ICA model: {ica_path}")
    m = joblib.load(ica_path)
    time_courses = m.mixing_   # shape: (time, n_components)

    np.savetxt(melodic_dir / "melodic_mix", time_courses,
               fmt='%.6f', delimiter='  ')
    print(f"  Saved melodic_mix: shape {time_courses.shape}")

    # Compute FFT of mixing matrix (positive frequencies only)
    ft = np.abs(fft(time_courses, axis=0))
    n_freqs = ft.shape[0] // 2 + 1
    np.savetxt(melodic_dir / "melodic_FTmix", ft[:n_freqs, :],
               fmt='%.6f', delimiter='  ')
    print(f"  Saved melodic_FTmix: shape {ft[:n_freqs].shape}")

    # Copy 4D spatial maps
    spatial_src = sample_dir / "spatial_maps_4d_resampled.nii.gz"
    if not spatial_src.exists():
        raise FileNotFoundError(f"Spatial maps not found: {spatial_src}")
    shutil.copy(spatial_src, melodic_dir / "melodic_IC.nii.gz")
    print(f"  Copied melodic_IC.nii.gz")


def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_aroma.py <subject_id>")
        sys.exit(1)

    subject_id = sys.argv[1]
    print(f"=== Preparing AROMA inputs for {subject_id} ===")

    print("\n[1/2] Building motion .par file...")
    make_confounds_par(subject_id)

    print("\n[2/2] Building melodic.ica directory...")
    prepare_melodic_dir(subject_id)

    # Print the ICA-AROMA run command for convenience
    mask = PARCELLATE_OUT / subject_id / "mask.nii.gz"
    par  = AROMA_BASE / subject_id / f"{subject_id}_motion.par"
    mel  = AROMA_BASE / subject_id / "ICA_AROMA_out" / "melodic.ica"
    out  = AROMA_BASE / subject_id / "ICA_AROMA_output"
    # TODO: set path to MNI warp file
    warp = "/path/to/mni2009_to_fslmni152_warp.nii.gz"

    print(f"\n=== Run ICA-AROMA with ===")
    print(f"python ICA-AROMA/ICA_AROMA.py \\")
    print(f"  -o {out} \\")
    print(f"  -m {mask} \\")
    print(f"  -mc {par} \\")
    print(f"  -md {mel} \\")
    print(f"  -w {warp} \\")
    print(f"  -tr 2.0 \\")
    print(f"  -den no")
    print("\nAdd --overwrite if rerunning into an existing output directory.")


if __name__ == "__main__":
    main()
