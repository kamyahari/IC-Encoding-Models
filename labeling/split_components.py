#!/usr/bin/env python3
"""
Split a 4D ICA spatial maps file into individual 3D NIfTI volumes.

parcellate saves all IC spatial maps as a single 4D file:
  spatial_maps_4d_resampled.nii.gz  — shape (x, y, z, n_components)

This script splits it into one file per component, needed for:
  - ICA-AROMA (melodic_IC.nii.gz, handled separately in prepare_aroma.py)
  - Atlas matching / ranking (rank_ics.py)

Output: ICAComponents/ICA_component_01.nii.gz ... ICA_component_N.nii.gz

Usage:
    python split_components.py <subject_id>
    e.g.: python split_components.py sub-UTS01
"""

import sys
import os
import nibabel as nib
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION — set your path here
# ============================================================

# TODO: directory containing parcellate output per subject
PARCELLATE_OUT = Path("/path/to/parcellate/output/")

# ============================================================


def split_components(subject_id: str):
    sample_dir = PARCELLATE_OUT / subject_id / "sample" / "main"
    input_path  = sample_dir / "spatial_maps_4d_resampled.nii.gz"
    output_dir  = sample_dir / "ICAComponents"

    if not input_path.exists():
        raise FileNotFoundError(f"Spatial maps not found: {input_path}")

    output_dir.mkdir(exist_ok=True)

    img_4d  = nib.load(input_path)
    data_4d = img_4d.get_fdata()          # (x, y, z, n_components)
    affine  = img_4d.affine
    header  = img_4d.header
    n_components = data_4d.shape[3]

    print(f"  {subject_id}: splitting {n_components} components -> {output_dir}")
    for i in range(n_components):
        comp_img  = nib.Nifti1Image(data_4d[..., i], affine, header)
        out_path  = output_dir / f"ICA_component_{i+1:02d}.nii.gz"
        nib.save(comp_img, out_path)

    print(f"  Saved {n_components} components.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python split_components.py <subject_id>")
        sys.exit(1)

    split_components(sys.argv[1])


if __name__ == "__main__":
    main()
