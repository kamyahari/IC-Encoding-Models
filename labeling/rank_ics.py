#!/usr/bin/env python3
"""
Rank ICA components by spatial correlation with a reference atlas.

For each split IC component (from split_components.py), computes the Pearson
correlation with a reference atlas and ranks them. Useful for identifying
which components correspond to auditory, language, visual, etc. networks.

Usage:
    python rank_ics.py <reference_atlas> <ic_directory> [--mask <mask_file>] [--out <output.csv>]

    e.g.:
    python rank_ics.py \\
        ../parcellate/resources/DU15_LANG.nii.gz \\
        /path/to/sub-UTS01/sample/main/ICAComponents/ \\
        --mask /path/to/sub-UTS01/mask.nii.gz \\
        --out sub-UTS01_LANG_rankings.csv
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to


def load_nifti(filepath):
    img = nib.load(filepath)
    return img.get_fdata(), img


def resample_to_reference(ic_img, ref_img, order=3):
    return resample_from_to(ic_img, ref_img, order=order).get_fdata()


def compute_correlation(reference, ic, mask=None):
    ref_flat = reference.flatten()
    ic_flat  = ic.flatten()

    if mask is not None:
        m = mask.flatten().astype(bool)
        ref_flat = ref_flat[m]
        ic_flat  = ic_flat[m]

    valid = np.isfinite(ref_flat) & np.isfinite(ic_flat)
    ref_flat, ic_flat = ref_flat[valid], ic_flat[valid]

    if len(ref_flat) < 2:
        return np.nan
    if np.std(ref_flat) == 0 or np.std(ic_flat) == 0:
        return np.nan

    return np.corrcoef(ic_flat, ref_flat)[0, 1]


def rank_ics_by_correlation(reference_path, ic_paths, mask_path=None,
                            use_absolute=True, output_csv=None):
    if isinstance(ic_paths, str) and os.path.isdir(ic_paths):
        ic_paths = sorted(glob.glob(os.path.join(ic_paths, '*.nii.gz')))
    if len(ic_paths) == 0:
        raise ValueError("No IC files found.")

    reference_data, reference_img = load_nifti(reference_path)

    mask_data = None
    if mask_path:
        mask_data, mask_img = load_nifti(mask_path)
        if mask_data.shape != reference_data.shape:
            mask_data = resample_to_reference(mask_img, reference_img, order=0)
        mask_data = (mask_data > 0).astype(np.float32)

    results = []
    for i, ic_path in enumerate(ic_paths):
        ic_data, ic_img = load_nifti(ic_path)
        if ic_data.shape != reference_data.shape:
            ic_data = resample_to_reference(ic_img, reference_img)
        r = compute_correlation(reference_data, ic_data, mask_data)
        results.append({
            'IC_Number':       i + 1,
            'IC_Filename':     os.path.basename(ic_path),
            'Correlation':     r,
            'Abs_Correlation': abs(r) if not np.isnan(r) else np.nan,
            'IC_Path':         ic_path,
        })

    df = pd.DataFrame(results)
    sort_col = 'Abs_Correlation' if use_absolute else 'Correlation'
    df = df.sort_values(by=sort_col, ascending=False)
    df['Rank'] = range(1, len(df) + 1)
    df = df[['Rank', 'IC_Number', 'IC_Filename', 'Correlation', 'Abs_Correlation', 'IC_Path']]

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Rank ICs by atlas correlation")
    parser.add_argument("reference", help="Path to reference atlas NIfTI")
    parser.add_argument("ic_dir",    help="Directory containing split IC .nii.gz files")
    parser.add_argument("--mask",    help="Brain mask NIfTI (recommended)", default=None)
    parser.add_argument("--out",     help="Output CSV path", default=None)
    parser.add_argument("--signed",  action="store_true",
                        help="Rank by signed correlation instead of absolute value")
    args = parser.parse_args()

    results = rank_ics_by_correlation(
        reference_path=args.reference,
        ic_paths=args.ic_dir,
        mask_path=args.mask,
        use_absolute=not args.signed,
        output_csv=args.out,
    )

    print("\n=== TOP 10 COMPONENTS ===")
    print(results.head(10)[['Rank', 'IC_Filename', 'Correlation']].to_string(index=False))
    print(f"\nMean |r|: {results['Abs_Correlation'].mean():.4f}")
    print(f"Max  r:   {results['Correlation'].max():.4f}")


if __name__ == "__main__":
    main()
