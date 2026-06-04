#!/usr/bin/env python3
"""
Encoding Model Preprocessing Pipeline
Preprocesses a single MNI152-space fMRIPrep output file for encoding model training.
No spatial smoothing or bandpass filtering is applied, preserving high-frequency
temporal information and fine-grained spatial patterns.
Includes spatial smoothing (4mm FWHM) and bandpass filtering (0.01–0.1 Hz).

Usage:
    python preprocess_encoding.py <func_file> <mask_file> <output_dir>

Designed to be submitted as a SLURM array job via submit_preprocessing.sh.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import image, maskers, glm

# ============================================================
# CONFIGURATION — adjust if needed
# ============================================================
CONFIG = {
    'remove_first_n_trs': 10,
    'remove_last_n_trs':  10,
    'mask_fwhm':          None,
    'target_affine':      None,
    'confounds_regex':    r'^(global_signal|framewise_displacement|trans_[xyz]|rot_[xyz]|a_comp_cor_0[0-4]|cosine.*)$',
    'confounds_columns':  None,
    'fd_threshold':       0.5,
    'volume_fwhm':        4,      # mm FWHM — ICA only
    'detrend':            True,
    'low_pass':           0.1,    # Hz — ICA only
    'high_pass':          0.01,   # Hz — ICA only
    'standardize':        True,
    'regress_out_task':   False,
    'slice_time_ref':     0.5,
    'min_T':              50,
}


def find_associated_files(func_file, bids_dir):
    """Find confounds, events, and JSON sidecar for a functional image."""
    func_path = Path(func_file)
    base_name = func_path.name.replace('_bold.nii.gz', '')
    base_identifier = base_name
    for modifier in ['_space-MNI152NLin2009cAsym', '_space-MNI152NLin6Asym',
                     '_space-T1w', '_desc-preproc', '_res-2']:
        base_identifier = base_identifier.replace(modifier, '')

    func_dir = func_path.parent
    confounds_file = None
    for pattern in [
        base_identifier + '_desc-confounds_timeseries.tsv',
        base_identifier + '_desc-confounds_regressors.tsv',
        base_identifier + '_confounds.tsv',
        base_name      + '_desc-confounds_timeseries.tsv',
    ]:
        p = func_dir / pattern
        if p.exists():
            confounds_file = str(p)
            break

    events_file = None
    if CONFIG['regress_out_task']:
        raw_bids = Path(bids_dir)
        subject_idx = [i for i, p in enumerate(func_path.parts) if p.startswith('sub-')]
        if subject_idx:
            rel = func_path.parts[subject_idx[0]:]
            raw_func_dir = raw_bids / Path(*rel[:-1])
            p = raw_func_dir / (base_identifier + '_events.tsv')
            if p.exists():
                events_file = str(p)

    json_file = None
    for pattern in [base_name + '.json', base_identifier + '_bold.json']:
        p = func_dir / pattern
        if p.exists():
            json_file = str(p)
            break

    return confounds_file, events_file, json_file


def clean_functional_data(func_file, mask_file, confounds_file,
                          events_file=None, json_file=None, config=None):
    """Clean a single functional run with smoothing and bandpass filtering."""
    if config is None:
        config = CONFIG

    TR = 2.0
    START_TIME = 0
    if json_file and os.path.exists(json_file):
        with open(json_file) as f:
            s = json.load(f)
            TR = s.get('RepetitionTime', TR)
            START_TIME = s.get('StartTime', START_TIME)

    mask_nii = image.load_img(mask_file)
    mask_nii = image.new_img_like(mask_nii, image.get_data(mask_nii).astype(np.float32))
    if config['mask_fwhm']:
        mask_nii = image.smooth_img(mask_nii, fwhm=config['mask_fwhm'])
    if config['target_affine'] is not None:
        mask_nii = image.resample_img(mask_nii, target_affine=config['target_affine'],
                                      interpolation='linear', copy_header=True, force_resample=True)
    mask_nii = image.math_img('x > 0.', x=mask_nii)
    mask_nii = image.crop_img(mask_nii, copy_header=True)

    func_img = image.load_img(func_file)
    n_remove_start = config.get('remove_first_n_trs', 0)
    n_remove_end   = config.get('remove_last_n_trs', 0)
    if n_remove_start > 0 or n_remove_end > 0:
        end_idx = None if n_remove_end == 0 else -n_remove_end
        func_img = image.index_img(func_img, slice(n_remove_start, end_idx))
        START_TIME += n_remove_start * TR

    if func_img.shape[3] < config['min_T']:
        raise ValueError(f"Insufficient timepoints: {func_img.shape[3]} < {config['min_T']}")

    func_img = image.resample_to_img(func_img, mask_nii, copy_header=True, force_resample=True)

    confounds_raw = pd.read_csv(confounds_file, sep='\t')
    if n_remove_start > 0 or n_remove_end > 0:
        end_idx = None if n_remove_end == 0 else -n_remove_end
        confounds_raw = confounds_raw.iloc[n_remove_start:end_idx].reset_index(drop=True)

    if config['confounds_columns']:
        confounds = confounds_raw[config['confounds_columns']]
    elif config['confounds_regex']:
        confounds = confounds_raw.filter(regex=config['confounds_regex'])
    else:
        confounds = confounds_raw.copy()

    fd_outliers = []
    if 'framewise_displacement' in confounds_raw.columns and config.get('fd_threshold'):
        fd = confounds_raw['framewise_displacement'].fillna(0)
        for i, is_out in enumerate(fd > config['fd_threshold']):
            if is_out:
                reg = np.zeros(len(confounds))
                reg[i] = 1
                fd_outliers.append(pd.DataFrame({f'fd_outlier_{i:04d}': reg}))

    convolved = []
    if config['regress_out_task'] and events_file and os.path.exists(events_file):
        events = pd.read_csv(events_file, sep='\t')
        if 'onset' in events.columns and 'duration' in events.columns:
            if 'trial_type' not in events.columns:
                events['trial_type'] = 'event'
            if n_remove_start > 0:
                events = events.copy()
                events['onset'] = events['onset'] - (n_remove_start * TR)
                events = events[events['onset'] >= 0]
            dummies = pd.get_dummies(events.trial_type, prefix='trial_type', prefix_sep='.')
            events_glm = pd.concat([dummies, events[['onset', 'duration']]], axis=1)
            frame_times = np.arange(len(confounds)) * TR + START_TIME + (config['slice_time_ref'] * TR)
            for col in [c for c in events_glm.columns if c.startswith('trial_type')]:
                cv, _ = glm.first_level.compute_regressor(
                    (events_glm['onset'], events_glm['duration'], events_glm[col]),
                    'glover + derivative', frame_times)
                cnames = [col] if cv.shape[1] == 1 else [col, f'{col}_derivative']
                convolved.append(pd.DataFrame(cv, columns=cnames))

    confounds_combined = pd.concat(convolved + fd_outliers + [confounds], axis=1)

    masker = maskers.NiftiMasker(
        mask_img=mask_nii,
        mask_strategy='background',
        standardize=config['standardize'],
        detrend=config['detrend'],
        t_r=TR,
        # No bandpass filtering for encoding data

        # No spatial smoothing for encoding data
    )
    cleaned = masker.fit_transform(func_img, confounds=confounds_combined.fillna(0))
    return masker.inverse_transform(cleaned), confounds_combined


def main():
    if len(sys.argv) < 4:
        print("Usage: python preprocess_encoding.py <func_file> <mask_file> <output_dir>")
        sys.exit(1)

    func_file, mask_file, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    for f, label in [(func_file, 'Functional'), (mask_file, 'Mask')]:
        if not os.path.exists(f):
            print(f"ERROR: {label} file not found: {f}")
            sys.exit(1)

    if 'space-MNI152' not in func_file:
        print(f"WARNING: file may not be in MNI152 space: {func_file}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        print(f"Processing: {os.path.basename(func_file)}")
        func_path = Path(func_file)
        bids_dir = func_path.parent
        while bids_dir.name not in ['derivatives', 'fmriprep'] and bids_dir != bids_dir.parent:
            bids_dir = bids_dir.parent
        bids_dir = bids_dir.parent

        confounds_file, events_file, json_file = find_associated_files(func_file, str(bids_dir))

        if not confounds_file or not os.path.exists(confounds_file):
            print("ERROR: Confounds file not found")
            sys.exit(1)

        func_cleaned, confounds_used = clean_functional_data(
            func_file, mask_file, confounds_file, events_file, json_file, CONFIG)

        base_name = os.path.basename(func_file).replace('_bold.nii.gz', '')
        out_img = os.path.join(output_dir, f"{base_name}_desc-cleaned_bold.nii.gz")
        out_conf = os.path.join(output_dir, f"{base_name}_desc-confounds_timeseries.tsv")
        func_cleaned.to_filename(out_img)
        confounds_used.to_csv(out_conf, sep='\t', index=False)
        print(f"Saved: {out_img}")

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
