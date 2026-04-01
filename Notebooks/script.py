import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Create a Script to create HDF5 files and two CSV file for every patient 
# Controls: n01, n02, n03, n04, n05, n09, n10, n11, ins1, ins2, ins3, ins4, ins5, ins6, ins7, ins8, ins9
# RBD patients: rbd02, rbd03, rbd05, rbd07, rbd08, rbd09, rbd10, rbd11, rbd12, rbd13, rbd17, rbd18, rbd19, rbd21, rbd22

# CSV File should have the following columns:
    # Patient ID
    # Group (Control or RBD)
    # each row should be a epoch of 30 seconds with the following columns:
    # Epoch Start Time
    # Epoch End Time
    # sleep stage (Wake, N1, N2, N3, REM)
    # USING EEG_BANDPOWER FUNCTION FOR ALL BANDS (DELTA, THETA, ALPHA, BETA, GAMMA)
        # delta_power (using eeg_bandpower function)
        # theta_power (using eeg_bandpower function)
        # alpha_power (using eeg_bandpower function)
        # beta_power (using eeg_bandpower function)
        # gamma_power (using eeg_bandpower function)
    # USING HRV_PER_EPOCH FUNCTION FOR ALL HRV METRICS (N_BEATS, HR_MEAN_BPM, RMSSD_MS, SDNN_MS, PNN50_PCT)
        # n_beats (using hrv_per_epoch function) 
        # hr_mean_bpm (using hrv_per_epoch function) 
        # rmssd_ms (using hrv_per_epoch function) 
        # sdnn_ms (using hrv_per_epoch function) 
        # pnn50_pct (using hrv_per_epoch function) 
    # USING HPC_METRIC FUNCTION FOR HFC, LFC, LFC/HFC
        # HFC (using hpc_metric function)
        # LFC (using hpc_metric function)
        # LFC/HFC (using hpc_metric function)

# The second CSV file should have the following columns:
# Patient ID
# Group (Control or RBD)
# Sleep stage (Wake, N1, N2, N3, REM) per row with the following columns:
# USING HEP_METRIC FUNCTION FOR HEP AMPLITUDE
# USING DELTA_POWER_1S FUNCTION FOR DELTA POWER 
# USING PEARSONR AND SPEARMANR FOR CORRELATION BETWEEN HEP AMPLITUDE AND DELTA POWER
# PER SLEEP STAGE (Wake, N1, N2, N3, REM)
    # Pearson correlation pearson_r  
    # Pearson correlation pearson_p
    # Spearman correlation spearman_r
    # Spearman correlation spearman_p

# HDF5 file (one per patient) should have the following structure:
# /Patient_ID
#     /Epochs
#         /Epoch_1
#             /Epoch Start Time
#             /Epoch End Time   
#             /sleep stage
#             /HEP values (30,)
#             /delta power values (30,) 
#             / Pearson correlation pearson_r 
#             / Pearson correlation pearson_p
#             / Spearman correlation spearman_r
#             / Spearman correlation spearman_p
#         /Epoch_2
#             /Epoch Start Time
#             /Epoch End Time   
#             /sleep stage
#             /HEP values (30,)
#             /delta power values (30,) 
#             / Pearson correlation pearson_r 
#             / Pearson correlation pearson_p
#             / Spearman correlation spearman_r
#             / Spearman correlation spearman_p

# PIPELINE ----------------------------------

# Load EDF 
# Load sleep staging (hypnogram)
# Define patient_id and group (Control / RBD)

# 1. TIME STRUCTURE
    # add epoch timeline (30s epochs) with add_epoch_onset()
    # extract epochs per sleep stage using compute_stage_epochs()
    # create stage_epochs_dict

# 2. SIGNAL PROCESSING
    # select revelent channels (EEG + ECG) using pick_core_channels()
    # preprocess EEG using preprocess_eeg()
    # preprocess ECG using preprocess_ecg()

# 3. ECG FEATURE EXTRACTION 
    # extract ECG segments per epochs using extract_ecg_per_epochs() -> rpeaks + HR 
    # compute HRV metrics per epoch using hrv_per_epoch()

# 4. RESPIRATION FOR CPC
    # extract respiration from ECG (full signal)

# 5. COMPUTE CPC 
    # calculate HFC LFC and ratio using hpc_metric()

# 6. EEG FEATURES 
    # extract bandpower for delta, theta, alpha, beta, gamma using eeg_bandpower()

# 7. HEP + DELTA (CORE ANALYSIS)
    # compute hep_vals using hep_metric()
    # compute delta_vals using delta_power_1s()

# 8. HEART-BRAIN COUPLING
    # compute Pearson + Spearman using plot_hep_delta_correlation()



"""Create per-patient feature exports (CSV + HDF5).

Outputs for each patient:
1) <patient_id>_epochs.csv
2) <patient_id>_stage_correlations.csv
3) <patient_id>.h5

The script reuses feature extractors defined in Notebooks/functions.py:
- eeg_bandpower
- hrv_per_epoch
- hpc_metric
- hep_metric
- delta_power_1s
"""


from Notebooks.functions import (
    add_epoch_onsets,
    compute_stage_epochs,
    delta_power_1s,
    eeg_bandpower,
    extract_ecg_per_epoch,
    extract_resp_from_ecg,
    hep_metric,
    hpc_metric,
    hrv_per_epoch,
    preprocess_ecg,
    preprocess_eeg,
)

CONTROL_IDS = [
    "n01", "n02", "n03", "n04", "n05", "n09", "n10", "n11",
    "ins1", "ins2", "ins3", "ins4", "ins5", "ins6", "ins7", "ins8", "ins9",
]

RBD_IDS = [
    "rbd02", "rbd03", "rbd05", "rbd07", "rbd08", "rbd09", "rbd10", "rbd11",
    "rbd12", "rbd13", "rbd17", "rbd18", "rbd19", "rbd21", "rbd22",
]

PATIENT_GROUP = {pid: "Control" for pid in CONTROL_IDS}
PATIENT_GROUP.update({pid: "RBD" for pid in RBD_IDS})

STAGE_LABEL_MAP = {
    "W": "Wake",
    "S1": "N1",
    "S2": "N2",
    "S3": "N3",
    "S4": "N3",
    "R": "REM",
}

