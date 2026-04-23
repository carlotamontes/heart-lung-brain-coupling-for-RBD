# power eeg of 30s 
# overall 
# per patients (rbd control)
# per stage (wake, n1, n2, n3, rem)
# hrv statistic analysis 

# Create a Script to create HDF5 files and two CSV file for every patient 
# Controls: n1, n2, n3, n5, n10, n11, ins1, ins2, ins3, ins4, ins5, ins6, ins7, ins8, ins9
# RBD patients: rbd2, rbd3, rbd5, rbd7, rbd8, rbd9, rbd10, rbd11, rbd12, rbd13, rbd17, rbd18, rbd19, rbd21, rbd22

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
#             / Pearson correlation pearson_r 1 second window
#             / Pearson correlation pearson_p 1 second window
#             / Spearman correlation spearman_r 1 second window
#             / Spearman correlation spearman_p 1 second window
#             / Pearson correlation pearson_r 30 second window
#             / Pearson correlation pearson_p 30 second window
#             / Spearman correlation spearman_r 30 second window
#             / Spearman correlation spearman_p 30 second window
#         /Epoch_2
#             /Epoch Start Time
#             /Epoch End Time   
#             /sleep stage
#             / Pearson correlation pearson_r 1 second window
#             / Pearson correlation pearson_p 1 second window
#             / Spearman correlation spearman_r 1 second window
#             / Spearman correlation spearman_p 1 second window
#             / Pearson correlation pearson_r 30 second window
#             / Pearson correlation pearson_p 30 second window
#             / Spearman correlation spearman_r 30 second window
#             / Spearman correlation spearman_p 30 second window

# JUST ONE HDF5 GILE PER PATIENT WITH THE FOLLOWING STRUCTURE:

#/n01
# Group (Control or RBD)
#    /epochs
#        /epoch_0001
#            start_time
#            end_time
#            sleep_stage
#            eeg_bandpower
#                delta
#                theta
#                alpha
#                beta
#                gamma
#            hrv
#                n_beats
#                hr_mean_bpm
#                rmssd
#                sdnn
#                pnn50
#            cpc
#                hfc
#                lfc
#                ratio
#            hep
#                pearson_1s_p
#                spearman_1s_r
#                spearman_1s_p
#                pearson_30s_r
#                pearson_30s_p
#                spearman_30s_r
#                spearman_30s_p

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
    # compute hep_vals_1s using hep_metric()
    # compute delta_vals_1s using delta_power_1s()
    # compute hep_vals_30s using hep_metric_30s()
    # compute delta_vals_30s using delta_power_30s()

# 8. HEART-BRAIN COUPLING
    # compute Pearson + Spearman using plot_hep_delta_correlation() for 1s arrays HEP and delta 
    # compute Pearson + Spearman using plot_hep_delta_correlation() for 30s arrays HEP and delta 
    # compute Pearson + Spearman using plot_hep_delta_correlation() for 1s arrays HEP and delta for each sleep stage (Wake, N1, N2, N3, REM)
    # compute Pearson + Spearman using plot_hep_delta_correlation() for 1s arrays HEP and delta for each sleep stage (Wake, N1, N2, N3, REM)



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
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from functions import (
    add_epoch_onsets,
    compute_stage_epochs,
    delta_power_1s,
    delta_power_30s,
    eeg_bandpower,
    extract_ecg_per_epoch,
    extract_resp_from_ecg,
    hep_metric,
    hep_metric_30s,
    hpc_metric,
    hrv_per_epoch,
    preprocess_ecg,
    preprocess_eeg,
    hep_waveform_mean,
    hep_waveform_max,
    hep_waveform_time_axis,
)


# MAIN FUNCTION: process ONE patient → save to HDF5

def process_patient_to_hdf5(
    raw,
    patient_id,
    group,
    df,                  # hypnogram dataframe
    output_dir           # where to save the HDF5 file
):

    # 1. BASIC INFO
    sf = raw.info["sfreq"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / f"{patient_id}.h5"

    # 2. PREPROCESS SIGNALS
    eeg_filtered = preprocess_eeg(raw)
    ecg_clean    = preprocess_ecg(raw)

    # 3. GET EPOCHS PER STAGE
    epoch_len = 30

    # convert df → epoch list
    epochs = []
    sleep_stages = []

    stage_map = {
        "W": "Wake",
        "R": "REM",
        "S1": "N1",
        "S2": "N2",
        "S3": "N3",
        "S4": "N3"
    }

    # Keep only stage-scoring rows that map cleanly to full 30 s epochs in the EDF.
    df = df.copy()

    if "Event" in df.columns:
        sleep_event_mask = df["Event"].astype(str).str.startswith("SLEEP-", na=False)
        if sleep_event_mask.any():
            df = df.loc[sleep_event_mask].copy()

    if "Duration[s]" in df.columns:
        durations = pd.to_numeric(df["Duration[s]"], errors="coerce")
        df = df.loc[np.isclose(durations, epoch_len)].copy()
        df["Duration[s]"] = durations.loc[df.index]

    df = df.loc[df["Sleep Stage"].astype(str).isin(stage_map)].copy()
    df = df.reset_index(drop=True)
    df = add_epoch_onsets(df, epoch_len)

    raw_duration_s = raw.n_times / sf
    df = df.loc[(df["onset_s"] + epoch_len) <= (raw_duration_s + 1e-9)].copy()
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid 30 s sleep-stage epochs remain after filtering the hypnogram.")

    for _, row in df.iterrows():
            t0 = row["onset_s"]
            t1 = t0 + epoch_len

            stage = stage_map.get(row["Sleep Stage"], row["Sleep Stage"])

            epochs.append({
                "t0_s": t0,
                "t1_s": t1
            })

            sleep_stages.append(stage)

    # 5. EXTRACT ECG PER EPOCH (ONCE)
    epochs_tuples = [(ep["t0_s"], ep["t1_s"]) for ep in epochs]

    ecg_df = extract_ecg_per_epoch(ecg_clean, ecg_clean, epochs_tuples, sf)

    hrv_df = hrv_per_epoch(ecg_clean, epochs_tuples, sf)

    resp = extract_resp_from_ecg(ecg_clean, sf)

    cpc_window_epochs = 5
    cpc_df = hpc_metric(ecg_clean, resp, epochs_tuples, sf, window_epochs=cpc_window_epochs)


    stage_epochs_dict = {
    "W": [(ep["t0_s"], ep["t1_s"]) for ep, s in zip(epochs, sleep_stages) if s == "Wake"],
    "R": [(ep["t0_s"], ep["t1_s"]) for ep, s in zip(epochs, sleep_stages) if s == "REM"],
    "S1": [(ep["t0_s"], ep["t1_s"]) for ep, s in zip(epochs, sleep_stages) if s == "N1"],
    "S2": [(ep["t0_s"], ep["t1_s"]) for ep, s in zip(epochs, sleep_stages) if s == "N2"],
    "S3": [(ep["t0_s"], ep["t1_s"]) for ep, s in zip(epochs, sleep_stages) if s == "N3"],
    }

    

    bp_df = eeg_bandpower(eeg_filtered, stage_epochs_dict, sf)

    print("BP DF length:", len(bp_df))

    # Summary accumulators for patient-level HEP waveform metrics.
    nrem_stages = {"N1", "N2", "N3"}
    n_epochs_all = len(epochs)
    n_epochs_rem = sum(stage == "REM" for stage in sleep_stages)
    n_epochs_nrem = sum(stage in nrem_stages for stage in sleep_stages)
    n_valid_hep_epochs = 0

    tmin = -0.2
    tmax = 0.6
    hep_time = hep_waveform_time_axis(sf=sf, tmin=tmin, tmax=tmax)

    hep_mean_all = []
    hep_mean_rem = []
    hep_mean_nrem = []
    hep_30s_all = []
    delta_30s_all = []
    hep_30s_rem = []
    delta_30s_rem = []
    hep_30s_nrem = []
    delta_30s_nrem = []

    # 6. CREATE HDF5
    with h5py.File(h5_path, "w") as f:

        patient_group = f.create_group(patient_id)
        patient_group.attrs["group"] = group
        patient_group.attrs["sfreq"] = sf

        epochs_group = patient_group.create_group("epochs")

        # 7. LOOP THROUGH EPOCHS
        for i in range(len(epochs)):

            ep = epochs[i]
            stage = sleep_stages[i]
            ecg_epoch = ecg_df.iloc[i]

            ep_group = epochs_group.create_group(f"epoch_{i:04d}")
            # epoch_0001, epoch_0002, ..., epoch_0010
            # f["n01/epochs/epoch_0023"]

            # ---- BASIC INFO ----
            ep_group.create_dataset("start_time", data=ep["t0_s"])
            ep_group.create_dataset("end_time", data=ep["t1_s"])
            ep_group.create_dataset("sleep_stage", data=stage.encode("utf-8"))

            # ---- EEG BANDPOWER ----
            bp_group = ep_group.require_group("eeg_bandpower")

            row = bp_df[np.isclose(bp_df["start_s"], ep["t0_s"])]

            if not row.empty:
                row = row.iloc[0]
                for band in ["delta", "theta", "alpha", "beta", "gamma"]:
                    bp_group.create_dataset(band, data=row[f"{band}_power"])
            else:
                for band in ["delta", "theta", "alpha", "beta", "gamma"]:
                    bp_group.create_dataset(band, data=np.nan)

            # ---- HRV INFO ----
            hrv_group = ep_group.require_group("hrv")

            row = hrv_df.iloc[i]

            for key in ["n_beats", "hr_mean_bpm", "rmssd_ms", "sdnn_ms", "pnn50_pct"]:
                hrv_group.create_dataset(key, data=row.get(key, np.nan))

            # ---- CPC INFO ----
            cpc_group = ep_group.require_group("cpc")

            cpc_epoch = i - (i % cpc_window_epochs)
            row = cpc_df[cpc_df["epoch"] == cpc_epoch]

            if not row.empty:
                row = row.iloc[0]
                cpc_group.create_dataset("hfc", data=row.get("HFC", np.nan))
                cpc_group.create_dataset("lfc", data=row.get("LFC", np.nan))
                cpc_group.create_dataset("ratio", data=row.get("LFC/HFC", np.nan))
            else:
                cpc_group.create_dataset("hfc", data=np.nan)
                cpc_group.create_dataset("lfc", data=np.nan)
                cpc_group.create_dataset("ratio", data=np.nan)

            # ---- HEP + DELTA ----
            try:
                hep_group = ep_group.create_group("hep")

                # 1s
                hep_1s   = hep_metric(eeg_filtered, ecg_epoch, sf=sf)
                delta_1s = delta_power_1s(
                    eeg_filtered,
                    epoch=(ep["t0_s"], ep["t1_s"]),
                    sf=sf,
                )

                if hep_1s is not None and delta_1s is not None:
                    hep_arr = np.asarray(hep_1s, dtype=float)
                    delta_arr = np.asarray(delta_1s, dtype=float)

                    n_common = min(len(hep_arr), len(delta_arr))
                    hep_arr = hep_arr[:n_common]
                    delta_arr = delta_arr[:n_common]

                    valid_mask = np.isfinite(hep_arr) & np.isfinite(delta_arr)

                    if valid_mask.sum() >= 3:
                        r_p, p_p = pearsonr(hep_arr[valid_mask], delta_arr[valid_mask])
                        r_s, p_s = spearmanr(hep_arr[valid_mask], delta_arr[valid_mask])
                    else:
                        r_p = p_p = r_s = p_s = np.nan
                else:
                    r_p = p_p = r_s = p_s = np.nan

                hep_group.create_dataset("pearson_1s_r", data=r_p)
                hep_group.create_dataset("pearson_1s_p", data=p_p)
                hep_group.create_dataset("spearman_1s_r", data=r_s)
                hep_group.create_dataset("spearman_1s_p", data=p_s)

                # 30s (values)
                hep_30s   = hep_metric_30s(eeg_filtered, ecg_epoch, sf=sf)
                delta_30s = delta_power_30s(eeg_filtered, epoch=ep, sf=sf)

                hep_group.create_dataset("hep_30s", data=hep_30s if hep_30s is not None else np.nan)
                hep_group.create_dataset("delta_30s", data=delta_30s if delta_30s is not None else np.nan)

                hep_30s_value = float(hep_30s) if hep_30s is not None and np.isfinite(hep_30s) else np.nan
                delta_30s_value = float(delta_30s) if delta_30s is not None and np.isfinite(delta_30s) else np.nan

                hep_30s_all.append(hep_30s_value)
                delta_30s_all.append(delta_30s_value)

                if stage == "REM":
                    hep_30s_rem.append(hep_30s_value)
                    delta_30s_rem.append(delta_30s_value)
                elif stage in nrem_stages:
                    hep_30s_nrem.append(hep_30s_value)
                    delta_30s_nrem.append(delta_30s_value)

                if hep_30s is not None and np.isfinite(hep_30s):
                    n_valid_hep_epochs += 1

            except Exception as e:
                print(f"[HEP] Epoch {i} failed: {e}")
            
            # ---- HEP WAVEFORMS ----
            try:
                wf_group = ep_group.create_group("waveform")

                # ---- MEAN waveform ----
                hep_mean = hep_waveform_mean(eeg_filtered, ecg_epoch, sf=sf)

                mean_group = wf_group.create_group("mean")

                if hep_mean is not None:
                    mean_group.create_dataset("signal", data=hep_mean)
                    mean_group.create_dataset("time", data=hep_time)

                    hep_mean_array = np.asarray(hep_mean, dtype=float)
                    if np.all(np.isfinite(hep_mean_array)):
                        hep_mean_all.append(hep_mean_array)
                        if stage == "REM":
                            hep_mean_rem.append(hep_mean_array)
                        elif stage in nrem_stages:
                            hep_mean_nrem.append(hep_mean_array)
                else:
                    mean_group.create_dataset("signal", data=np.full(len(hep_time), np.nan))
                    mean_group.create_dataset("time", data=hep_time)

                # ---- MAX waveform ----
                hep_max = hep_waveform_max(eeg_filtered, ecg_epoch, sf=sf)

                max_group = wf_group.create_group("max")

                if hep_max is not None:
                    max_group.create_dataset("signal", data=hep_max)
                    max_group.create_dataset("time", data=hep_time)
                else:
                    max_group.create_dataset("signal", data=np.full(len(hep_time), np.nan))
                    max_group.create_dataset("time", data=hep_time)

            except Exception as e:
                print(f"[Waveform] Epoch {i} failed: {e}")

        # ---- PATIENT-LEVEL SUMMARY ----
        patient_group.create_dataset("n_epochs_all", data=n_epochs_all)
        patient_group.create_dataset("n_epochs_rem", data=n_epochs_rem)
        patient_group.create_dataset("n_epochs_nrem", data=n_epochs_nrem)
        patient_group.create_dataset("n_valid_hep_epochs", data=n_valid_hep_epochs)
        patient_group.create_dataset("mean_delta_power_all", data=float(np.nanmean(delta_30s_all)) if len(delta_30s_all) > 0 and np.isfinite(delta_30s_all).any() else np.nan)
        patient_group.create_dataset("mean_delta_power_REM", data=float(np.nanmean(delta_30s_rem)) if len(delta_30s_rem) > 0 and np.isfinite(delta_30s_rem).any() else np.nan)
        patient_group.create_dataset("mean_delta_power_NREM", data=float(np.nanmean(delta_30s_nrem)) if len(delta_30s_nrem) > 0 and np.isfinite(delta_30s_nrem).any() else np.nan)
        patient_group.create_dataset("mean_hep_all", data=float(np.nanmean(hep_30s_all)) if len(hep_30s_all) > 0 and np.isfinite(hep_30s_all).any() else np.nan)
        patient_group.create_dataset("mean_hep_REM", data=float(np.nanmean(hep_30s_rem)) if len(hep_30s_rem) > 0 and np.isfinite(hep_30s_rem).any() else np.nan)
        patient_group.create_dataset("mean_hep_NREM", data=float(np.nanmean(hep_30s_nrem)) if len(hep_30s_nrem) > 0 and np.isfinite(hep_30s_nrem).any() else np.nan)

        hep_waveform_group = patient_group.create_group("hep_waveform_derived")
        hep_waveform_group.create_dataset("time", data=hep_time)

        waveform_mean_all = np.mean(np.vstack(hep_mean_all), axis=0) if hep_mean_all else np.full(len(hep_time), np.nan)
        waveform_mean_rem = np.mean(np.vstack(hep_mean_rem), axis=0) if hep_mean_rem else np.full(len(hep_time), np.nan)
        waveform_mean_nrem = np.mean(np.vstack(hep_mean_nrem), axis=0) if hep_mean_nrem else np.full(len(hep_time), np.nan)

        hep_waveform_group.create_dataset("hep_waveform_mean_all", data=waveform_mean_all)
        hep_waveform_group.create_dataset("hep_waveform_mean_REM", data=waveform_mean_rem)
        hep_waveform_group.create_dataset("hep_waveform_mean_NREM", data=waveform_mean_nrem)

        # Use the post-R interval only for latency and AUC.
        post_r_mask = (hep_time >= 0.0) & (hep_time <= 0.6)
        hep_waveform_group.create_dataset("peak_window_start_s", data=0.0)
        hep_waveform_group.create_dataset("peak_window_end_s", data=0.6)
        hep_waveform_group.create_dataset("auc_window_start_s", data=0.0)
        hep_waveform_group.create_dataset("auc_window_end_s", data=0.6)

        all_wave = waveform_mean_all[post_r_mask]
        all_time = hep_time[post_r_mask]
        all_finite = np.isfinite(all_wave)
        if np.any(all_finite):
            all_wave_valid = all_wave[all_finite]
            all_time_valid = all_time[all_finite]
            hep_peak_latency_all = float(all_time_valid[np.argmax(all_wave_valid)])
            hep_auc_all = float(np.trapezoid(all_wave_valid, all_time_valid))
        else:
            hep_peak_latency_all = np.nan
            hep_auc_all = np.nan

        rem_wave = waveform_mean_rem[post_r_mask]
        rem_time = hep_time[post_r_mask]
        rem_finite = np.isfinite(rem_wave)
        if np.any(rem_finite):
            rem_wave_valid = rem_wave[rem_finite]
            rem_time_valid = rem_time[rem_finite]
            hep_peak_latency_rem = float(rem_time_valid[np.argmax(rem_wave_valid)])
            hep_auc_rem = float(np.trapezoid(rem_wave_valid, rem_time_valid))
        else:
            hep_peak_latency_rem = np.nan
            hep_auc_rem = np.nan

        nrem_wave = waveform_mean_nrem[post_r_mask]
        nrem_time = hep_time[post_r_mask]
        nrem_finite = np.isfinite(nrem_wave)
        if np.any(nrem_finite):
            nrem_wave_valid = nrem_wave[nrem_finite]
            nrem_time_valid = nrem_time[nrem_finite]
            hep_peak_latency_nrem = float(nrem_time_valid[np.argmax(nrem_wave_valid)])
            hep_auc_nrem = float(np.trapezoid(nrem_wave_valid, nrem_time_valid))
        else:
            hep_peak_latency_nrem = np.nan
            hep_auc_nrem = np.nan

        hep_waveform_group.create_dataset("hep_peak_latency_all", data=hep_peak_latency_all)
        hep_waveform_group.create_dataset("hep_peak_latency_REM", data=hep_peak_latency_rem)
        hep_waveform_group.create_dataset("hep_peak_latency_NREM", data=hep_peak_latency_nrem)
        hep_waveform_group.create_dataset("hep_auc_all", data=hep_auc_all)
        hep_waveform_group.create_dataset("hep_auc_REM", data=hep_auc_rem)
        hep_waveform_group.create_dataset("hep_auc_NREM", data=hep_auc_nrem)

        corr_group = patient_group.create_group("correlations")

        def compute_epochwise_corr(x_vals, y_vals):
            x = np.asarray(x_vals, dtype=float)
            y = np.asarray(y_vals, dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) < 3:
                return np.nan, np.nan, np.nan, np.nan
            pearson_r, pearson_p = pearsonr(x[mask], y[mask])
            spearman_r, spearman_p = spearmanr(x[mask], y[mask])
            return float(pearson_r), float(pearson_p), float(spearman_r), float(spearman_p)

        pearson_r, pearson_p, spearman_r, spearman_p = compute_epochwise_corr(hep_30s_all, delta_30s_all)
        corr_group.create_dataset("pearson_30s_r_all", data=pearson_r)
        corr_group.create_dataset("pearson_30s_p_all", data=pearson_p)
        corr_group.create_dataset("spearman_30s_r_all", data=spearman_r)
        corr_group.create_dataset("spearman_30s_p_all", data=spearman_p)

        pearson_r, pearson_p, spearman_r, spearman_p = compute_epochwise_corr(hep_30s_rem, delta_30s_rem)
        corr_group.create_dataset("pearson_30s_r_REM", data=pearson_r)
        corr_group.create_dataset("pearson_30s_p_REM", data=pearson_p)
        corr_group.create_dataset("spearman_30s_r_REM", data=spearman_r)
        corr_group.create_dataset("spearman_30s_p_REM", data=spearman_p)

        pearson_r, pearson_p, spearman_r, spearman_p = compute_epochwise_corr(hep_30s_nrem, delta_30s_nrem)
        corr_group.create_dataset("pearson_30s_r_NREM", data=pearson_r)
        corr_group.create_dataset("pearson_30s_p_NREM", data=pearson_p)
        corr_group.create_dataset("spearman_30s_r_NREM", data=spearman_r)
        corr_group.create_dataset("spearman_30s_p_NREM", data=spearman_p)

    print(f"Saved → {h5_path}")
