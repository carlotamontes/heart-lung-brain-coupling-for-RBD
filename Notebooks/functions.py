from multiprocessing.resource_sharer import stop
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import threshold
import neurokit2 as nk
import pyedflib
from mne.time_frequency import psd_array_welch
import mne
from IPython.display import display
from scipy.signal import welch
from scipy.signal import coherence, csd


def compute_stage_epochs(df, stage_label):
    # (t0, t1) = (onset_s, onset_s + duration)

    stage_df = df[df["Sleep Stage"] == stage_label].copy()

    if stage_df.empty:
        return []

    epochs = list(zip(
        stage_df["onset_s"].astype(float),
        stage_df["onset_s"] + stage_df["Duration[s]"]
    ))

    return epochs

def add_epoch_onsets(df, epoch_len):
    # add all epochs to the data frame 
    df = df.copy()
    df["onset_s"] = np.arange(len(df)) * epoch_len
    return df

def pick_core_channels(raw):
    core_chs = ["F3-C3", "F4-C4", "ECG1-ECG2"]
    return raw.copy().pick(core_chs)

def preprocess_eeg(raw, notch_freqs=(50, 100), l_freq=0.3, h_freq=35.0):
    eeg = raw.copy()
    if notch_freqs is not None:
        # Notch filter at 50 and 100 Hz
        # Remove power line noise
        eeg.notch_filter(notch_freqs, fir_design='firwin', verbose=False)

    #Bandpass 0.3–35 Hz
    eeg.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    return eeg

def preprocess_ecg(raw, method="neurokit", lowcut=0.5, highcut=45):
    sf = raw.info["sfreq"] # sampling frequency
    ecg = raw.get_data(picks="ECG1-ECG2")[0]

    # Clean using NeuroKit2's ecg_clean function 
    # Bandpass 0.5–45 Hz
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=sf, method=method, lowcut=lowcut, highcut=highcut)

    return ecg_clean

def compute_psd(raw, fmin=0.5, fmax=40,  n_per_seg=2048, n_fft=4096,  n_overlap=1024):

    # n_per_seg: Length of each segment for Welch's method 
    # window length = 2048 / 512 = 4.0 seconds

    # n_fft: Length of the FFT used 
    # FFT length = 4096 / 512 = 8.0 seconds worth of zero-padding

    # Δf = sf / n_fft = 512 / 4096 = 0.125 Hz
    # 0.5 Hz = 1 cycle every 2 seconds
    # 4 seconds -> 2 cycles

    # n_overlap: Number of points to overlap between segments 
    # overlap = 1024 / 512 = 2.0 seconds

    X = raw.get_data() # shape (n_channels, n_samples)
    sf = raw.info["sfreq"] 

    # psd shape (n_channels, n_freqs)
    # freqs shape (n_freqs,)

    psd, freqs = psd_array_welch(X, sfreq=sf, fmin=fmin, fmax=fmax, n_per_seg=n_per_seg,n_fft=n_fft, n_overlap=n_overlap, verbose=False)
    # psd is in V^2 / Hz
    # V^2 / Hz -> µV^2 / Hz by multiplying by 1e12
    psd = psd * 1e12
    
    return psd, freqs

def summarize_psd(raw_data, filtered_data, ch_names):
    for i in (range(len(ch_names))):
        # mean and std of PSD across frequencies for each channel - RAW
        raw_mean = raw_data[i].mean()
        raw_std = raw_data[i].std()

        # mean and std of PSD across frequencies for each channel - CLEANED
        filt_mean = filtered_data[i].mean()
        filt_std = filtered_data[i].std()

        print(f"{ch_names[i]}: {raw_mean:.2e} ± {raw_std:.2e} µV²/Hz")
        print(f"{ch_names[i]}: {filt_mean:.2e} ± {filt_std:.2e} µV²/Hz")
        # if std is much smaller than mean -> more stable 
        # if std is large compared to mean ->  more variability 

def plot_psd_comparison(freqs, psd_raw, psd_filt, ch_names):

    plt.figure(figsize=(10,5))

    for i, ch in enumerate(ch_names):

        # plotting PSD on a log scale to better visualize differences across frequencies
        plt.semilogy(freqs, psd_raw[i], linestyle="--", linewidth=1, alpha=0.7, label=f"{ch} RAW")

        plt.semilogy(freqs, psd_filt[i], linewidth=1, alpha=0.7, label=f"{ch} FILTERED")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (µV²/Hz)")
    plt.title("EEG PSD Comparison")
    plt.xlim(0.5, 40)
    plt.legend()
    plt.show()

def eeg_bandpower_per_epoch(eeg_filtered, epochs, sf):
    # Define frequency bands of interest
    bands = { "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta":  (12, 30), "gamma": (30, 40) }
    
    rows = []

    for i, (t0, t1) in enumerate(epochs):
        # epochs -> [(t0, t1), (t0, t1), ...]
        a, b = int(t0*sf), int(t1*sf)
        # MNE raw object (filtered) -> numpy array for the epoch
        X = eeg_filtered.get_data(start=a, stop=b) 

        psd, freqs = psd_array_welch(X, sfreq=sf, fmin=0.5, fmax=40, verbose=False)

        row = {"epoch": i, "start_s": float(t0)}

        for name, (f1, f2) in bands.items():
            idx = (freqs >= f1) & (freqs <= f2)
            # Integrate PSD over the band using the trapezoidal rule to get band power
            bp_ch = np.trapezoid(psd[:, idx], freqs[idx], axis=1)  
            row[f"{name}_power"] = float(bp_ch.mean())      
        rows.append(row) 
        # Data Frame -> epoch index , start time, delta power, theta power, alpha power, beta power, gamma power

    return pd.DataFrame(rows)

def eeg_bandpower_all_stages(raw, stage_epochs_dict, sf):
    dfs = []
    for stage, epochs in stage_epochs_dict.items():
        # ("N2", [(t0,t1), (t0,t1)]) ("REM", [(t0,t1)]) ("N3", [])
        if len(epochs) == 0:
            continue
        # For each stage, compute bandpower per epoch and add a column for the stage label
        tmp = eeg_bandpower_per_epoch(raw, epochs, sf)
        tmp["stage"] = stage
        dfs.append(tmp)

    return pd.concat(dfs, ignore_index=True)

def extract_ecg_per_epoch(ecg_sig_total, ecg_clean_total, epochs, sf, detect_peaks=True):
    rows = []
    for i, (t0, t1) in enumerate(epochs):

        # Convert epoch start and end times from seconds to sample indices
        a, b = int(t0 * sf), int(t1 * sf)
        # t0 = 30.0s, sf = 512 Hz -> a = 15360

        x_raw = ecg_sig_total[a:b]
        x_clean = ecg_clean_total[a:b]

        try:
            if detect_peaks:
                # Detect R-peaks in the cleaned ECG segment using NeuroKit2
                _, info = nk.ecg_peaks(x_clean, sampling_rate=sf)
                rpeaks = info["ECG_R_Peaks"]

                # beats per minute (bpm) = 60 * number of peaks / duration in seconds
                n_peaks = int(len(rpeaks))
                hr_mean = 60.0 * n_peaks / (t1 - t0)

            else:
                rpeaks = None
                n_peaks = 0

            rows.append({"epoch": i, "t0_s": float(t0), "t1_s": float(t1), "dur_s": float(t1 - t0), "raw_seg": x_raw, "clean_seg": x_clean, "rpeaks": rpeaks, "n_peaks": n_peaks, "hr_mean_bpm": hr_mean, "ok": True})

        except Exception as e:
            rows.append({"epoch": i,  "t0_s": float(t0), "t1_s": float(t1), "dur_s": float(t1 - t0), "raw_seg": x_raw, "clean_seg": x_clean, "rpeaks": None, "n_peaks": 0,"hr_mean_bpm": hr_mean,   "ok": False, "error": str(e)})

    return pd.DataFrame(rows)

def hrv_per_epoch(ecg, epochs, sf):
    rows = []
    for i, (t0, t1) in enumerate(epochs):
        a, b = int(t0*sf), int(t1*sf)
        x = ecg[a:b]
        try:
            # Process the ECG segment to extract HRV metrics using NeuroKit2
            signals, info = nk.ecg_process(x, sampling_rate=sf)
            # mean HR from ECG_Rate (bpm)
            hr_mean = float(np.nanmean(signals["ECG_Rate"]))

            # HRV metrics (time domain) from detected R-peaks
            hrv = nk.hrv_time(info, sampling_rate=sf).iloc[0].to_dict()

            rows.append({"epoch": i, "start_s": float(t0), "hr_mean_bpm": hr_mean, "rmssd_ms": float(hrv.get("HRV_RMSSD", np.nan)), "sdnn_ms": float(hrv.get("HRV_SDNN", np.nan)), "pnn50_pct": float(hrv.get("HRV_pNN50", np.nan)), "n_beats": int(len(info["ECG_R_Peaks"])) if "ECG_R_Peaks" in info else np.nan, "ok": True})
        except Exception as e:
            rows.append({"epoch": i, "start_s": float(t0), "ok": False, "error": str(e)})

    return pd.DataFrame(rows)

def extract_resp_from_ecg(ecg, sf, method="neurokit"):
    # Extract respiratory signal from ECG
    # resp : 1D numpy array
    # NeuroKit extracts low-frequency modulation of ECG
    # caused by breathing (respiratory sinus arrhythmia and thoracic impedance effects).
    resp = nk.ecg_rsp(ecg, sampling_rate=sf)

    # 0.1–0.4 Hz (6–24 breaths per minute)
    # Remove very slow baseline drift (<0.05 Hz)
    # Remove high-frequency noise (>0.7 Hz)
    resp = nk.signal_filter(resp, sampling_rate=sf, lowcut=0.05, highcut=0.7)
    return resp

def hpc_metric(ecg, resp, epochs, sf, window_epochs):
    # ecg : 1D numpy array
    # resp : 1D numpy array
    # epochs : list of tuples
    # sf : float
    # window_epochs : int

    rows = []
    fs_cpc = 4  # 4 Hz is typical for HRV spectral analysis
    for i in range(0, len(epochs), window_epochs):
        # to select a window of epochs 
        group = epochs[i:i+window_epochs]

        if len(group) < window_epochs:
            continue  

        t0 = group[0][0]
        t1 = group[-1][1]

        a, b = int(t0*sf), int(t1*sf)

        x_ecg = ecg[a:b]
        x_resp = resp[a:b]

        try:
             # Extract rpeaks of the segment using NeuroKit2
             _, info = nk.ecg_peaks(x_ecg, sampling_rate=sf)
             rpeaks = info["ECG_R_Peaks"]

             rr = np.diff(rpeaks) / sf  # RR intervals in seconds
             t_rr = np.cumsum(rr)  # Time points of RR intervals

             # Resample RR intervals and respiratory signal to a common time grid 
             # Create uniform time grid at fs_cpc Hz
             t_grid = np.arange(t_rr[0], t_rr[-1], 1/fs_cpc)

             # Interpolate RR intervals onto uniform grid
             rr_resampled = np.interp(t_grid, t_rr, rr)  # Resample RR intervals to common grid
            
             # time axis for respiration
             t_resp_grid = np.arange(len(x_resp))/sf
             # Interpolate respiration to same time grid 
             x_resp_resampled = np.interp(t_grid, t_resp_grid, x_resp)  

             # Number of samples per segment
             nps = min(64, len(rr_resampled))

             # Cross-spectral density between RR and respiration 
             f, Cxy = csd(rr_resampled, x_resp_resampled, fs=fs_cpc, nperseg=nps)

             # Coherence between RR and respiration
             _, Coh = coherence(rr_resampled, x_resp_resampled, fs=fs_cpc, nperseg=nps)

             # CPC spectrum as magnitude of cross-spectrum weighted by coherence
             CPC = np.abs(Cxy) * Coh

             lf_mask = (f >= 0.01) & (f <= 0.15)
             hf_mask = (f >= 0.15) & (f <= 0.40)
             HFC = np.trapezoid(CPC[hf_mask], f[hf_mask])
             LFC = np.trapezoid(CPC[lf_mask], f[lf_mask])
             LFC_HFC_ratio = LFC / HFC if HFC > 0 else np.nan

             rows.append({"epoch": i, "HFC": HFC, "LFC": LFC, "LFC/HFC": LFC_HFC_ratio, "ok": True})

        except Exception as e:
             rows.append({"epoch": i, "start_s": float(t0), "ok": False, "error": str(e)})
             continue   

    return pd.DataFrame(rows)

def classify_sleep_stable_unstable(hpc_df):

    df = hpc_df.copy()

    # if HFC > LFC -> stable, else unstable
    df["sleep_stability"] = np.where(df["HFC"] > df["LFC"], "stable", "unstable")

    # if both HFC and LFC are zero -> undefined
    df.loc[(df["HFC"] == 0) & (df["LFC"] == 0), "sleep_stability"] = "undefined"
    # if either HFC or LFC is NaN -> undefined
    df.loc[(df["HFC"].isna()) | (df["LFC"].isna()), "sleep_stability"] = "undefined"

    return df

def hep_metric(eeg, ecg, epochs, sf):
    # time window around R-peaks for HEP extraction
    tmin = -0.2
    tmax = 0.6
    rows = []
    # compute rpeaks for each epoch and store in a DataFrame
    ecg_epochs = extract_ecg_per_epoch(ecg, ecg, epochs, sf, detect_peaks=True) 
    # extract full EEG data as numpy array
    eeg_data = eeg.get_data()

    for i, (t0, t1) in enumerate(epochs):
        hep_segments = []
        epoch_data = ecg_epochs.iloc[i]
        if epoch_data.empty or not epoch_data["ok"]:
            continue         
        
        if epoch_data["rpeaks"] is None or len(epoch_data["rpeaks"]) < 2:
            continue

        # r is relative to the epoch start
        for r in epoch_data["rpeaks"]:
            r_global = r + t0 * sf  # Convert r to global time index
            # Extract EEG segment around the R-peak
            start = int(r_global + tmin * sf)
            stop  = int(r_global + tmax * sf)
            x_ecg = ecg[start:stop]

            segment = eeg_data[:, start:stop] # (n_channels, window_length_samples)

            hep_segments.append(segment)

        if len(hep_segments) == 0:
            continue

        hep_epochs = np.array(hep_segments)  

        hep_avg = hep_epochs.mean(axis=0) # (n_channels, window_samples)

        hep_mean_amp = hep_avg.mean()
 
        rows.append({"epoch": i, "hep_mean_amp": float(hep_mean_amp), "n_beats_used": len(hep_epochs), "ok": True})

    return pd.DataFrame(rows)

def HEP_Delta_plot_selected(hep_df, eeg_band_df, epoch_indices):

    merged = pd.merge(hep_df, eeg_band_df, on="epoch")
    merged = merged[merged["epoch"].isin(epoch_indices)]
    merged = merged.sort_values("start_s").reset_index(drop=True)

    x = (merged["start_s"] - merged["start_s"].iloc[0]) / 60

    fig, ax1 = plt.subplots(figsize=(8,4))

    ax1.plot(x,
             merged["hep_mean_amp"],
             color="blue",
             marker="o")

    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("HEP Mean Amplitude (µV)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(x,
             merged["delta_power"],
             color="red",
             marker="o")

    ax2.set_ylabel("Delta Power (µV²)", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    plt.title(f"HEP and Delta (Epochs {epoch_indices})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
