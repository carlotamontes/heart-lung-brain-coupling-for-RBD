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
from scipy.signal import butter, detrend, freqs, welch
from scipy.signal import coherence, csd
from scipy.stats import pearsonr, spearmanr


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
    core_chs = ["F4-C4", "ECG1-ECG2"]
    return raw.copy().pick(core_chs)

def preprocess_eeg(raw, notch_freqs=(50, 100), l_freq=0.3, h_freq=35.0):
    eeg = raw.copy()
    eeg = eeg.pick(["F4-C4"])  # pick only the EEG channel for preprocessing
    if notch_freqs is not None:
        # Notch filter at 50 and 100 Hz
        # Remove power line noise
        eeg.notch_filter(notch_freqs, fir_design='firwin', verbose=False)

    #Bandpass 0.3–35 Hz
    eeg.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    return eeg.get_data()[0] # return 1D array of the EEG channel

def preprocess_ecg(raw, method="neurokit", lowcut=0.5, highcut=45):
    sf = raw.info["sfreq"] # sampling frequency
    ecg = raw.get_data(picks="ECG1-ECG2")[0]

    # Clean using NeuroKit2's ecg_clean function 
    # Bandpass 0.5–45 Hz
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=sf, method=method, lowcut=lowcut, highcut=highcut)

    return ecg_clean # return 1D array of the cleaned ECG signal

def compute_psd(eeg_filtered, sf, fmin=0.5, fmax=40,  n_per_seg=2048, n_fft=4096,  n_overlap=1024):

    # n_per_seg: Length of each segment for Welch's method 
    # window length = 2048 / 512 = 4.0 seconds

    # n_fft: Length of the FFT used 
    # FFT length = 4096 / 512 = 8.0 seconds worth of zero-padding

    # Δf = sf / n_fft = 512 / 4096 = 0.125 Hz
    # 0.5 Hz = 1 cycle every 2 seconds
    # 4 seconds -> 2 cycles

    # n_overlap: Number of points to overlap between segments 
    # overlap = 1024 / 512 = 2.0 seconds

    # psd shape (n_channels, n_freqs)
    # freqs shape (n_freqs,)

    psd, freqs = psd_array_welch(eeg_filtered[np.newaxis, :], sfreq=sf, fmin=fmin, fmax=fmax, n_per_seg=n_per_seg,n_fft=n_fft, n_overlap=n_overlap, verbose=False)
    # psd is in V^2 / Hz
    # V^2 / Hz -> µV^2 / Hz by multiplying by 1e12
    psd = psd * 1e12
    
    return psd, freqs

def summarize_psd(raw_data, filtered_data):
    # mean and std of PSD across frequencies for each channel - RAW
    raw_mean = raw_data.mean()
    raw_std = raw_data.std()

    # mean and std of PSD across frequencies for each channel - CLEANED
    filt_mean = filtered_data.mean()
    filt_std = filtered_data.std()

    print(f"RAW: {raw_mean:.2e} ± {raw_std:.2e} µV²/Hz")
    print(f"FILTERED: {filt_mean:.2e} ± {filt_std:.2e} µV²/Hz")
    # if std is much smaller than mean -> more stable 
    # if std is large compared to mean ->  more variability 

def eeg_bandpower(eeg_filtered, stage_epochs_dict, sf, min_epoch_s=2.0):
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 40)}
    rows = []
    n = len(eeg_filtered)
    min_samples = int(min_epoch_s * sf)


    for stage_name, epochs_list in stage_epochs_dict.items():
        for i, (t0, t1) in enumerate(epochs_list):
            a, b = int(t0 * sf), int(t1 * sf)
            a, b = max(0, a), min(n, b)  # ensure indices are within bounds

            if b - a < min_samples:
                continue  # skip epochs that are too short for reliable PSD estimation

            X = eeg_filtered[a:b]

            if np.all(X == 0):
                continue  # skip epochs with no signal (e.g., due to artifacts or missing data)
            
            # Cálculo do PSD
            psd, freqs = psd_array_welch(X[np.newaxis, :], sfreq=sf, fmin=0.5, fmax=40, verbose=False)
            
            row = {"stage": stage_name, "epoch": i, "start_s": float(t0)}
            bp = {}
            for name, (f1, f2) in bands.items():
                idx = (freqs >= f1) & (freqs <= f2)
                val = float(np.trapezoid(psd[:, idx], freqs[idx], axis=1).mean())
                bp[name] = val
                row[f"{name}_power"] = val
            
            total_power = sum(bp.values())
            row["delta_relative"] = bp["delta"] / total_power if total_power > 0 else np.nan
            rows.append(row) 
            
    return pd.DataFrame(rows)

def plot_psd_comparison(eeg_filtered, sf):

    psd, freqs = psd_array_welch(eeg_filtered[np.newaxis, :], sfreq=sf, fmin=0.5, fmax=40, verbose=False)
    psd = psd[0]

    bands = {"Delta": (0.5, 4, "blue"), "Theta": (4, 8, "green"), "Alpha": (8, 12, "red"), "Beta": (12, 30, "orange"), "Gamma": (30, 40, "purple")}

    plt.figure(figsize=(12, 5))

    plt.semilogy(freqs, psd, color='black', linewidth=2, label='Total PSD')

    for name, (fmin, fmax, color) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        plt.fill_between(freqs[idx], psd[idx], color=color, alpha=0.3, label=name)

    plt.title("EEG Power Spectral Density por Bandas (Sinal Total)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (µV²/Hz)")
    plt.xlim(0.5, 40)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.show()



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
        x = ecg[a:b] * 1000 # convert to mV for better numerical stability in HRV calculations
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
    peaks, info = nk.ecg_peaks(ecg, sampling_rate=sf)

    ecg_rate = nk.ecg_rate(peaks, sampling_rate=sf, desired_length=len(ecg))

    resp = nk.ecg_rsp(ecg_rate, sampling_rate=sf)

    # 0.1–0.4 Hz (6–24 breaths per minute)
    # Remove very slow baseline drift (<0.05 Hz)
    # Remove high-frequency noise (>0.7 Hz)
    resp = nk.signal_filter(resp, sampling_rate=sf, lowcut=0.05, highcut=0.7)
    
    # resp = (resp - np.mean(resp)) / np.std(resp)

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

        x_ecg = ecg[a:b] * 1000 # convert to mV for better numerical stability in HRV calculations
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
             HFC = np.trapezoid(CPC[hf_mask], f[hf_mask]) * 1e6  # convert to µV²
             LFC = np.trapezoid(CPC[lf_mask], f[lf_mask]) * 1e6  # convert to µV²
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


def hep_metric(eeg_filtered, epoch, sf, window_s=1.0, min_rr_s=0.88, amplitude_threshold_uv=100.0):
    # epoch = ecg_R.iloc[1]
    tmin, tmax = -0.2, 0.6

    t0 = epoch["t0_s"]
    t1 = epoch["t1_s"]
    # Convert epoch start and end times from seconds to sample indices
    a, b = int(t0 * sf), int(t1 * sf) 
    window_samples = int(window_s * sf)

    # get rpeaks for each epoch and store in a DataFrame
    rpeaks = epoch["rpeaks"]
    if rpeaks is None or len(rpeaks) < 2:
            return None
    
    # extract full EEG data as numpy array
    rpeaks_global = np.array(rpeaks) + a  # relative → global
    # = [100+7200, 450+7200, 800+7200]   (se sf=120 e t0=60s → a=7200)
    # = [7300, 7650, 8000]  ← posição real no sinal EEG completo

    hep_values_scalar = [] # list of mean HEP amplitude values for each 1s window
    

    rr_intervals = np.diff(rpeaks_global)  # em amostras
    
    min_rr_samples = int(min_rr_s * sf)
    # define a minimum RR interval in samples to avoid overlapping heartbeats
    
    amplitude_threshold = amplitude_threshold_uv * 1e-6  # µV → V
    # define an amplitude threshold to exclude artifacts

    # iterate over 1-second windows within the epoch

    n_windows = int((b - a) / window_samples)
    # b-a = 15360 samples (30s epoch) / 512 Hz = 30s → 30 windows of 1s each 
    # window_samples = 512 samples (1s) → 30 windows of 512 samples each
    # n_windows = total_duration / 1 second = 15360 / 512 = 30 windows

    for w in range(n_windows):
        start = a + w * window_samples
        end   = start + window_samples
        # for w=0: start=7200, end=7712
        # for w=1: start=7712, end=8224

        # blocks of 1s that do not overlap: [a, a+window_samples), [a+window_samples, a+2*window_samples), ...
        #  [7200, 7328), [7328, 7456), [7456, 7584)

        # select only Rpeaks that fall in that window 
        selected_rpeaks = rpeaks_global[(rpeaks_global >= start) & (rpeaks_global < end)]
        
        # If no R-peaks are found in the window, we can skip or assign NaN
        if len(selected_rpeaks) == 0:
            hep_values_scalar.append(np.nan)
            continue
        
        # create hep segments for each Rpeak from -200 to +500 ms around the Rpeak
        hep_segments = [] # shape (1, 700)
    
        # r is relative to the epoch start
        # r is the global sample index 
        # is the position of the R peak in the full EEG signal (not just the epoch)
        for r in selected_rpeaks:
            # Extract EEG segment around the R-peak

            seg_start = int(r + tmin * sf) # r - 200ms
            seg_stop  = int(r + tmax * sf) # r + 500ms

            # r is the center of the extracted segment
            # it's the exact sample in eeg_filtered where the heartbeat occurred 
            # seg_start and seg_stop define the time window around the R-peak

            if seg_start < 0 or seg_stop > len(eeg_filtered):
                continue
            # if the rpeak is out of the bounds of the whole EEG signal, skip it

            r_idx_global = np.where(rpeaks_global == r)[0] 
            # find the index of the R-peak in the global list of R-peaks for the epoch
            if len(r_idx_global) > 0:
                r_idx_global = r_idx_global[0] 
                if r_idx_global < len(rr_intervals):
                    if rr_intervals[r_idx_global] < min_rr_samples: 
                        # if the RR interval after this R-peak is too short, skip it
                        continue  

            # extract the segment of the EEG of -200 ms to +500 ms
            segment = eeg_filtered[seg_start:seg_stop][None, :] # shape (1, segment_samples)
            
            # amplitude check to exclude artifacts
            if np.max(np.abs(segment)) > amplitude_threshold:
                continue

            # remove linear trends that could bias the average
            segment = detrend(segment, axis=1)  # axis=1 = along time
            
            # append the segments 
            # (1, 700) → (n_beats, 700) 
            hep_segments.append(segment)

        # if all rpeaks on the 1s window are discarded then discard all window 
        if len(hep_segments) == 0:
            hep_values_scalar.append(np.nan)
            continue

        hep_epochs = np.array(hep_segments)  
        # shape (n_beats, 1, segment_samples) → (n_beats, segment_samples)

        # Baseline corrections (-200ms a 0ms, modal value of paper)
        r_idx    = int(abs(tmin) * sf)           # position of R-peak no segmento
        bl_start = 0                             # -200ms (início do segmento)
        bl_end   = r_idx                         # 0ms (R-peak)
        # baseline shape (n_beats, 1, 1) → (n_beats, 1, segment_samples)
        baseline = hep_epochs[:, :, bl_start:bl_end].mean(axis=2, keepdims=True) 
        # baseline is the mean amplitude in the -200 to 0 ms window for each beat, used to correct for slow drifts in the EEG signal
        hep_epochs = hep_epochs - baseline


        # (3, 1, 700) → (1, 700)
        # (n_beats, 1, n_samples) → (1, n_samples) → (n_samples,)
        # average of the EEG amplitude across beats to get the HEP for that 1-second window
        hep_avg = hep_epochs.mean(axis=0).squeeze()  # shape: (n_samples,)


        # mean amplitude of the HEP in the -200 to +600 ms window 
        hep_values_scalar.append(float(hep_avg.mean()) * 1e6)

    return { "scalar": hep_values_scalar, } # list of floats (µV)


def delta_power_1s(eeg_filtered, epoch, sf, window_frequency=1.0):
    delta_vals  = []
    window_samples = int(window_frequency * sf)
    t0, t1 = epoch
    a, b = int(t0 * sf), int(t1 * sf)

    if a < 0 or b > len(eeg_filtered) or a >= b:
        return None

    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 40)}

    epoch_signal = eeg_filtered[a:b]

    for start in range(0, (b - a) - window_samples + 1, window_samples):
        end = start + window_samples
        segment = epoch_signal[start:end]

        if segment.size < window_samples:
            continue

        try:
            # Compute PSD for the 1-second segment
            # expects (n_channels, n_samples)
            #epoch_signal is 1D array (nsamples)

            # relative delta power = delta power / total power
            psd, freqs = psd_array_welch(segment[None, :], sfreq=sf, fmin=0.5, fmax=40.0, verbose=False)
            psd_seg = psd[0]

            powers = {}
            for name, (f1, f2) in bands.items():
                idx = (freqs >= f1) & (freqs <= f2)

                powers[name] = float(np.trapezoid(psd_seg[idx], freqs[idx])) if np.any(idx) else 0.0
            
            total_power = sum(powers.values())
            rel_delta = powers["delta"] / total_power if total_power > 0 else 0.0
            
            delta_vals.append(rel_delta)
        except Exception:
            continue
    return delta_vals

def HEP_Delta_plot_one_epoch(hep_values, delta_vals, epoch):
    hep_values = hep_values["scalar"] 
    t = np.arange(len(delta_vals))

    fig, ax1 = plt.subplots(figsize=(10,4))

    ax1.plot(t, hep_values, color="blue", marker="o")
    ax1.set_xlabel("Time within epoch (s)")
    ax1.set_ylabel("HEP Mean Amplitude (µV)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(t, delta_vals, color="red", marker="o")
    ax2.set_ylabel("Delta Power (µV²)", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    plt.title(f"HEP and Delta — epoch {epoch['t0_s']:.0f}s to {epoch['t1_s']:.0f}s")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Plot HEP and Delta Power across epochs for a given stage

def HEP_Delta_HR_plot_new(eeg_filtered, ecg_R, epochs, sf, epoch_range):
    # [epoch1 (30s), epoch2 (30s), epoch3 (30s), ...]
    # for each epoch, calculate HEP and Delta Power for 1s windows within the epoch
    hep_values = []
    delta_vals = []
    hr_values = []

    epoch_range = list(epoch_range)

    # iterating over the specified range of epochs
    for i in epoch_range:
        epoch = epochs[i] # (t0, t1) OR dict
        ecg_epoch = ecg_R.iloc[i] if i < len(ecg_R) else None # rpeaks + HR

        hep = hep_metric(eeg_filtered, ecg_epoch, sf) if ecg_epoch is not None else None

        # Extract the time range for the delta power calculation
        delta_epoch = (epoch["t0_s"], epoch["t1_s"]) if isinstance(epoch, (pd.Series, dict)) else epoch

        delta = delta_power_1s(eeg_filtered, delta_epoch, sf)
        hr = ecg_epoch["hr_mean_bpm"] if ecg_epoch is not None and "hr_mean_bpm" in ecg_epoch else np.nan

        # Append the results for this epoch to the lists
        hep_values.extend(hep["scalar"] if hep is not None else [])
        delta_vals.extend(delta if delta is not None else [])
        hr_values.extend([hr] * len(delta if delta is not None else [])) # Repeats HR 30 times → makes it 1 Hz

    n = min(len(hep_values), len(delta_vals), len(hr_values))
    hep_values = hep_values[:n]
    delta_vals = delta_vals[:n]
    hr_values = hr_values[:n]

    t = np.arange(n)
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t, hep_values, color="blue")
    ax1.set_xlabel("Time (1s windows)")
    ax1.set_ylabel("HEP (µV)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(t, delta_vals, color="red")
    ax2.set_ylabel("Delta Relative Power", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.08))
    ax3.plot(t, hr_values, color="green", linestyle="--")
    ax3.set_ylabel("HR (bpm)", color="green")
    ax3.tick_params(axis='y', labelcolor="green")

    plt.title(f"HEP + Delta + HR | Epochs {epoch_range[0]} to {epoch_range[-1]}")
    plt.tight_layout()
    plt.show()

# Plot HEP and EEG signal for one epoch 
def HEP_EEG_plot_one_epoch(hep_values, eeg_filtered, epoch, sf):
    hep_values = hep_values["scalar"] 
    t0 = epoch["t0_s"]
    t1 = epoch["t1_s"]

    a, b = int(t0 * sf), int(t1 * sf)
    eeg_segment = eeg_filtered[a:b]

    t_eeg = np.linspace(0, 30, len(eeg_segment))
    t_hep = np.linspace(0, 30, len(hep_values))

    fig, ax1 = plt.subplots(figsize=(10,4))

    ax1.plot(t_eeg, eeg_segment * 1e6, color="red", alpha=0.5)  # EEG in µV
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("EEG (µV)", color="red")
    ax1.tick_params(axis='y', labelcolor="red")

    ax2 = ax1.twinx()
    ax2.plot(t_hep, hep_values, color="blue", marker="o")
    ax2.set_ylabel("HEP Mean Amplitude (µV)", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")

    plt.title(f"HEP and EEG — epoch {epoch['t0_s']:.0f}s to {epoch['t1_s']:.0f}s")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Correlation between HEP and Delta Power across epochs
# using pearson correlation coefficient
def plot_hep_delta_correlation(hep_values, delta_vals):
    hep   = np.array(hep_values, dtype=float)
    delta = np.array(delta_vals, dtype=float)

    # remove nan pairs
    mask  = ~np.isnan(hep) & ~np.isnan(delta)
    hep_clean   = hep[mask]
    delta_clean = delta[mask]

    if len(hep_clean) < 3:
        print("Not enough valid values to correlate")
        return None

    r_pearson,  p_pearson  = pearsonr(hep_clean, delta_clean)
    r_spearman, p_spearman = spearmanr(hep_clean, delta_clean)

    print(f"Pearson:  r={r_pearson:.3f},  p={p_pearson:.3f}")
    print(f"Spearman: r={r_spearman:.3f}, p={p_spearman:.3f}")

    return {
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "spearman_r": r_spearman,
        "spearman_p": p_spearman,
    }