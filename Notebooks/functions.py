from multiprocessing.resource_sharer import stop
from pathlib import Path
import pandas as pd
import mne
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pyedflib
from mne.time_frequency import psd_array_welch
import mne
from IPython.display import display
from scipy.signal import welch
from scipy.signal import coherence, csd


def compute_stage_epochs(df, stage_label):

    stage_df = df[df["Sleep Stage"] == stage_label].copy()

    if stage_df.empty:
        return []

    epochs = list(zip(
        stage_df["onset_s"].astype(float),
        stage_df["onset_s"] + stage_df["Duration[s]"]
    ))

    return epochs

def add_epoch_onsets(df, epoch_len):
    df = df.copy()
    df["onset_s"] = np.arange(len(df)) * epoch_len
    return df

def pick_core_channels(raw):
    core_chs = ["F3-C3", "F4-C4", "ECG1-ECG2"]
    return raw.copy().pick(core_chs)

def preprocess_eeg(raw, notch_freqs=(50, 100), l_freq=0.3, h_freq=35.0):
    eeg = raw.copy()
    if notch_freqs is not None:
        eeg.notch_filter(notch_freqs,
                         fir_design='firwin',
                         verbose=False)

    eeg.filter(l_freq=l_freq,
               h_freq=h_freq,
               verbose=False)

    return eeg

def preprocess_ecg(raw, method="neurokit", lowcut=0.5, highcut=45):

    ecg_channel = "ECG1-ECG2"
    sf = raw.info["sfreq"]

    ecg = raw.copy().pick([ecg_channel]).load_data().get_data()[0]

    ecg_clean = nk.ecg_clean(ecg, sampling_rate=sf, method=method, lowcut=lowcut, highcut=highcut)

    return ecg_clean

def compute_psd(raw, fmin=0.5, fmax=40,  n_per_seg=2048, n_fft=4096,  n_overlap=1024):

    X = raw.get_data()
    sf = raw.info["sfreq"]

    psd, freqs = psd_array_welch(X, sfreq=sf, fmin=fmin, fmax=fmax, n_per_seg=n_per_seg,n_fft=n_fft, n_overlap=n_overlap, verbose=False)

    return psd, freqs

def summarize_psd(raw_data, filtered_data, ch_names):
    for i in (range(len(ch_names))):
        raw_mean = raw_data[i].mean()
        raw_std = raw_data[i].std()

        filt_mean = filtered_data[i].mean()
        filt_std = filtered_data[i].std()

        print(f"{ch_names[i]}: {raw_mean:.2e} ± {raw_std:.2e} µV²/Hz")
        print(f"{ch_names[i]}: {filt_mean:.2e} ± {filt_std:.2e} µV²/Hz")

def plot_psd_comparison(freqs, psd_raw, psd_filt, ch_names):

    m_raw, s_raw = psd_raw.mean(axis=0), psd_raw.std(axis=0)
    m_flt, s_flt = psd_filt.mean(axis=0), psd_filt.std(axis=0)

    plt.figure(figsize=(10,5))

    for i, ch in enumerate(ch_names):
        plt.semilogy(freqs, psd_raw[i]*1e12,
                     linestyle="--", linewidth=1,
                     alpha=0.7, label=f"{ch} RAW")

        plt.semilogy(freqs, psd_filt[i]*1e12,
                     linewidth=1,
                     alpha=0.7, label=f"{ch} FILTERED")

    plt.fill_between(freqs, (m_raw-s_raw)*1e12,
                     (m_raw+s_raw)*1e12, alpha=0.15)

    plt.fill_between(freqs, (m_flt-s_flt)*1e12,
                     (m_flt+s_flt)*1e12, alpha=0.15)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (µV²/Hz)")
    plt.title("EEG PSD Comparison")
    plt.xlim(0.5, 40)
    plt.legend()
    plt.show()

def eeg_bandpower_per_epoch(eeg_filtered, epochs):
    bands = { "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta":  (12, 30), "gamma": (30, 40) }
    sf = eeg_filtered.info["sfreq"]
    rows = []

    for i, (t0, t1) in enumerate(epochs):
        a, b = int(t0*sf), int(t1*sf)
        X = eeg_filtered.get_data(start=a, stop=b) 
        psd, freqs = psd_array_welch(X, sfreq=sf, fmin=0.5, fmax=40, verbose=False)

        row = {"epoch": i, "start_s": float(t0)}

        for name, (f1, f2) in bands.items():
            idx = (freqs >= f1) & (freqs <= f2)
            bp_ch = np.trapezoid(psd[:, idx], freqs[idx], axis=1)  
            row[f"{name}_power"] = float(bp_ch.mean())      
        rows.append(row)

    return pd.DataFrame(rows)

def eeg_bandpower_all_stages(raw, stage_epochs_dict):
    dfs = []
    for stage, epochs in stage_epochs_dict.items():
        if len(epochs) == 0:
            continue
        tmp = eeg_bandpower_per_epoch(raw, epochs)
        tmp["stage"] = stage
        dfs.append(tmp)

    return pd.concat(dfs, ignore_index=True)

def extract_ecg_per_epoch(ecg_sig_total, ecg_clean_total, epochs, sf, detect_peaks=True):
    rows = []
    for i, (t0, t1) in enumerate(epochs):
        a, b = int(t0 * sf), int(t1 * sf)

        x_raw = ecg_sig_total[a:b]
        x_clean = ecg_clean_total[a:b]

        try:
            if detect_peaks:
                _, info = nk.ecg_peaks(x_clean, sampling_rate=sf)
                rpeaks = info["ECG_R_Peaks"]
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
            signals, info = nk.ecg_process(x, sampling_rate=sf)
            # mean HR from ECG_Rate (bpm)
            hr_mean = float(np.nanmean(signals["ECG_Rate"]))

            # HRV metrics (time domain) from detected R-peaks
            hrv = nk.hrv_time(info, sampling_rate=sf).iloc[0].to_dict()

            rows.append({
                "epoch": i,
                "start_s": float(t0),
                "hr_mean_bpm": hr_mean,
                "rmssd_ms": float(hrv.get("HRV_RMSSD", np.nan)),
                "sdnn_ms": float(hrv.get("HRV_SDNN", np.nan)),
                "pnn50_pct": float(hrv.get("HRV_pNN50", np.nan)),
                "n_beats": int(len(info["ECG_R_Peaks"])) if "ECG_R_Peaks" in info else np.nan,
                "ok": True
            })
        except Exception as e:
            rows.append({"epoch": i, "start_s": float(t0), "ok": False, "error": str(e)})

    return pd.DataFrame(rows)

def extract_resp_from_ecg(ecg, sf, method="neurokit"):
    if method == "neurokit":
        resp = nk.ecg_rsp(ecg, sampling_rate=sf)
        resp = nk.signal_filter(resp, sampling_rate=sf, lowcut=0.05, highcut=0.7)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return resp

def hpc_metric(ecg, resp, epochs, sf, window_epochs):
    rows = []
    fs_cpc = 4  
    for i in range(0, len(epochs), window_epochs):

        group = epochs[i:i+window_epochs]

        if len(group) < window_epochs:
            continue  

        t0 = group[0][0]
        t1 = group[-1][1]

        a, b = int(t0*sf), int(t1*sf)
        x_ecg = ecg[a:b]
        x_resp = resp[a:b]

        try:
             _, info = nk.ecg_peaks(x_ecg, sampling_rate=sf)
             rpeaks = info["ECG_R_Peaks"]

             rr = np.diff(rpeaks) / sf  # RR intervals in seconds
             t_rr = np.cumsum(rr)  # Time points of RR intervals

             t_grid = np.arange(t_rr[0], t_rr[-1], 1/fs_cpc)
             rr_resampled = np.interp(t_grid, t_rr, rr)  # Resample RR intervals to common grid
            

             t_resp_grid = np.arange(len(x_resp))/sf
             x_resp_resampled = np.interp(t_grid, t_resp_grid, x_resp)  # Resample Resp to common grid

             nps = min(64, len(rr_resampled))

             f, Cxy = csd(rr_resampled, x_resp_resampled, fs=fs_cpc, nperseg=nps)
             _, Coh = coherence(rr_resampled, x_resp_resampled, fs=fs_cpc, nperseg=nps)

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

    df["sleep_stability"] = np.where(df["HFC"] > df["LFC"], "stable", "unstable")

    df.loc[(df["HFC"] == 0) & (df["LFC"] == 0), "sleep_stability"] = "undefined"
    df.loc[(df["HFC"].isna()) | (df["LFC"].isna()), "sleep_stability"] = "undefined"

    return df

def hep_metric(eeg, ecg, epochs, sf):
    tmin = -0.2
    tmax = 0.6
    rows = []
    ecg_epochs = extract_ecg_per_epoch(ecg, ecg, epochs, sf, detect_peaks=True) 
    eeg_data = eeg.get_data()

    for i, (t0, t1) in enumerate(epochs):
        hep_segments = []
        epoch_data = ecg_epochs.iloc[i]
        if epoch_data.empty or not epoch_data["ok"]:
            continue         
        
        if epoch_data["rpeaks"] is None or len(epoch_data["rpeaks"]) < 2:
            continue
        
        for r in epoch_data["rpeaks"]:
            r_global = r + t0 * sf  # Convert local R-peak to global time index
            start = int(r_global + tmin * sf)
            stop  = int(r_global + tmax * sf)
            x_ecg = ecg[start:stop]

            segment = eeg_data[:, start:stop]
            hep_segments.append(segment)

        if len(hep_segments) == 0:
            continue

        hep_epochs = np.array(hep_segments)  

        hep_avg = hep_epochs.mean(axis=0)

        hep_mean_amp = hep_avg.mean()
 
        rows.append({"epoch": i, "hep_mean_amp": float(hep_mean_amp), "n_beats_used": len(hep_epochs), "ok": True})

    return pd.DataFrame(rows)

