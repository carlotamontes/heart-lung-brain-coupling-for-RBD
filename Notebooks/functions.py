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
from scipy.signal import coherence


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

            rows.append({"epoch": i, "t0_s": float(t0), "t1_s": float(t1), "dur_s": float(t1 - t0), "raw_seg": x_raw, "clean_seg": x_clean, "rpeaks": rpeaks, "n_peaks": n_peaks, "ok": True})

        except Exception as e:
            rows.append({"epoch": i,  "t0_s": float(t0), "t1_s": float(t1), "dur_s": float(t1 - t0), "raw_seg": x_raw, "clean_seg": x_clean, "rpeaks": None, "n_peaks": 0,  "ok": False, "error": str(e)})

    return pd.DataFrame(rows)
