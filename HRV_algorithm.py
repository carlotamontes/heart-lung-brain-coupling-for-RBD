from pathlib import Path
import pandas as pd
import mne
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pyedflib
from mne.time_frequency import psd_array_welch
from IPython.display import display
from scipy.signal import welch
from scipy.signal import coherence

data = Path(r"C:\Users\carlo\OneDrive - Universidade de Lisboa\Documents\GitHub\heart-lung-brain-coupling-for-RBD\Data\cap-sleep-database-1.0.0\rbd1.edf")
raw = mne.io.read_raw_edf(data, preload=False, verbose=False)
raw.load_data()
raw.info["sfreq"], len(raw.ch_names), raw.ch_names[:10]

eeg_chs  = ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2", "C4-A1", "F8-T4", "F7-T3"]
ecg_chs  = ["ECG1-ECG2"]
resp_chs = ["TORACE", "ADDOME"]
eog_chs  = ["ROC-LOC"]

raw.get_channel_types(picks=eeg_chs[:5] + ecg_chs + resp_chs)
#print("Sampling frequency:", raw.info["sfreq"])
#print("Duration (s):", raw.n_times / raw.info["sfreq"])

txt_path = r"C:\Users\carlo\OneDrive - Universidade de Lisboa\Documents\GitHub\heart-lung-brain-coupling-for-RBD\Data\cap-sleep-database-1.0.0\rbd1.txt"
df = pd.read_csv(txt_path, sep='\t', header=None, names=["Sleep Stage", "Position", "Time [hh:mm:ss]", "Event", "Duration[s]", "Location"], skiprows=22)

epoch_len = 30.0
df["onset_s"] = np.arange(len(df)) * epoch_len

# WAKE EPOCHS
wake = df[df["Sleep Stage"] == "W"]
wake = df[df["Sleep Stage"] == "W"].copy()
wake_epochs = list(zip(wake["onset_s"].to_numpy(), (wake["onset_s"] + wake["Duration[s]"]).to_numpy()))
t0_wake, t1_wake = wake_epochs[0]
dur_wake = float(t1_wake - t0_wake)
t0_wake, t1_wake, dur_wake

# REM EPOCHS
rem = df[df["Sleep Stage"] == "R"]
rem = df[df["Sleep Stage"] == "R"].copy()
rem_epochs = list(zip(rem["onset_s"].to_numpy(), (rem["onset_s"] + rem["Duration[s]"]).to_numpy()))
t0_rem, t1_rem = rem_epochs[0]
dur_rem = float(t1_rem - t0_rem)
t0_rem, t1_rem, dur_rem

#S1 EPOCHS
S1 = df[df["Sleep Stage"] == "S1"]
S1 = df[df["Sleep Stage"] == "S1"].copy()
S1_epochs = list(zip(S1["onset_s"].to_numpy(), (S1["onset_s"] + S1["Duration[s]"]).to_numpy()))
t0_S1, t1_S1 = S1_epochs[0]
dur_S1 = float(t1_S1 - t0_S1)
t0_S1, t1_S1, dur_S1

# S2 EPOCHS
S2 = df[df["Sleep Stage"] == "S2"]
S2 = df[df["Sleep Stage"] == "S2"].copy()
S2_epochs = list(zip(S2["onset_s"].to_numpy(), (S2["onset_s"] + S2["Duration[s]"]).to_numpy()))
t0_S2, t1_S2 = S2_epochs[0]
dur_S2 = float(t1_S2 - t0_S2)
t0_S2, t1_S2, dur_S2

# S3 EPOCHS
S3 = df[df["Sleep Stage"] == "S3"]
S3 = df[df["Sleep Stage"] == "S3"].copy()
S3_epochs = list(zip(S3["onset_s"].to_numpy(), (S3["onset_s"] + S3["Duration[s]"]).to_numpy()))
t0_S3, t1_S3 = S3_epochs[0]
dur_S3 = float(t1_S3 - t0_S3)
t0_S3, t1_S3, dur_S3

# S4 EPOCHS
S4 = df[df["Sleep Stage"] == "S4"]
S4 = df[df["Sleep Stage"] == "S4"].copy()
S4_epochs = list(zip(S4["onset_s"].to_numpy(), (S4["onset_s"] + S4["Duration[s]"]).to_numpy()))
t0_S4, t1_S4 = S4_epochs[0]
dur_S4 = float(t1_S4 - t0_S4)
t0_S4, t1_S4, dur_S4

sf = raw.info["sfreq"]
ecg_1d = raw.copy().pick(ecg_chs).load_data().get_data()[0]
a, b = int(t0_rem*sf), int(t1_rem*sf)
ecg_seg = ecg_1d[a:b]


# Filter ECG 
ecg_data = raw.copy().pick(ecg_chs).load_data().get_data()[0]
ecg_filtered = nk.ecg_clean(ecg_data, sampling_rate=raw.info["sfreq"], method="neurokit", lowcut=0.5, highcut=45)
# Create a new Raw object for plotting
info = mne.create_info(ch_names=ecg_chs, sfreq=raw.info["sfreq"], ch_types=['ecg'])
ecg_filtered_raw = mne.io.RawArray(ecg_filtered.reshape(1, -1), info)

ecg_ch = ecg_chs[0]  

ecg_sig_total = raw.copy().pick([ecg_ch]).load_data().get_data()[0]
ecg_clean_total = nk.ecg_clean(ecg_sig_total, sampling_rate=sf, method="neurokit")

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

resp = raw.copy().pick(["TORACE"]).load_data().get_data()[0]

def hpc_metric(ecg, resp, epochs, sf):
    rows = []
    fs_cpc = 4  # Resample ECG and Resp to a common frequency for coherence analysis
    for i, (t0, t1) in enumerate(epochs):
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
             f, Cxy = coherence(rr_resampled, x_resp_resampled, fs=fs_cpc, nperseg=nps)

             lf_mask = (f >= 0.01) & (f <= 0.15)
             hf_mask = (f >= 0.15) & (f <= 0.40)
             HFC = np.trapezoid(Cxy[hf_mask], f[hf_mask])
             LFC = np.trapezoid(Cxy[lf_mask], f[lf_mask])
             LFC_HFC_ratio = LFC / HFC if HFC > 0 else np.nan
             rows.append({"epoch": i, "HFC": HFC, "LFC": LFC, "LFC/HFC": LFC_HFC_ratio, "ok": True})
        except Exception as e:
             rows.append({"epoch": i, "start_s": float(t0), "ok": False, "error": str(e)})
             continue   

    return pd.DataFrame(rows)