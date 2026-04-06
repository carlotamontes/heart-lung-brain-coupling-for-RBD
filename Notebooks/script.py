import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from Notebooks.functions import (
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

STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM"]


def _safe_corr(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = ~np.isnan(xa) & ~np.isnan(ya)
    xa = xa[mask]
    ya = ya[mask]

    if xa.size < 3 or np.all(xa == xa[0]) or np.all(ya == ya[0]):
        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }

    pr, pp = pearsonr(xa, ya)
    sr, sp = spearmanr(xa, ya)
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def _find_input_file(base_dir: Path, patient_id: str, suffixes: Tuple[str, ...]) -> Path:
    candidates = []
    for suffix in suffixes:
        candidates.extend(base_dir.glob(f"{patient_id}*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No file found for patient '{patient_id}' in {base_dir}")
    return sorted(candidates)[0]


def _load_hypnogram(path: Path, epoch_seconds: int = 30) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    hyp = pd.read_csv(path, sep=sep)

    cols = {c.lower().strip(): c for c in hyp.columns}
    stage_col = cols.get("sleep stage") or cols.get("sleep_stage")
    if stage_col is None:
        raise ValueError(f"Hypnogram missing sleep stage column: {path}")

    hyp = hyp.rename(columns={stage_col: "Sleep Stage"})
    hyp["Sleep Stage"] = hyp["Sleep Stage"].replace(STAGE_LABEL_MAP)

    duration_col = cols.get("duration[s]") or cols.get("duration")
    if duration_col and duration_col != "Duration[s]":
        hyp = hyp.rename(columns={duration_col: "Duration[s]"})
    if "Duration[s]" not in hyp.columns:
        hyp["Duration[s]"] = float(epoch_seconds)

    onset_col = cols.get("onset_s") or cols.get("onset")
    if onset_col:
        hyp = hyp.rename(columns={onset_col: "onset_s"})
    if "onset_s" not in hyp.columns:
        hyp = add_epoch_onsets(hyp, epoch_len=epoch_seconds)

    hyp = hyp[hyp["Sleep Stage"].isin(STAGE_ORDER)].reset_index(drop=True)
    return hyp[["Sleep Stage", "Duration[s]", "onset_s"]]


def _stage_epochs_dict(hypnogram_df: pd.DataFrame) -> Dict[str, List[Tuple[float, float]]]:
    out = {}
    for stage in STAGE_ORDER:
        out[stage] = compute_stage_epochs(hypnogram_df, stage)
    return out


def _epoch_rows(hypnogram_df: pd.DataFrame) -> List[Tuple[float, float, str]]:
    rows = []
    for _, row in hypnogram_df.iterrows():
        t0 = float(row["onset_s"])
        t1 = t0 + float(row["Duration[s]"])
        rows.append((t0, t1, row["Sleep Stage"]))
    return rows


def _expand_cpc_to_epochs(cpc_df: pd.DataFrame, n_epochs: int, window_epochs: int) -> pd.DataFrame:
    out = pd.DataFrame({"epoch": np.arange(n_epochs), "HFC": np.nan, "LFC": np.nan, "LFC/HFC": np.nan})
    if cpc_df.empty:
        return out

    for _, row in cpc_df.iterrows():
        start = int(row["epoch"])
        end = min(start + window_epochs, n_epochs)
        out.loc[start:end - 1, ["HFC", "LFC", "LFC/HFC"]] = [row.get("HFC", np.nan), row.get("LFC", np.nan), row.get("LFC/HFC", np.nan)]
    return out


def _rolling_corr(values_x: List[float], values_y: List[float], window_epochs: int) -> List[Dict[str, float]]:
    output = []
    for i in range(len(values_x)):
        a = max(0, i - window_epochs + 1)
        corr = _safe_corr(values_x[a:i + 1], values_y[a:i + 1])
        output.append(corr)
    return output


def process_patient(
    patient_id: str,
    edf_path: Path,
    hypnogram_path: Path,
    output_dir: Path,
    epoch_seconds: int = 30,
    cpc_window_epochs: int = 5,
    corr30_window_epochs: int = 30,
) -> None:
    group = PATIENT_GROUP.get(patient_id, "Unknown")

    logging.info("[%s] Loading EDF: %s", patient_id, edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    logging.info("[%s] Loading hypnogram: %s", patient_id, hypnogram_path)
    hypnogram_df = _load_hypnogram(hypnogram_path, epoch_seconds=epoch_seconds)
    epoch_timeline = _epoch_rows(hypnogram_df)
    stage_epochs = _stage_epochs_dict(hypnogram_df)

    sf = float(raw.info["sfreq"])

    logging.info("[%s] Preprocessing EEG/ECG", patient_id)
    eeg_filtered = preprocess_eeg(raw)
    ecg_clean = preprocess_ecg(raw)
    ecg_raw = raw.get_data(picks="ECG1-ECG2")[0]

    epochs_only = [(t0, t1) for t0, t1, _ in epoch_timeline]

    logging.info("[%s] ECG epoch features", patient_id)
    ecg_epoch_df = extract_ecg_per_epoch(ecg_raw, ecg_clean, epochs_only, sf, detect_peaks=True)
    hrv_df = hrv_per_epoch(ecg_clean, epochs_only, sf)

    logging.info("[%s] CPC features", patient_id)
    resp = extract_resp_from_ecg(ecg_clean, sf)
    cpc_df = hpc_metric(ecg_clean, resp, epochs_only, sf, window_epochs=cpc_window_epochs)
    cpc_epoch_df = _expand_cpc_to_epochs(cpc_df, n_epochs=len(epochs_only), window_epochs=cpc_window_epochs)

    logging.info("[%s] EEG bandpower", patient_id)
    bandpower_df = eeg_bandpower(eeg_filtered, stage_epochs, sf)

    # Build per-epoch baseline table
    epoch_df = pd.DataFrame({
        "epoch": np.arange(len(epoch_timeline)),
        "patient_id": patient_id,
        "group": group,
        "epoch_start_time": [x[0] for x in epoch_timeline],
        "epoch_end_time": [x[1] for x in epoch_timeline],
        "sleep_stage": [x[2] for x in epoch_timeline],
    })

    # Merge bandpower (stage + start_s based)
    bp_merge = bandpower_df.rename(columns={"start_s": "epoch_start_time", "stage": "sleep_stage"})
    epoch_df = epoch_df.merge(
        bp_merge[["epoch_start_time", "sleep_stage", "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"]],
        on=["epoch_start_time", "sleep_stage"],
        how="left",
    )

    # Merge HRV and CPC
    epoch_df = epoch_df.merge(
        hrv_df[["epoch", "n_beats", "hr_mean_bpm", "rmssd_ms", "sdnn_ms", "pnn50_pct"]],
        on="epoch",
        how="left",
    )
    epoch_df = epoch_df.merge(cpc_epoch_df, on="epoch", how="left")

    # Per-epoch HEP/Delta (1s and 30s metrics + correlations)
    hep_30s_values: List[float] = []
    delta_30s_values: List[float] = []
    one_sec_corrs: List[Dict[str, float]] = []
    hep_stage_acc: Dict[str, List[float]] = {k: [] for k in STAGE_ORDER}
    delta_stage_acc: Dict[str, List[float]] = {k: [] for k in STAGE_ORDER}
    per_epoch_payload: List[Dict[str, object]] = []

    logging.info("[%s] HEP + Delta features", patient_id)
    for idx, row in epoch_df.iterrows():
        ecg_epoch_row = ecg_epoch_df.iloc[idx] if idx < len(ecg_epoch_df) else None
        stage = row["sleep_stage"]

        hep_values = hep_metric(eeg_filtered, ecg_epoch_row, sf) if ecg_epoch_row is not None else None
        delta_values = delta_power_1s(eeg_filtered, (row["epoch_start_time"], row["epoch_end_time"]), sf)

        if hep_values is None:
            hep_values = []
        if delta_values is None:
            delta_values = []

        n = min(len(hep_values), len(delta_values))
        hep_values = list(np.asarray(hep_values[:n], dtype=float))
        delta_values = list(np.asarray(delta_values[:n], dtype=float))

        corr_1s = _safe_corr(hep_values, delta_values)
        one_sec_corrs.append(corr_1s)

        hep30 = float(hep_metric_30s(eeg_filtered, ecg_epoch_row, sf)) if ecg_epoch_row is not None else np.nan
        delta30 = float(delta_power_30s(eeg_filtered, ecg_epoch_row, sf)) if ecg_epoch_row is not None else np.nan
        hep_30s_values.append(hep30)
        delta_30s_values.append(delta30)

        hep_stage_acc[stage].extend(hep_values)
        delta_stage_acc[stage].extend(delta_values)

        per_epoch_payload.append(
            {
                "epoch_index": int(row["epoch"]),
                "epoch_start_time": float(row["epoch_start_time"]),
                "epoch_end_time": float(row["epoch_end_time"]),
                "sleep_stage": stage,
                "hep_values_1s": np.asarray(hep_values, dtype=float),
                "delta_values_1s": np.asarray(delta_values, dtype=float),
                "corr_1s": corr_1s,
            }
        )

    corr30 = _rolling_corr(hep_30s_values, delta_30s_values, window_epochs=corr30_window_epochs)

    epoch_df["pearson_r_1s"] = [x["pearson_r"] for x in one_sec_corrs]
    epoch_df["pearson_p_1s"] = [x["pearson_p"] for x in one_sec_corrs]
    epoch_df["spearman_r_1s"] = [x["spearman_r"] for x in one_sec_corrs]
    epoch_df["spearman_p_1s"] = [x["spearman_p"] for x in one_sec_corrs]
    epoch_df["hep_30s_amplitude"] = hep_30s_values
    epoch_df["delta_30s_power"] = delta_30s_values
    epoch_df["pearson_r_30s"] = [x["pearson_r"] for x in corr30]
    epoch_df["pearson_p_30s"] = [x["pearson_p"] for x in corr30]
    epoch_df["spearman_r_30s"] = [x["spearman_r"] for x in corr30]
    epoch_df["spearman_p_30s"] = [x["spearman_p"] for x in corr30]

    # Sleep-stage correlation CSV
    stage_corr_rows = []
    for stage in STAGE_ORDER:
        metrics = _safe_corr(hep_stage_acc[stage], delta_stage_acc[stage])
        stage_corr_rows.append(
            {
                "patient_id": patient_id,
                "group": group,
                "sleep_stage": stage,
                **metrics,
            }
        )
    stage_corr_df = pd.DataFrame(stage_corr_rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_csv = output_dir / f"{patient_id}_epochs.csv"
    stage_csv = output_dir / f"{patient_id}_stage_correlations.csv"
    h5_path = output_dir / f"{patient_id}.h5"

    epoch_df.to_csv(epoch_csv, index=False)
    stage_corr_df.to_csv(stage_csv, index=False)

    logging.info("[%s] Writing HDF5: %s", patient_id, h5_path)
    with h5py.File(h5_path, "w") as h5f:
        patient_group = h5f.create_group(patient_id)
        epochs_group = patient_group.create_group("Epochs")

        for payload, corr30_epoch in zip(per_epoch_payload, corr30):
            grp = epochs_group.create_group(f"Epoch_{payload['epoch_index'] + 1}")
            grp.create_dataset("Epoch Start Time", data=payload["epoch_start_time"])
            grp.create_dataset("Epoch End Time", data=payload["epoch_end_time"])
            grp.create_dataset("sleep stage", data=np.string_(payload["sleep_stage"]))

            grp.create_dataset("HEP values 1 second window", data=payload["hep_values_1s"])
            grp.create_dataset("Delta power 1 second window", data=payload["delta_values_1s"])

            grp.create_dataset("Pearson correlation pearson_r 1 second window", data=payload["corr_1s"]["pearson_r"])
            grp.create_dataset("Pearson correlation pearson_p 1 second window", data=payload["corr_1s"]["pearson_p"])
            grp.create_dataset("Spearman correlation spearman_r 1 second window", data=payload["corr_1s"]["spearman_r"])
            grp.create_dataset("Spearman correlation spearman_p 1 second window", data=payload["corr_1s"]["spearman_p"])

            grp.create_dataset("Pearson correlation pearson_r 30 second window", data=corr30_epoch["pearson_r"])
            grp.create_dataset("Pearson correlation pearson_p 30 second window", data=corr30_epoch["pearson_p"])
            grp.create_dataset("Spearman correlation spearman_r 30 second window", data=corr30_epoch["spearman_r"])
            grp.create_dataset("Spearman correlation spearman_p 30 second window", data=corr30_epoch["spearman_p"])

    logging.info("[%s] Done. Files: %s | %s | %s", patient_id, epoch_csv.name, stage_csv.name, h5_path.name)


def _parse_patients(arg_patients: Optional[Sequence[str]]) -> List[str]:
    if arg_patients:
        return list(arg_patients)
    return CONTROL_IDS + RBD_IDS


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export per-patient epoch features to CSV and HDF5.")
    parser.add_argument("--edf-dir", type=Path, required=True, help="Directory with EDF files.")
    parser.add_argument("--hypnogram-dir", type=Path, required=True, help="Directory with hypnogram CSV/TSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for CSV/HDF5 files.")
    parser.add_argument("--patients", nargs="*", default=None, help="Patient IDs to process. Default: all listed controls + RBD.")
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--cpc-window-epochs", type=int, default=5)
    parser.add_argument("--corr30-window-epochs", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")

    patients = _parse_patients(args.patients)
    failures = []

    for patient_id in patients:
        try:
            edf_path = _find_input_file(args.edf_dir, patient_id, (".edf", ".EDF"))
            hyp_path = _find_input_file(args.hypnogram_dir, patient_id, (".csv", ".CSV", ".tsv", ".TSV", ".txt", ".TXT"))
            process_patient(
                patient_id=patient_id,
                edf_path=edf_path,
                hypnogram_path=hyp_path,
                output_dir=args.output_dir,
                epoch_seconds=args.epoch_seconds,
                cpc_window_epochs=args.cpc_window_epochs,
                corr30_window_epochs=args.corr30_window_epochs,
            )
        except Exception as exc:  # continue processing other patients
            logging.exception("[%s] Failed: %s", patient_id, exc)
            failures.append(patient_id)

    if failures:
        logging.warning("Completed with failures (%d/%d): %s", len(failures), len(patients), ", ".join(failures))
        return 1

    logging.info("All patients processed successfully (%d).", len(patients))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
