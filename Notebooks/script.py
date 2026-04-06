import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

STAGE_LABEL_MAP = {"W": "Wake", "S1": "N1", "S2": "N2", "S3": "N3", "S4": "N3", "R": "REM"}
STAGE_ORDER = ["REM", "N1", "N2", "N3", "Wake"]


def _safe_corr(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = ~np.isnan(xa) & ~np.isnan(ya)
    xa, ya = xa[mask], ya[mask]

    if xa.size < 3 or np.allclose(xa, xa[0]) or np.allclose(ya, ya[0]):
        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }

    p_r, p_p = pearsonr(xa, ya)
    s_r, s_p = spearmanr(xa, ya)
    return {
        "pearson_r": float(p_r),
        "pearson_p": float(p_p),
        "spearman_r": float(s_r),
        "spearman_p": float(s_p),
    }


def _rolling_corr(values_x: List[float], values_y: List[float], window_epochs: int) -> List[Dict[str, float]]:
    corr = []
    for i in range(len(values_x)):
        start = max(0, i - window_epochs + 1)
        corr.append(_safe_corr(values_x[start : i + 1], values_y[start : i + 1]))
    return corr


def _find_input_file(base_dir: Path, patient_id: str, suffixes: Tuple[str, ...]) -> Path:
    candidates: List[Path] = []
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

    hyp = hyp[hyp["Sleep Stage"].isin(set(STAGE_ORDER))].reset_index(drop=True)
    return hyp[["Sleep Stage", "Duration[s]", "onset_s"]]


def _stage_epochs_dict(hypnogram_df: pd.DataFrame) -> Dict[str, List[Tuple[float, float]]]:
    return {stage: compute_stage_epochs(hypnogram_df, stage) for stage in STAGE_ORDER}


def _expand_cpc_to_epochs(cpc_df: pd.DataFrame, n_epochs: int, window_epochs: int) -> pd.DataFrame:
    out = pd.DataFrame({"epoch": np.arange(n_epochs), "hfc": np.nan, "lfc": np.nan, "ratio": np.nan})
    if cpc_df.empty:
        return out

    for _, row in cpc_df.iterrows():
        start = int(row["epoch"])
        end = min(start + window_epochs, n_epochs)
        out.loc[start : end - 1, ["hfc", "lfc", "ratio"]] = [
            row.get("HFC", np.nan),
            row.get("LFC", np.nan),
            row.get("LFC/HFC", np.nan),
        ]
    return out


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

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sf = float(raw.info["sfreq"])

    hypnogram = _load_hypnogram(hypnogram_path, epoch_seconds)
    timeline = []
    for _, row in hypnogram.iterrows():
        t0 = float(row["onset_s"])
        t1 = t0 + float(row["Duration[s]"])
        timeline.append((t0, t1, row["Sleep Stage"]))

    epochs_only = [(t0, t1) for t0, t1, _ in timeline]

    eeg = preprocess_eeg(raw)
    ecg_clean = preprocess_ecg(raw)
    ecg_raw = raw.get_data(picks="ECG1-ECG2")[0]

    ecg_epoch_df = extract_ecg_per_epoch(ecg_raw, ecg_clean, epochs_only, sf, detect_peaks=True)
    hrv_df = hrv_per_epoch(ecg_clean, epochs_only, sf)

    resp = extract_resp_from_ecg(ecg_clean, sf)
    cpc_df = hpc_metric(ecg_clean, resp, epochs_only, sf, window_epochs=cpc_window_epochs)
    cpc_epoch_df = _expand_cpc_to_epochs(cpc_df, n_epochs=len(epochs_only), window_epochs=cpc_window_epochs)

    bandpower_df = eeg_bandpower(eeg, _stage_epochs_dict(hypnogram), sf)
    bp_map = {
        (float(r["start_s"]), str(r["stage"])): {
            "delta": float(r.get("delta_power", np.nan)),
            "theta": float(r.get("theta_power", np.nan)),
            "alpha": float(r.get("alpha_power", np.nan)),
            "beta": float(r.get("beta_power", np.nan)),
            "gamma": float(r.get("gamma_power", np.nan)),
        }
        for _, r in bandpower_df.iterrows()
    }

    hrv_map = {
        int(r["epoch"]): {
            "n_beats": float(r.get("n_beats", np.nan)),
            "hr_mean_bpm": float(r.get("hr_mean_bpm", np.nan)),
            "rmssd": float(r.get("rmssd_ms", np.nan)),
            "sdnn": float(r.get("sdnn_ms", np.nan)),
            "pnn50": float(r.get("pnn50_pct", np.nan)),
        }
        for _, r in hrv_df.iterrows()
    }

    cpc_map = {
        int(r["epoch"]): {
            "hfc": float(r.get("hfc", np.nan)),
            "lfc": float(r.get("lfc", np.nan)),
            "ratio": float(r.get("ratio", np.nan)),
        }
        for _, r in cpc_epoch_df.iterrows()
    }

    hep30_list: List[float] = []
    delta30_list: List[float] = []
    epoch_payload = []

    stage_1s_hep: Dict[str, List[float]] = {s: [] for s in STAGE_ORDER}
    stage_1s_delta: Dict[str, List[float]] = {s: [] for s in STAGE_ORDER}
    stage_30s_hep: Dict[str, List[float]] = {s: [] for s in STAGE_ORDER}
    stage_30s_delta: Dict[str, List[float]] = {s: [] for s in STAGE_ORDER}

    for i, (t0, t1, stage) in enumerate(timeline):
        ecg_row = ecg_epoch_df.iloc[i] if i < len(ecg_epoch_df) else None

        hep_1s_vals = hep_metric(eeg, ecg_row, sf) if ecg_row is not None else None
        delta_1s_vals = delta_power_1s(eeg, (t0, t1), sf)
        hep_1s_vals = [] if hep_1s_vals is None else list(np.asarray(hep_1s_vals, dtype=float))
        delta_1s_vals = [] if delta_1s_vals is None else list(np.asarray(delta_1s_vals, dtype=float))

        n = min(len(hep_1s_vals), len(delta_1s_vals))
        hep_1s_vals, delta_1s_vals = hep_1s_vals[:n], delta_1s_vals[:n]
        corr_1s = _safe_corr(hep_1s_vals, delta_1s_vals)

        hep_30s = float(hep_metric_30s(eeg, ecg_row, sf)) if ecg_row is not None else np.nan
        delta_30s = float(delta_power_30s(eeg, ecg_row, sf)) if ecg_row is not None else np.nan
        hep30_list.append(hep_30s)
        delta30_list.append(delta_30s)

        stage_1s_hep[stage].extend(hep_1s_vals)
        stage_1s_delta[stage].extend(delta_1s_vals)
        stage_30s_hep[stage].append(hep_30s)
        stage_30s_delta[stage].append(delta_30s)

        epoch_payload.append(
            {
                "epoch_index": i + 1,
                "start_time": t0,
                "end_time": t1,
                "sleep_stage": stage,
                "bandpower": bp_map.get((t0, stage), {"delta": np.nan, "theta": np.nan, "alpha": np.nan, "beta": np.nan, "gamma": np.nan}),
                "hrv": hrv_map.get(i, {"n_beats": np.nan, "hr_mean_bpm": np.nan, "rmssd": np.nan, "sdnn": np.nan, "pnn50": np.nan}),
                "cpc": cpc_map.get(i, {"hfc": np.nan, "lfc": np.nan, "ratio": np.nan}),
                "hep_1s_corr": corr_1s,
            }
        )

    corr30_list = _rolling_corr(hep30_list, delta30_list, window_epochs=corr30_window_epochs)

    stage_summary: Dict[str, Dict[str, float]] = {}
    for stage in STAGE_ORDER:
        c1 = _safe_corr(stage_1s_hep[stage], stage_1s_delta[stage])
        c30 = _safe_corr(stage_30s_hep[stage], stage_30s_delta[stage])
        stage_summary[stage] = {
            "pearson_1s_r": c1["pearson_r"],
            "pearson_1s_p": c1["pearson_p"],
            "spearman_1s_r": c1["spearman_r"],
            "spearman_1s_p": c1["spearman_p"],
            "pearson_30s_r": c30["pearson_r"],
            "pearson_30s_p": c30["pearson_p"],
            "spearman_30s_r": c30["spearman_r"],
            "spearman_30s_p": c30["spearman_p"],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / f"{patient_id}.h5"

    with h5py.File(h5_path, "w") as h5f:
        patient_grp = h5f.create_group(patient_id)
        patient_grp.create_dataset("Group", data=np.bytes_(group))

        epochs_grp = patient_grp.create_group("epochs")
        for ep, corr30 in zip(epoch_payload, corr30_list):
            ep_grp = epochs_grp.create_group(f"epoch_{ep['epoch_index']:04d}")
            ep_grp.create_dataset("start_time", data=float(ep["start_time"]))
            ep_grp.create_dataset("end_time", data=float(ep["end_time"]))
            ep_grp.create_dataset("sleep_stage", data=np.bytes_(ep["sleep_stage"]))

            bp_grp = ep_grp.create_group("eeg_bandpower")
            bp_grp.create_dataset("delta", data=ep["bandpower"]["delta"])
            bp_grp.create_dataset("theta", data=ep["bandpower"]["theta"])
            bp_grp.create_dataset("alpha", data=ep["bandpower"]["alpha"])
            bp_grp.create_dataset("beta", data=ep["bandpower"]["beta"])
            bp_grp.create_dataset("gamma", data=ep["bandpower"]["gamma"])

            hrv_grp = ep_grp.create_group("hrv")
            hrv_grp.create_dataset("n_beats", data=ep["hrv"]["n_beats"])
            hrv_grp.create_dataset("hr_mean_bpm", data=ep["hrv"]["hr_mean_bpm"])
            hrv_grp.create_dataset("rmssd", data=ep["hrv"]["rmssd"])
            hrv_grp.create_dataset("sdnn", data=ep["hrv"]["sdnn"])
            hrv_grp.create_dataset("pnn50", data=ep["hrv"]["pnn50"])

            cpc_grp = ep_grp.create_group("cpc")
            cpc_grp.create_dataset("hfc", data=ep["cpc"]["hfc"])
            cpc_grp.create_dataset("lfc", data=ep["cpc"]["lfc"])
            cpc_grp.create_dataset("ratio", data=ep["cpc"]["ratio"])

            hep_grp = ep_grp.create_group("hep")
            hep_grp.create_dataset("pearson_1s_r", data=ep["hep_1s_corr"]["pearson_r"])
            hep_grp.create_dataset("pearson_1s_p", data=ep["hep_1s_corr"]["pearson_p"])
            hep_grp.create_dataset("spearman_1s_r", data=ep["hep_1s_corr"]["spearman_r"])
            hep_grp.create_dataset("spearman_1s_p", data=ep["hep_1s_corr"]["spearman_p"])
            hep_grp.create_dataset("pearson_30s_r", data=corr30["pearson_r"])
            hep_grp.create_dataset("pearson_30s_p", data=corr30["pearson_p"])
            hep_grp.create_dataset("spearman_30s_r", data=corr30["spearman_r"])
            hep_grp.create_dataset("spearman_30s_p", data=corr30["spearman_p"])

        stage_grp = patient_grp.create_group("stage_summary")
        for stage in STAGE_ORDER:
            sg = stage_grp.create_group(stage)
            corr_grp = sg.create_group("hep_delta_correlation")
            corr = stage_summary[stage]
            corr_grp.create_dataset("pearson_1s_r", data=corr["pearson_1s_r"])
            corr_grp.create_dataset("pearson_1s_p", data=corr["pearson_1s_p"])
            corr_grp.create_dataset("spearman_1s_r", data=corr["spearman_1s_r"])
            corr_grp.create_dataset("spearman_1s_p", data=corr["spearman_1s_p"])
            corr_grp.create_dataset("pearson_30s_r", data=corr["pearson_30s_r"])
            corr_grp.create_dataset("pearson_30s_p", data=corr["pearson_30s_p"])
            corr_grp.create_dataset("spearman_30s_r", data=corr["spearman_30s_r"])
            corr_grp.create_dataset("spearman_30s_p", data=corr["spearman_30s_p"])

    logging.info("[%s] Wrote %s", patient_id, h5_path)


def _parse_patients(arg_patients: Optional[Sequence[str]]) -> List[str]:
    return list(arg_patients) if arg_patients else CONTROL_IDS + RBD_IDS


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate one HDF5 file per patient with epoch and stage summaries.")
    parser.add_argument("--edf-dir", type=Path, required=True)
    parser.add_argument("--hypnogram-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--cpc-window-epochs", type=int, default=5)
    parser.add_argument("--corr30-window-epochs", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    failures: List[str] = []
    for patient_id in _parse_patients(args.patients):
        try:
            edf = _find_input_file(args.edf_dir, patient_id, (".edf", ".EDF"))
            hyp = _find_input_file(args.hypnogram_dir, patient_id, (".csv", ".CSV", ".tsv", ".TSV", ".txt", ".TXT"))
            process_patient(
                patient_id=patient_id,
                edf_path=edf,
                hypnogram_path=hyp,
                output_dir=args.output_dir,
                epoch_seconds=args.epoch_seconds,
                cpc_window_epochs=args.cpc_window_epochs,
                corr30_window_epochs=args.corr30_window_epochs,
            )
        except Exception as exc:
            logging.exception("[%s] Failed: %s", patient_id, exc)
            failures.append(patient_id)

    if failures:
        logging.warning("Completed with failures (%d): %s", len(failures), ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
