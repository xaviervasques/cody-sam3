#!/usr/bin/env python3
"""
inference_sam3.py
=================

Run TabICLv2 inference on SAM 3 dense per-frame time-series (no labels).

For each input file:
  1) Apply the tier-specific derived signals (posture / regional) as the
     training pipeline did.
  2) Cut the timeseries into 300-frame sliding windows with stride 150
     (mirrors the cody-2 paper protocol: 10-second windows at 30 fps with
     50% overlap).
  3) Compute the 19 cody-2 statistical descriptors per signal per window.
  4) Align the feature schema to the training bundle.
  5) Predict per-window probabilities with each per-label model.
  6) Aggregate at the patient level using --agg_map (or a default per-label
     method) and threshold with --thresholds (or a global default).
  7) Save window-level probabilities, patient-level predictions and a
     data-quality summary.

Inputs
------
  --infer_root    Directory containing *_sam3_timeseries.xlsx files. Subjects
                  are auto-discovered by file stem (`subject_id` column).
  --bundle_dir    Output directory of a previous train_sam3 run; we read
                  `bundle_meta.json` from it.
  --out_dir       Where to write the inference outputs.

Optional inputs
---------------
  --agg_map       JSON {label: agg_method}, e.g.
                  {"Dystonia":"p95","Tremor":"p90","Myoclonus":"max", ...}.
                  If absent, uses --default_agg (default: "p95").
  --thresholds    JSON {label: float}. If absent, uses --default_threshold
                  (default 0.5).
  --tier          Override the tier used (default: read from bundle_meta).
  --window_size   Default 300 frames.
  --stride        Default 150 frames (50% overlap).
  --save_windows  If set, save the window-level probabilities.

Outputs
-------
  out_dir/
  ├── reports/
  │   ├── inference_config.json
  │   └── tables/
  │       ├── inference_window_predictions.csv.gz   (if --save_windows)
  │       ├── inference_patient_predictions.csv
  │       └── inference_data_quality.csv

Usage example
-------------
    python inference_sam3.py \\
        --infer_root /path/to/sam3/outputs_dataset_2 \\
        --bundle_dir /path/to/runs/sam3_tier2 \\
        --out_dir    /path/to/runs/sam3_tier2/infer_dataset_2
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_SYMPTOMS = ["Dystonia", "Tremor", "Myoclonus", "Chorea",
                    "Athetosis", "Ballismus", "Stereotypies", "Tics"]


# ---------------------------------------------------------------------------
# Helper imports (cody-2 + sam3_features)
# ---------------------------------------------------------------------------

def _import_cody2(cody2_root: Optional[Path] = None):
    """Import the vendored cody-2 helpers from cody2_utils.py.

    ``cody2_root`` is kept for backwards compatibility but is ignored.
    """
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import cody2_utils as cody2_inf  # noqa: E402
    return cody2_inf


def _import_sam3_features():
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import sam3_features  # noqa: E402
    return sam3_features


# ---------------------------------------------------------------------------
# SAM 3 file discovery
# ---------------------------------------------------------------------------

def find_sam3_timeseries(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*_sam3_timeseries.xlsx"):
        if "__MACOSX" in str(p) or p.name.startswith("~$"):
            continue
        files.append(p)
    return sorted(files)


def patient_id_from_sam3(df: pd.DataFrame, path: Path) -> str:
    if "subject_id" in df.columns and df["subject_id"].notna().any():
        return str(df["subject_id"].dropna().iloc[0]).strip().upper()
    return path.stem.replace("_sam3_timeseries", "").strip().upper()


def dataset_name_from_path(p: Path) -> str:
    for parent in p.parents:
        nm = parent.name.lower()
        if nm.startswith("dataset_") or nm.startswith("outputs_dataset_"):
            return nm.replace("outputs_", "")
    return "dataset_unknown"


# ---------------------------------------------------------------------------
# Featurisation: take a SAM 3 timeseries, slide windows of 300 frames,
# compute the 19 statistical descriptors per signal per window.
# ---------------------------------------------------------------------------

def featurize_one_sam3_file(
    xlsx_path: Path,
    cody2,
    sam3_features,
    tier: int,
    fps: float,
    robust_norm: bool,
    window_size: int,
    stride: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    # Data-quality stats
    n_rows = len(df)
    if "patient_detected" in df.columns:
        det_rate = float(df["patient_detected"].mean())
    else:
        det_rate = float("nan")

    # Optional: drop frames where the patient is not detected
    if "patient_detected" in df.columns:
        df = df[df["patient_detected"] == 1].reset_index(drop=True)

    # Apply tier-specific derived signals
    df = sam3_features.add_derived_signals(df, tier=tier)
    sig_cols = sam3_features.get_tier_signal_cols(df, tier=tier)

    patient_id = patient_id_from_sam3(df, xlsx_path)
    dataset_nm = dataset_name_from_path(xlsx_path)

    n = len(df)
    if n == 0:
        return pd.DataFrame(), dict(
            patient_id=patient_id, source_file=xlsx_path.name,
            n_rows=int(n_rows), n_detected_rows=0,
            detection_rate=det_rate, n_windows=0,
        )

    # Sliding windows
    ws = int(window_size); st = int(stride)
    if n < ws:
        windows = [(0, n - 1)]
    else:
        starts = list(range(0, n - ws + 1, st))
        windows = [(s, s + ws - 1) for s in starts]
        if not windows:
            windows = [(0, n - 1)]

    rows = []
    for (start, end) in windows:
        block = df.iloc[start:end + 1]
        row = {
            "patient_id": patient_id,
            "From": float(start),
            "To": float(end),
            "dataset": dataset_nm,
            "source_file": xlsx_path.name,
        }
        for c in sig_cols:
            v = pd.to_numeric(block[c], errors="coerce").to_numpy(dtype=float)
            feats = cody2.compute_features_1d(v, fps=fps, robust_norm=robust_norm)
            for k, val in feats.items():
                row[f"f__{c}__{k}"] = val
        rows.append(row)

    out = pd.DataFrame(rows)
    out["patient_id"] = out["patient_id"].astype("string").str.strip().str.upper()

    quality = dict(
        patient_id=patient_id, source_file=xlsx_path.name,
        n_rows=int(n_rows), n_detected_rows=int(n),
        detection_rate=det_rate, n_windows=int(len(windows)),
    )
    return out, quality


# ---------------------------------------------------------------------------
# Schema alignment & aggregation
# ---------------------------------------------------------------------------

def ensure_feature_schema(dfw: pd.DataFrame, feat_cols_train: List[str]
                          ) -> pd.DataFrame:
    df = dfw.copy()
    # Add missing features as 0 (cody-2 policy)
    missing = [c for c in feat_cols_train if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    # Drop unexpected features (extras)
    keep = ["patient_id", "From", "To", "dataset", "source_file"] + feat_cols_train
    extras = [c for c in df.columns if c not in keep]
    if extras:
        df = df.drop(columns=extras)
    # Fill NaNs with 0 (cody-2 policy)
    df[feat_cols_train] = df[feat_cols_train].fillna(0.0)
    return df


def agg_probs(probs: np.ndarray, method: str) -> float:
    if probs.size == 0:
        return float("nan")
    method = str(method).lower()
    if method == "max":
        return float(np.max(probs))
    if method == "mean":
        return float(np.mean(probs))
    if method == "median":
        return float(np.median(probs))
    if method == "noisy_or":
        return float(1.0 - np.prod(1.0 - probs))
    m = re.match(r"^p(\d{2,3})$", method)
    if m:
        q = float(m.group(1))
        return float(np.percentile(probs, q))
    raise ValueError(f"Unknown aggregation method: {method}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer_root", type=Path, required=True)
    ap.add_argument("--bundle_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--cody2_root", type=Path, default=None,
                    help="Deprecated: ignored. The cody-2 helpers are now "
                         "vendored in cody2_utils.py next to this script.")
    ap.add_argument("--agg_map", type=Path, default=None,
                    help="JSON {label: agg_method}.")
    ap.add_argument("--thresholds", type=Path, default=None,
                    help="JSON {label: float}.")
    ap.add_argument("--default_agg", type=str, default="p95")
    ap.add_argument("--default_threshold", type=float, default=0.5)
    ap.add_argument("--tier", type=int, default=None,
                    help="Override the tier (default: read from bundle).")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--robust_norm", type=lambda s: str(s).lower() in
                    ("1", "true", "yes"), default=None,
                    help="Override robust_norm (default: read from bundle).")
    ap.add_argument("--window_size", type=int, default=300)
    ap.add_argument("--stride", type=int, default=150)
    ap.add_argument("--save_windows", action="store_true")
    args = ap.parse_args(argv)

    cody2 = _import_cody2(args.cody2_root)
    sam3_features = _import_sam3_features()
    from tabicl import TabICLClassifier

    bundle_path = args.bundle_dir / "bundle_meta.json"
    meta = json.loads(bundle_path.read_text())
    symptoms = list(meta["symptoms"])
    feat_cols_train = list(meta["feature_columns"])
    tier = int(args.tier) if args.tier is not None else int(meta.get("tier", 2))
    robust_norm = (bool(args.robust_norm) if args.robust_norm is not None
                   else bool(meta.get("robust_norm", False)))

    # Load aggregation map / thresholds
    agg_map = {s: args.default_agg for s in symptoms}
    if args.agg_map and args.agg_map.exists():
        agg_map.update(json.loads(args.agg_map.read_text()))
    thresholds = {s: float(args.default_threshold) for s in symptoms}
    if args.thresholds and args.thresholds.exists():
        thresholds.update({k: float(v) for k, v in
                           json.loads(args.thresholds.read_text()).items()})

    out_tables = args.out_dir / "reports" / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    # ---- featurise every input file ----
    xlsx_files = find_sam3_timeseries(args.infer_root)
    if not xlsx_files:
        raise RuntimeError(f"No *_sam3_timeseries.xlsx under {args.infer_root}")
    print(f"[INFO] {len(xlsx_files)} SAM 3 timeseries files to process")

    win_dfs, quality_rows = [], []
    for i, p in enumerate(xlsx_files, 1):
        print(f"  [{i}/{len(xlsx_files)}] {p.relative_to(args.infer_root)}")
        try:
            dfp, qual = featurize_one_sam3_file(
                p, cody2, sam3_features,
                tier=tier, fps=float(args.fps),
                robust_norm=robust_norm,
                window_size=int(args.window_size),
                stride=int(args.stride),
            )
        except Exception as e:
            print(f"     [WARN] failed: {e}")
            quality_rows.append(dict(
                patient_id="?", source_file=p.name,
                n_rows=0, n_detected_rows=0,
                detection_rate=float("nan"), n_windows=0,
                error=str(e),
            ))
            continue
        if len(dfp) > 0:
            win_dfs.append(dfp)
        quality_rows.append(qual)

    if not win_dfs:
        raise RuntimeError("No file produced any window. Check the input.")

    dfw = pd.concat(win_dfs, axis=0, ignore_index=True)

    # ---- align feature schema ----
    dfw = ensure_feature_schema(dfw, feat_cols_train)

    # ---- predict per label ----
    X = dfw[feat_cols_train].to_numpy(dtype=float)
    proba_cols = {}
    for sym in symptoms:
        model_path = args.bundle_dir / "models" / f"label_{sym}" / "model.joblib"
        if not model_path.exists():
            print(f"  [SKIP] {sym}: no model at {model_path}")
            continue
        try:
            clf = TabICLClassifier.load(model_path)
        except Exception:
            import joblib
            clf = joblib.load(model_path)
        probs = clf.predict_proba(X)[:, 1]
        proba_cols[f"prob__{sym}"] = probs

    # ---- combine into window-level dataframe ----
    win = dfw[["patient_id", "From", "To", "dataset", "source_file"]].copy()
    for c, v in proba_cols.items():
        win[c] = v

    if args.save_windows:
        win_path = out_tables / "inference_window_predictions.csv.gz"
        with gzip.open(win_path, "wt", encoding="utf-8") as f:
            win.to_csv(f, index=False)
        print(f"[OK] {win_path}")

    # ---- patient-level aggregation + thresholding ----
    pat_rows = []
    for pid, g in win.groupby("patient_id", sort=True):
        row = {"patient_id": pid, "n_windows": int(len(g))}
        for sym in symptoms:
            col = f"prob__{sym}"
            if col not in g.columns:
                row[f"pred__{sym}"] = np.nan
                row[f"score__{sym}"] = np.nan
                row[f"agg__{sym}"] = agg_map.get(sym, args.default_agg)
                row[f"thr__{sym}"] = thresholds.get(sym, args.default_threshold)
                continue
            v = g[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            score = agg_probs(v, agg_map.get(sym, args.default_agg))
            thr = float(thresholds.get(sym, args.default_threshold))
            pred = int(score >= thr) if np.isfinite(score) else np.nan
            row[f"score__{sym}"] = score
            row[f"thr__{sym}"] = thr
            row[f"agg__{sym}"] = agg_map.get(sym, args.default_agg)
            row[f"pred__{sym}"] = pred
        pat_rows.append(row)
    patient_df = pd.DataFrame(pat_rows)
    patient_path = out_tables / "inference_patient_predictions.csv"
    patient_df.to_csv(patient_path, index=False)
    print(f"[OK] {patient_path}")

    # ---- data quality ----
    quality_df = pd.DataFrame(quality_rows)
    quality_path = out_tables / "inference_data_quality.csv"
    quality_df.to_csv(quality_path, index=False)
    print(f"[OK] {quality_path}")

    # ---- inference config ----
    config = {
        "bundle_dir": str(args.bundle_dir),
        "tier": tier,
        "robust_norm": robust_norm,
        "fps": float(args.fps),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "agg_map": agg_map,
        "thresholds": thresholds,
        "symptoms": symptoms,
    }
    (args.out_dir / "reports" / "inference_config.json").write_text(
        json.dumps(config, indent=2)
    )
    print("\n[DONE] inference complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
