#!/usr/bin/env python3
"""
tune_threshold.py
=================

Per-phenomenology decision-threshold calibration from cross-validated
out-of-fold (OOF) predictions produced by ``cv_sam3.py``.

The 0.5 default threshold is almost always wrong for imbalanced clinical
labels: a model can have an excellent ROC-AUC yet predict "negative" for
everyone at 0.5 (F1 = 0). This script sweeps thresholds and reports, per
phenomenology, the operating point that maximises a chosen objective:

    - "f1"      : maximise F1 (default)
    - "youden"  : maximise Youden's J = sensitivity + specificity - 1
    - "balanced": maximise balanced accuracy = (sens + spec) / 2

It works at the **patient level** by default (the clinically relevant unit),
using the per-patient scores in ``cv_oof_patient_predictions.csv`` produced by
cv_sam3.py. It can also calibrate at the **window level** from
``cv_oof_window_predictions.csv.gz``.

Inputs
------
  --cv_dir       The CV output directory (e.g. runs/sam3_tier1/cv_5fold).
  --level        "patient" (default) or "window".
  --objective    "f1" | "youden" | "balanced"   (default: f1)
  --min_pos      Skip labels with fewer than this many positive units.

Outputs (written into --cv_dir)
-------------------------------
  thresholds_<level>_<objective>.json     {label: best_threshold}
  threshold_sweep_<level>.csv              full sweep table (for plots)
  threshold_summary_<level>_<objective>.csv  metrics at the chosen threshold

Usage
-----
    python tune_threshold.py --cv_dir runs/sam3_tier1/cv_5fold --level patient --objective f1
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def sweep_metrics(y_true: np.ndarray, score: np.ndarray,
                  thresholds: np.ndarray) -> pd.DataFrame:
    """Compute per-threshold metrics for one label."""
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    rows = []
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    for thr in thresholds:
        y_pred = (score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        youden = (sens + spec - 1) if (np.isfinite(sens) and np.isfinite(spec)) else np.nan
        bal_acc = ((sens + spec) / 2) if (np.isfinite(sens) and np.isfinite(spec)) else np.nan
        rows.append({
            "threshold": float(thr),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "sensitivity": float(sens) if np.isfinite(sens) else np.nan,
            "specificity": float(spec) if np.isfinite(spec) else np.nan,
            "f1": float(f1),
            "accuracy": float(acc),
            "youden": float(youden) if np.isfinite(youden) else np.nan,
            "balanced_accuracy": float(bal_acc) if np.isfinite(bal_acc) else np.nan,
            "n_pos": n_pos, "n_neg": n_neg,
        })
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_dir", type=Path, required=True)
    ap.add_argument("--level", choices=["patient", "window"], default="patient")
    ap.add_argument("--objective", choices=["f1", "youden", "balanced"],
                    default="f1")
    ap.add_argument("--min_pos", type=int, default=3,
                    help="Skip labels with fewer than this many positive units.")
    ap.add_argument("--grid_step", type=float, default=0.01)
    args = ap.parse_args(argv)

    obj_col = {"f1": "f1", "youden": "youden",
               "balanced": "balanced_accuracy"}[args.objective]

    # --- Load OOF predictions ---
    if args.level == "patient":
        path = args.cv_dir / "cv_oof_patient_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        score_col, true_col, label_col = "score", "y_true", "label"
    else:
        path = args.cv_dir / "cv_oof_window_predictions.csv.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            df = pd.read_csv(f)
        score_col, true_col, label_col = "proba", "y_true", "label"

    print(f"[INFO] Loaded {len(df)} {args.level}-level OOF rows from {path.name}")

    thresholds = np.arange(0.01, 1.00, float(args.grid_step))

    best_thresholds: Dict[str, float] = {}
    summary_rows = []
    sweep_all = []

    for label, g in df.groupby(label_col):
        y = g[true_col].to_numpy(dtype=int)
        s = g[score_col].to_numpy(dtype=float)
        ok = np.isfinite(s)
        y, s = y[ok], s[ok]
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))

        if n_pos < args.min_pos or n_neg < 1:
            print(f"  [SKIP] {label}: n_pos={n_pos} n_neg={n_neg}")
            continue

        sweep = sweep_metrics(y, s, thresholds)
        sweep.insert(0, "label", label)
        sweep_all.append(sweep)

        valid = sweep.dropna(subset=[obj_col])
        if valid.empty:
            print(f"  [SKIP] {label}: no valid threshold")
            continue
        best_idx = valid[obj_col].idxmax()
        best = valid.loc[best_idx]
        best_thr = float(best["threshold"])
        best_thresholds[label] = best_thr

        summary_rows.append({
            "label": label,
            "best_threshold": best_thr,
            "objective": args.objective,
            "objective_value": float(best[obj_col]),
            "f1": float(best["f1"]),
            "sensitivity": float(best["sensitivity"]),
            "specificity": float(best["specificity"]),
            "accuracy": float(best["accuracy"]),
            "youden": float(best["youden"]),
            "balanced_accuracy": float(best["balanced_accuracy"]),
            "tp": int(best["tp"]), "fp": int(best["fp"]),
            "fn": int(best["fn"]), "tn": int(best["tn"]),
            "n_pos": n_pos, "n_neg": n_neg,
        })

        print(f"  {label:<14s} best_thr={best_thr:.2f}  "
              f"{args.objective}={best[obj_col]:.3f}  "
              f"F1={best['f1']:.3f}  sens={best['sensitivity']:.3f}  "
              f"spec={best['specificity']:.3f}")

    # --- Save ---
    thr_path = args.cv_dir / f"thresholds_{args.level}_{args.objective}.json"
    thr_path.write_text(json.dumps(best_thresholds, indent=2))

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        sum_path = args.cv_dir / f"threshold_summary_{args.level}_{args.objective}.csv"
        summary.to_csv(sum_path, index=False)
    if sweep_all:
        sweep_df = pd.concat(sweep_all, ignore_index=True)
        sweep_path = args.cv_dir / f"threshold_sweep_{args.level}.csv"
        sweep_df.to_csv(sweep_path, index=False)

    print(f"\n[OK] Thresholds -> {thr_path}")
    if summary_rows:
        print(f"[OK] Summary    -> {sum_path}")
        print(f"\n=== Calibrated metrics ({args.level}-level, "
              f"objective={args.objective}) ===")
        cols = ["label", "best_threshold", "f1", "sensitivity",
                "specificity", "balanced_accuracy", "n_pos", "n_neg"]
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(summary[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
