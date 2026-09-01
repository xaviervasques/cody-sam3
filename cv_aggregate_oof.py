#!/usr/bin/env python3
"""
cv_aggregate_oof.py
===================

Compute *pooled* out-of-fold (OOF) metrics from the predictions saved by
``cv_sam3.py``. This is the statistically correct way to summarise
cross-validation when folds are small -- and the ONLY correct way for
Leave-One-Patient-Out (LOPO), where each fold has a single test patient and a
per-fold ROC is undefined.

Instead of averaging per-fold ROC-AUC (which is NaN when a fold has only one
class), we pool all OOF predictions across folds and compute a single
ROC-AUC / PR-AUC over the entire pooled set. Each patient (or window) appears
exactly once in the pool, having been predicted by a model that never saw it
in training -- so the pooled metric is a clean estimate of generalisation.

Inputs
------
  --cv_dir       A CV output directory produced by cv_sam3.py (5-fold or LOPO).
  --thresholds   Optional JSON {label: threshold}. If given, F1/sens/spec are
                 reported at those per-label thresholds. Otherwise 0.5.

Outputs (written into --cv_dir)
-------------------------------
  oof_pooled_patient_metrics.csv     pooled patient-level metrics per label
  oof_pooled_window_metrics.csv      pooled window-level metrics per label

Usage
-----
    python cv_aggregate_oof.py --cv_dir runs/sam3_tier1/cv_25fold
    python cv_aggregate_oof.py --cv_dir runs/sam3_tier1/cv_25fold \
        --thresholds runs/sam3_tier1/cv_25fold/thresholds_patient_f1.json
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


def pooled_metrics(y_true: np.ndarray, score: np.ndarray,
                   threshold: float) -> Dict[str, float]:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score, accuracy_score,
        confusion_matrix,
    )
    out: Dict[str, float] = {
        "n_total": int(len(y_true)),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
    }
    if out["n_pos"] == 0 or out["n_neg"] == 0:
        out.update({"roc_auc": np.nan, "pr_auc": np.nan, "f1": np.nan,
                    "accuracy": np.nan, "sensitivity": np.nan,
                    "specificity": np.nan, "balanced_accuracy": np.nan})
        return out
    out["roc_auc"] = float(roc_auc_score(y_true, score))
    out["pr_auc"] = float(average_precision_score(y_true, score))
    y_pred = (score >= threshold).astype(int)
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    out["sensitivity"] = float(sens) if np.isfinite(sens) else np.nan
    out["specificity"] = float(spec) if np.isfinite(spec) else np.nan
    out["balanced_accuracy"] = (float((sens + spec) / 2)
                                if np.isfinite(sens) and np.isfinite(spec)
                                else np.nan)
    out["tp"] = int(tp); out["fp"] = int(fp)
    out["fn"] = int(fn); out["tn"] = int(tn)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_dir", type=Path, required=True)
    ap.add_argument("--thresholds", type=Path, default=None,
                    help="Optional JSON {label: threshold}; default 0.5 each.")
    args = ap.parse_args(argv)

    thr_map: Dict[str, float] = {}
    if args.thresholds and args.thresholds.exists():
        thr_map = {k: float(v) for k, v in
                   json.loads(args.thresholds.read_text()).items()}
        print(f"[INFO] Loaded {len(thr_map)} per-label thresholds")

    # ---- Patient level ----
    pat_path = args.cv_dir / "cv_oof_patient_predictions.csv"
    if pat_path.exists():
        dfp = pd.read_csv(pat_path)
        rows = []
        for label, g in dfp.groupby("label"):
            y = g["y_true"].to_numpy(dtype=int)
            s = g["score"].to_numpy(dtype=float)
            ok = np.isfinite(s)
            thr = thr_map.get(label, 0.5)
            m = pooled_metrics(y[ok], s[ok], thr)
            m["label"] = label
            rows.append(m)
        pat = pd.DataFrame(rows).set_index("label")
        pat.to_csv(args.cv_dir / "oof_pooled_patient_metrics.csv")
        print("\n=== POOLED patient-level OOF metrics ===")
        cols = ["n_pos", "n_neg", "roc_auc", "pr_auc", "threshold",
                "f1", "sensitivity", "specificity", "balanced_accuracy"]
        cols = [c for c in cols if c in pat.columns]
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(pat[cols].to_string())
    else:
        print(f"[WARN] {pat_path} not found")

    # ---- Window level ----
    win_path = args.cv_dir / "cv_oof_window_predictions.csv.gz"
    if win_path.exists():
        with gzip.open(win_path, "rt", encoding="utf-8") as f:
            dfw = pd.read_csv(f)
        rows = []
        for label, g in dfw.groupby("label"):
            y = g["y_true"].to_numpy(dtype=int)
            s = g["proba"].to_numpy(dtype=float)
            ok = np.isfinite(s)
            thr = thr_map.get(label, 0.5)
            m = pooled_metrics(y[ok], s[ok], thr)
            m["label"] = label
            rows.append(m)
        win = pd.DataFrame(rows).set_index("label")
        win.to_csv(args.cv_dir / "oof_pooled_window_metrics.csv")
        print("\n=== POOLED window-level OOF metrics ===")
        cols = ["n_pos", "n_neg", "roc_auc", "pr_auc", "threshold",
                "f1", "sensitivity", "specificity", "balanced_accuracy"]
        cols = [c for c in cols if c in win.columns]
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(win[cols].to_string())
    else:
        print(f"[WARN] {win_path} not found")

    print(f"\n[DONE] Pooled metrics written to {args.cv_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
