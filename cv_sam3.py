#!/usr/bin/env python3
"""
cv_sam3.py (v2 — resumable)
===========================

Patient-grouped cross-validation for SAM 3 / TabICLv2 models.

This version saves predictions **incrementally**: after every label finishes,
the OOF tables are flushed to disk. If the process is interrupted (Windows
reboot, power loss, Ctrl+C), the work that was completed is preserved and
can be downstreamed to ``cv_aggregate_oof.py`` and ``tune_threshold.py``.

Resume semantics:
    On startup, the script checks whether ``cv_oof_patient_predictions.csv``
    already contains rows for a given (label) and skips it. Use
    ``--force_restart`` to re-run from scratch.

Inputs
------
  --bundle_dir   Output directory of a previous train_sam3 run.
  --out_dir      Where to write the CV outputs (default: bundle_dir/cv_Nfold).
  --n_splits     Number of folds (default 5; use 25 for LOPO).
  --patient_agg  Aggregation method for the patient-level score (default p95).
  --threshold    Decision threshold at the window level (default 0.5).
  --device       cuda / cpu (default: auto).
  --n_estimators Override TabICL's n_estimators (default: from bundle_meta).
  --predict_batch_size  Lower if VRAM-limited (default: 256).
  --force_restart  Ignore any existing OOF files and start fresh.

Outputs (in --out_dir)
----------------------
  cv_window_metrics.csv         per (fold, label) window-level metrics
  cv_patient_metrics.csv        per (fold, label) patient-level metrics
  cv_oof_window_predictions.csv.gz  per-window OOF probas
  cv_oof_patient_predictions.csv    per-patient OOF predictions
  cv_summary.csv                aggregated mean +- SD per label (at end)
  cv_config.json                what was run (at end)
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _import_helpers():
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import cody2_utils as cody2  # noqa: E402
    return cody2


def load_cache(bundle_dir: Path) -> pd.DataFrame:
    cache_paths = sorted(bundle_dir.glob("features_windows_train__t*.csv.gz"))
    if not cache_paths:
        raise FileNotFoundError(
            f"No cached feature table found under {bundle_dir}. "
            f"Run train_sam3.py first."
        )
    cache_path = cache_paths[0]
    print(f"[INFO] Loading cache: {cache_path}")
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        dfw = pd.read_csv(
            f, low_memory=False,
            dtype={"patient_id": "string",
                   "From": "Float64", "To": "Float64"},
        )
    print(f"[INFO] Loaded: shape={dfw.shape}, "
          f"patients={dfw['patient_id'].nunique()}")
    return dfw


def agg_patient_score(probs: np.ndarray, method: str) -> float:
    if probs.size == 0:
        return float("nan")
    m = str(method).lower()
    if m == "max": return float(np.max(probs))
    if m == "mean": return float(np.mean(probs))
    if m == "median": return float(np.median(probs))
    if m == "noisy_or": return float(1.0 - np.prod(1.0 - probs))
    if m.startswith("p"):
        try: return float(np.percentile(probs, float(m[1:])))
        except ValueError: pass
    raise ValueError(f"Unknown aggregation method: {method}")


def compute_window_metrics(y_true, y_proba, threshold):
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, accuracy_score, f1_score,
        confusion_matrix,
    )
    out = {"n_total": int(len(y_true)),
           "n_pos": int(np.sum(y_true == 1)),
           "n_neg": int(np.sum(y_true == 0))}
    if out["n_pos"] == 0 or out["n_neg"] == 0:
        out.update({"roc_auc": np.nan, "pr_auc": np.nan,
                    "accuracy": np.nan, "f1": np.nan,
                    "sensitivity": np.nan, "specificity": np.nan})
        return out
    out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    y_pred = (y_proba >= threshold).astype(int)
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    return out


def compute_patient_metrics(patient_true, patient_score, threshold):
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score, confusion_matrix,
    )
    out = {
        "n_patients": int(len(patient_true)),
        "n_pos_patients": int(np.sum(patient_true == 1)),
        "n_neg_patients": int(np.sum(patient_true == 0)),
    }
    if out["n_pos_patients"] == 0 or out["n_neg_patients"] == 0:
        out.update({"roc_auc": np.nan, "accuracy": np.nan, "f1": np.nan,
                    "sensitivity": np.nan, "specificity": np.nan})
        return out
    out["roc_auc"] = float(roc_auc_score(patient_true, patient_score))
    y_pred = (patient_score >= threshold).astype(int)
    out["accuracy"] = float(accuracy_score(patient_true, y_pred))
    out["f1"] = float(f1_score(patient_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(
        patient_true, y_pred, labels=[0, 1]
    ).ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    return out


# ----------------------------------------------------------------------------
# Incremental save helpers
# ----------------------------------------------------------------------------

def append_csv(rows: List[dict], path: Path) -> None:
    """Append rows to a CSV, writing a header only if the file does not yet
    exist. Atomic-ish: writes to a temp file then renames."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    # Append mode (mode='a') with header only on first write
    df.to_csv(path, mode="a", header=header, index=False)


def append_csv_gz(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    with gzip.open(path, "at", encoding="utf-8", newline="") as f:
        df.to_csv(f, header=header, index=False)


def load_already_done_labels(out_dir: Path) -> set:
    """Inspect existing OOF patient predictions to know which labels are done."""
    p = out_dir / "cv_oof_patient_predictions.csv"
    if not p.exists():
        return set()
    try:
        df = pd.read_csv(p, usecols=["label"])
        return set(df["label"].astype(str).unique())
    except Exception:
        return set()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--patient_agg", type=str, default="p95")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--symptoms", type=str, default=None)
    ap.add_argument("--min_pos", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=None)
    ap.add_argument("--predict_batch_size", type=int, default=256)
    ap.add_argument("--force_restart", action="store_true",
                    help="Ignore any existing OOF files and start fresh.")
    args = ap.parse_args(argv)

    cody2 = _import_helpers()
    from sklearn.model_selection import GroupKFold
    from tabicl import TabICLClassifier

    bundle_meta_path = args.bundle_dir / "bundle_meta.json"
    if not bundle_meta_path.exists():
        raise FileNotFoundError(bundle_meta_path)
    meta = json.loads(bundle_meta_path.read_text())

    out_dir = args.out_dir or (args.bundle_dir / f"cv_{args.n_splits}fold")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume logic
    if args.force_restart:
        for fn in ("cv_window_metrics.csv", "cv_patient_metrics.csv",
                   "cv_oof_window_predictions.csv.gz",
                   "cv_oof_patient_predictions.csv",
                   "cv_summary.csv", "cv_config.json"):
            p = out_dir / fn
            if p.exists():
                p.unlink()
        print("[INFO] --force_restart: existing OOF files removed.")
    already = load_already_done_labels(out_dir)
    if already:
        print(f"[INFO] Resume mode: already completed labels = "
              f"{sorted(already)}; they will be SKIPPED. Use --force_restart "
              f"to redo them.")

    symptoms = (args.symptoms.split(",") if args.symptoms
                else list(meta["symptoms"]))
    feat_cols_with_ctx = list(meta["feature_columns"])
    tabicl_cfg = meta.get("tabicl", {})

    device = cody2.resolve_device(args.device)
    cody2.print_env(device)

    dfw = load_cache(args.bundle_dir)
    missing = [c for c in feat_cols_with_ctx if c not in dfw.columns]
    if missing:
        raise RuntimeError(
            f"{len(missing)} feature columns missing from cache: "
            f"{missing[:5]} ..."
        )

    X_all = dfw[feat_cols_with_ctx].fillna(0).to_numpy(dtype=float)
    pids_all = dfw["patient_id"].astype(str).to_numpy()
    unique_patients = sorted(set(pids_all))
    print(f"[INFO] {len(unique_patients)} patients: {unique_patients}")

    n_splits = min(int(args.n_splits), len(unique_patients))
    if n_splits != args.n_splits:
        print(f"[WARN] Reducing n_splits to {n_splits} "
              f"(only {len(unique_patients)} patients).")

    for sym in symptoms:
        if sym in already:
            print(f"[SKIP] {sym}: already in OOF file.")
            continue
        if sym not in dfw.columns:
            print(f"[SKIP] {sym}: not in cache columns")
            continue
        y_raw = pd.to_numeric(dfw[sym], errors="coerce").to_numpy(dtype=float)
        yb, known = cody2.label_to_known_binary(y_raw)
        n_known = int(np.sum(known))
        n_pos = int(np.sum(yb[known] == 1))
        n_neg = int(np.sum(yb[known] == 0))

        if n_pos < int(args.min_pos):
            print(f"[SKIP] {sym}: only {n_pos} positive windows "
                  f"(< min_pos={args.min_pos})")
            continue
        if n_neg == 0:
            print(f"[SKIP] {sym}: no negatives"); continue

        print(f"\n[LABEL] {sym} known={n_known} pos={n_pos} neg={n_neg}")

        X_lab = X_all[known]
        y_lab = yb[known].astype(int)
        pids_lab = pids_all[known]

        patient_ids_sorted = sorted(set(pids_lab))
        pat_truth = {pid: int(np.any(y_lab[pids_lab == pid] == 1))
                     for pid in patient_ids_sorted}

        # Per-label accumulators (flushed at the end of the label)
        rows_window: List[dict] = []
        rows_patient: List[dict] = []
        rows_oof_window: List[dict] = []
        rows_oof_patient: List[dict] = []

        gkf = GroupKFold(n_splits=n_splits)
        fold_idx = 0
        for tr, te in gkf.split(X_lab, y_lab, groups=pids_lab):
            fold_idx += 1
            te_pids = sorted(set(pids_lab[te]))

            n_pos_tr = int(np.sum(y_lab[tr] == 1))
            n_pos_te = int(np.sum(y_lab[te] == 1))
            n_neg_tr = int(np.sum(y_lab[tr] == 0))
            n_neg_te = int(np.sum(y_lab[te] == 0))

            if n_pos_tr == 0 or n_neg_tr == 0:
                print(f"  [SKIP fold {fold_idx}] degenerate train: "
                      f"pos={n_pos_tr} neg={n_neg_tr}")
                continue
            if n_pos_te == 0 and n_neg_te == 0:
                continue

            keep_tr = cody2.sample_rows_ratio_keep_controls(
                y_lab[tr], pids_lab[tr],
                float(meta.get("sampling", {}).get("neg_pos_ratio", 20.0)),
                seed=int(args.seed) + fold_idx,
            )
            X_tr = X_lab[tr][keep_tr]
            y_tr = y_lab[tr][keep_tr]

            clf_kwargs = dict(
                checkpoint_version=tabicl_cfg.get(
                    "checkpoint_version", "tabicl-classifier-v2-20260212.ckpt"
                ),
                n_estimators=int(
                    args.n_estimators
                    if args.n_estimators is not None
                    else tabicl_cfg.get("n_estimators", 32)
                ),
                softmax_temperature=float(tabicl_cfg.get(
                    "softmax_temperature", 0.8)),
                batch_size=int(tabicl_cfg.get("batch_size", 64)),
                use_amp=str(tabicl_cfg.get("use_amp", "auto")),
                device=str(device),
                random_state=int(args.seed),
                outlier_threshold=float(tabicl_cfg.get(
                    "outlier_threshold", 6.0)),
            )
            if tabicl_cfg.get("average_logits", False):
                clf_kwargs["average_logits"] = True

            clf = TabICLClassifier(**clf_kwargs)
            try:
                clf.fit(X_tr, y_tr,
                        kv_cache=bool(tabicl_cfg.get("kv_cache", False)))
            except TypeError:
                clf.fit(X_tr, y_tr)

            X_test = X_lab[te]
            n_test = X_test.shape[0]
            bs = int(args.predict_batch_size)
            proba_chunks = []
            for start in range(0, n_test, bs):
                end = min(start + bs, n_test)
                p = clf.predict_proba(X_test[start:end])[:, 1]
                proba_chunks.append(p)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            proba_te = (np.concatenate(proba_chunks) if proba_chunks
                        else np.array([]))

            wm = compute_window_metrics(y_lab[te], proba_te,
                                        float(args.threshold))
            wm.update({"fold": fold_idx, "label": sym,
                       "n_train_after_subsampling": int(len(y_tr)),
                       "test_patients": ",".join(te_pids)})
            rows_window.append(wm)

            pat_score, pat_y = [], []
            for pid in te_pids:
                mask = (pids_lab[te] == pid)
                probs = proba_te[mask]
                s = agg_patient_score(probs, args.patient_agg)
                pat_score.append(s); pat_y.append(pat_truth[pid])
                rows_oof_patient.append({
                    "label": sym, "fold": fold_idx,
                    "patient_id": pid,
                    "y_true": int(pat_truth[pid]),
                    "score": float(s),
                    "y_pred": int(s >= float(args.threshold))
                              if np.isfinite(s) else -1,
                    "n_windows": int(np.sum(mask)),
                })
            pat_score = np.asarray(pat_score, dtype=float)
            pat_y = np.asarray(pat_y, dtype=int)
            pm = compute_patient_metrics(pat_y, pat_score,
                                         float(args.threshold))
            pm.update({"fold": fold_idx, "label": sym,
                       "test_patients": ",".join(te_pids)})
            rows_patient.append(pm)

            for idx, (i, p, pid_v) in enumerate(zip(te, proba_te,
                                                    pids_lab[te])):
                rows_oof_window.append({
                    "label": sym, "fold": fold_idx,
                    "patient_id": pid_v,
                    "y_true": int(y_lab[te][idx]),
                    "proba": float(p),
                })

            print(f"  fold {fold_idx}: "
                  f"test_patients={len(te_pids)}, "
                  f"window ROC={wm.get('roc_auc', np.nan):.3f}, "
                  f"patient ROC={pm.get('roc_auc', np.nan):.3f}")

            del clf
            try:
                import torch, gc
                gc.collect(); torch.cuda.empty_cache()
            except Exception:
                pass

        # ---- Flush per-label accumulators to disk ----
        append_csv(rows_window, out_dir / "cv_window_metrics.csv")
        append_csv(rows_patient, out_dir / "cv_patient_metrics.csv")
        append_csv(rows_oof_patient,
                   out_dir / "cv_oof_patient_predictions.csv")
        append_csv_gz(rows_oof_window,
                      out_dir / "cv_oof_window_predictions.csv.gz")
        print(f"  [FLUSHED] {sym}: {len(rows_oof_patient)} patient rows, "
              f"{len(rows_oof_window)} window rows.")

    # ---- Build summary at end (best-effort: works even if some labels
    # were skipped or resumed) ----
    try:
        df_w = pd.read_csv(out_dir / "cv_window_metrics.csv")
        df_p = pd.read_csv(out_dir / "cv_patient_metrics.csv")

        def _summarize(df, level: str):
            metric_cols = [c for c in df.columns
                           if c in ("roc_auc", "pr_auc", "accuracy", "f1",
                                    "sensitivity", "specificity")]
            rows = []
            for lab, g in df.groupby("label"):
                row = {"label": lab, "level": level,
                       "n_folds": int(len(g))}
                for m in metric_cols:
                    vals = g[m].dropna().to_numpy(dtype=float)
                    if vals.size:
                        row[f"{m}_mean"] = float(np.mean(vals))
                        row[f"{m}_sd"] = (float(np.std(vals, ddof=1))
                                          if vals.size > 1 else 0.0)
                    else:
                        row[f"{m}_mean"] = np.nan
                        row[f"{m}_sd"] = np.nan
                rows.append(row)
            return pd.DataFrame(rows)

        summary = pd.concat([_summarize(df_w, "window"),
                             _summarize(df_p, "patient")], ignore_index=True)
        summary.to_csv(out_dir / "cv_summary.csv", index=False)

        print("\n=== CV summary (mean +- SD across folds) ===")
        for level in ("window", "patient"):
            sub = summary[summary.level == level].sort_values("label")
            if sub.empty: continue
            print(f"\n[{level}-level]")
            cols = ["roc_auc_mean", "roc_auc_sd",
                    "f1_mean", "f1_sd",
                    "sensitivity_mean", "sensitivity_sd",
                    "specificity_mean", "specificity_sd"]
            cols = [c for c in cols if c in sub.columns]
            sub2 = sub[["label"] + cols].copy().set_index("label")
            with pd.option_context("display.float_format", "{:.3f}".format):
                print(sub2.to_string())
    except Exception as e:
        print(f"[WARN] Could not build summary: {e}")

    cfg = {
        "bundle_dir": str(args.bundle_dir),
        "n_splits": n_splits,
        "patient_agg": args.patient_agg,
        "threshold": float(args.threshold),
        "device": device,
        "min_pos": int(args.min_pos),
        "seed": int(args.seed),
        "tier": int(meta.get("tier", -1)),
    }
    (out_dir / "cv_config.json").write_text(json.dumps(cfg, indent=2))

    print(f"\n[DONE] CV outputs saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
