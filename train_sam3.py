#!/usr/bin/env python3
"""
train_sam3.py
=============

Train per-label TabICLv2 classifiers from SAM 3 merged time-series, for the
eight target hyperkinetic phenomenologies. Behaviour mirrors cody-2's
`train.py` (same windowing, same 19 statistical descriptors, same negative
subsampling, same patient-grouped logic), but the *signal columns* come from
the SAM 3 segmentation pipeline and are selected by the user-chosen feature
tier:

    Tier 1: 14 geometric + 12 posture-derived signals       (~26 signals)
    Tier 2: Tier 1 + 40 contour/grid regional aggregates    (~66 signals)
    Tier 3: 14 geometric + 128 contour + 192 grid (raw)     (~334 signals)

Inputs
------
  --train_root    Directory containing merged SAM 3 XLSX files. Standard layout
                  (one of the dataset_* subfolders, as produced by
                  merge_sam3_labels.py):

                    train_root/
                    ├── dataset_lc/
                    ├── dataset_dd/
                    ├── dataset_action/
                    ├── dataset_rest/
                    ├── dataset_posture/
                    └── dataset_consensus/   # (optional)

                  The user picks which subfolders to include via --include_tags
                  (default: all except dataset_consensus, matching the cody-2
                  paper protocol).

  --out_dir       Where to write models / bundle metadata.

  --tier          1, 2 or 3 (default: 2).

Outputs
-------
  out_dir/
  ├── models/label_<SYM>/model.joblib       one TabICL classifier per symptom
  ├── bundle_meta.json                       feature schema + training config
  ├── features_windows_train__t<tier>__...csv.gz   cached feature matrix
  └── training_label_summary.csv             per-label sanity counts

Usage example
-------------
    python train_sam3.py \\
        --train_root /path/to/merged \\
        --out_dir    /path/to/runs/sam3_tier2 \\
        --tier 2 \\
        --robust_norm \\
        --use_cache
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

# We import the vendored cody-2 helpers from `cody2_utils.py`, which is part
# of this pipeline. They are an exact copy of the helpers in the cody-2
# paper repository (cody-pipeline/train.py), so SAM 3 results remain
# bit-comparable with the cody-2 protocol. No external dependency on the
# cody-2 repo is needed.

DEFAULT_SYMPTOMS = ["Dystonia", "Tremor", "Myoclonus", "Chorea",
                    "Athetosis", "Ballismus", "Stereotypies", "Tics"]
EXTRA_LABELS = ["Bradykinesia", "FOG", "Ataxia"]

DEFAULT_INCLUDE_TAGS = [
    "dataset_lc", "dataset_dd",
    "dataset_action", "dataset_rest", "dataset_posture",
]


def _import_cody2_helpers(cody2_root: Optional[Path] = None):
    """Import the vendored cody-2 helpers.

    The ``cody2_root`` argument is kept for backwards compatibility but is
    ignored: the helpers are now vendored in ``cody2_utils.py`` next to this
    file.
    """
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import cody2_utils as cody2_train  # noqa: E402
    return cody2_train


# ---------------------------------------------------------------------------
# Feature tier handling (SAM 3 specific)
# ---------------------------------------------------------------------------

def _import_sam3_features() -> "module":
    """Find sam3_features in the same dir as this script (or sys.path)."""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import sam3_features  # noqa: E402
    return sam3_features


# ---------------------------------------------------------------------------
# Context flag from path (mirrors cody-2)
# ---------------------------------------------------------------------------

def context_flags_from_tag(tag: str) -> Dict[str, int]:
    t = str(tag).lower()
    return {
        "ctx__is_dd":       int(t == "dataset_dd"),
        "ctx__is_lc":       int(t == "dataset_lc"),
        "ctx__is_action":   int(t == "dataset_action"),
        "ctx__is_rest":     int(t == "dataset_rest"),
        "ctx__is_posture":  int(t == "dataset_posture"),
        "ctx__is_short10s": int(t in ("dataset_action", "dataset_rest",
                                       "dataset_posture")),
        "ctx__is_consensus": int(t == "dataset_consensus"),
    }


def dataset_tag_from_path(xlsx_path: Path) -> str:
    """Find a 'dataset_*' parent folder name."""
    for p in xlsx_path.parents:
        nm = p.name.strip().lower()
        if re.match(r"^dataset_[a-z0-9_]+$", nm):
            return nm
    return "dataset_unknown"


# ---------------------------------------------------------------------------
# Per-file featurisation
# ---------------------------------------------------------------------------

def find_xlsx_files(root: Path, include_tags: List[str]) -> List[Path]:
    """Recursively find xlsx files under the requested tag subfolders.

    Supports both layouts:
      * per-video files produced by merge_sam3_labels.py (named
        ``<stem>_merged.xlsx``)
      * per-patient files produced by aggregate_by_patient.py (named
        ``<patient_id>.xlsx``, e.g. ``P1.xlsx``).

    Files whose names start with an underscore (``_aggregation_summary.csv``
    and similar) are ignored.
    """
    tags_lc = {t.lower() for t in include_tags}
    files: List[Path] = []
    for p in root.rglob("*.xlsx"):
        if "__MACOSX" in str(p) or p.name.startswith("~$") or p.name.startswith("_"):
            continue
        tag = dataset_tag_from_path(p)
        if tag in tags_lc:
            files.append(p)
    return sorted(files)


def patient_id_from_filename(xlsx_path: Path) -> str:
    """For SAM 3 merged files, we prefer the subject_id column (if present),
    falling back to the file stem. We do this in featurize_one_file using the
    merged xlsx itself.
    """
    return xlsx_path.stem.replace("_merged", "").strip().upper()


def featurize_one_file(
    xlsx_path: Path,
    cody2,                # the imported cody-2 train module
    sam3_features,        # the SAM 3 features module
    symptoms: List[str],
    fps: float,
    robust_norm: bool,
    tier: int,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Read one merged file, add tier-specific derived signals, then run
    cody-2's 19-descriptor extractor on each signal column.

    Returns (one_row_per_window_df, list_of_signal_columns, list_of_label_columns).
    """
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    if "From" not in df.columns or "To" not in df.columns:
        raise ValueError(f"{xlsx_path.name} missing 'From'/'To'.")

    # Derived signals according to the chosen tier
    df = sam3_features.add_derived_signals(df, tier=tier)

    df = df.copy()
    df["From"] = cody2.to_numeric_any(df["From"])
    df["To"] = cody2.to_numeric_any(df["To"])
    df = df.dropna(subset=["From", "To"]).copy()
    if len(df) == 0:
        raise ValueError(f"{xlsx_path.name}: all rows dropped after From/To parsing.")

    # Subject / patient id, in this priority order:
    #  1) __patient_id column (set by aggregate_by_patient.py: real patient ID)
    #  2) subject_id column from SAM 3 (e.g. "P1_RPA" → strip suffix)
    #  3) the file stem (last resort; less reliable for date-based names)
    if "__patient_id" in df.columns and df["__patient_id"].notna().any():
        pid = str(df["__patient_id"].dropna().iloc[0]).strip().upper()
    elif "subject_id" in df.columns and df["subject_id"].notna().any():
        raw = str(df["subject_id"].dropna().iloc[0]).strip()
        # Strip _RPA / _whatever suffix to recover the real patient ID
        m = re.match(r"^([PC])_?(\d+)", raw, re.IGNORECASE)
        pid = f"{m.group(1).upper()}{int(m.group(2))}" if m else raw.upper()
    else:
        pid = patient_id_from_filename(xlsx_path)
    df["patient_id"] = pid

    # Context flags (from the parent folder)
    tag = dataset_tag_from_path(xlsx_path)
    flags = context_flags_from_tag(tag)
    for k, v in flags.items():
        df[k] = int(v)

    # Label columns: keep only those that exist
    label_cols = [s for s in symptoms + EXTRA_LABELS if s in df.columns]
    if not label_cols:
        raise ValueError(f"{xlsx_path.name}: no label columns present.")

    # Signal columns: tier-selected
    sig_cols = sam3_features.get_tier_signal_cols(df, tier=tier)
    if not sig_cols:
        raise ValueError(f"{xlsx_path.name}: no signal columns for tier {tier}.")

    # Aggregate per (patient_id, From, To)
    gb = df.groupby(["patient_id", "From", "To"], sort=False)

    rows = []
    for (p, fr, to), block in gb:
        row = {"patient_id": p, "From": float(fr), "To": float(to)}

        # Aggregate labels: take the *max* of {0,1,2} within the window,
        # mirroring cody-2's behaviour (so any '1' wins, then '2', then '0').
        for lc in label_cols:
            v = pd.to_numeric(block[lc], errors="coerce").to_numpy(dtype=float)
            row[lc] = int(np.nanmax(v)) if np.any(~np.isnan(v)) else np.nan

        # 19 statistical descriptors per signal column
        for c in sig_cols:
            v = pd.to_numeric(block[c], errors="coerce").to_numpy(dtype=float)
            feats = cody2.compute_features_1d(v, fps=fps, robust_norm=robust_norm)
            for k, val in feats.items():
                row[f"f__{c}__{k}"] = val

        # Context flags (constants within a file -> copy once)
        for k in flags:
            row[k] = int(block[k].iloc[0])

        rows.append(row)

    out = pd.DataFrame(rows)
    out["patient_id"] = out["patient_id"].astype("string").str.strip().str.upper()
    return out, sig_cols, label_cols


def cache_filename(tier: int, fps: float, robust_norm: bool) -> str:
    fps_i = int(round(float(fps)))
    rn = 1 if robust_norm else 0
    return f"features_windows_train__t{tier}__fps{fps_i}__rn{rn}.csv.gz"


def save_csv_gz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)


def load_csv_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return pd.read_csv(
            f, low_memory=False,
            dtype={"patient_id": "string", "From": "Float64", "To": "Float64"},
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=Path, required=True,
                    help="Directory with merged dataset_*/<file>_merged.xlsx.")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--cody2_root", type=Path, default=None,
                    help="Deprecated: ignored. The cody-2 helpers are now "
                         "vendored in cody2_utils.py next to this script. "
                         "The flag is kept only for backwards compatibility.")
    ap.add_argument("--tier", type=int, default=2, choices=(1, 2, 3),
                    help="SAM 3 feature tier. 1=minimal, 2=regional, 3=raw.")
    ap.add_argument("--include_tags", type=str,
                    default=",".join(DEFAULT_INCLUDE_TAGS),
                    help="Comma-separated dataset tags to include in training "
                         "(default: lc, dd, action, rest, posture).")
    ap.add_argument("--symptoms", type=str, default=",".join(DEFAULT_SYMPTOMS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--robust_norm", action="store_true")
    ap.add_argument("--neg_pos_ratio", type=float, default=20.0)
    ap.add_argument("--min_pos", type=int, default=5)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--n_estimators", type=int, default=32)
    ap.add_argument("--softmax_temperature", type=float, default=0.8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--use_amp", type=str, default="auto")
    ap.add_argument("--kv_cache", action="store_true")
    ap.add_argument("--outlier_threshold", type=float, default=6.0)
    ap.add_argument("--average_logits", action="store_true")
    ap.add_argument("--use_cache", action="store_true")
    ap.add_argument("--force_rebuild_cache", action="store_true")
    args = ap.parse_args(argv)

    cody2 = _import_cody2_helpers(args.cody2_root)
    sam3_features = _import_sam3_features()
    from tabicl import TabICLClassifier

    symptoms = [s.strip() for s in args.symptoms.split(",") if s.strip()]
    include_tags = [t.strip().lower() for t in args.include_tags.split(",") if t.strip()]

    device = cody2.resolve_device(args.device)
    cody2.print_env(device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "models").mkdir(parents=True, exist_ok=True)

    cache_path = args.out_dir / cache_filename(
        args.tier, args.fps, bool(args.robust_norm),
    )
    use_cache = (args.use_cache and cache_path.exists()
                 and not args.force_rebuild_cache)

    if use_cache:
        print(f"[INFO] Loading cached window table: {cache_path}")
        dfw = load_csv_gz(cache_path)
        sig_cols, label_cols = [], [c for c in symptoms if c in dfw.columns]
    else:
        xlsx_files = find_xlsx_files(args.train_root, include_tags)
        if not xlsx_files:
            raise RuntimeError(
                f"No merged .xlsx files under {args.train_root} for tags "
                f"{include_tags}.")
        print(f"[INFO] Found {len(xlsx_files)} merged XLSX files under: "
              f"{args.train_root}")
        for tag in include_tags:
            n = sum(1 for p in xlsx_files if dataset_tag_from_path(p) == tag)
            print(f"   {tag}: {n} files")

        rows = []
        sig_cols, label_cols = None, None
        for i, p in enumerate(xlsx_files, 1):
            try:
                dfp, sig, labs = featurize_one_file(
                    p, cody2, sam3_features, symptoms,
                    fps=float(args.fps),
                    robust_norm=bool(args.robust_norm),
                    tier=int(args.tier),
                )
            except Exception as e:
                print(f"  [WARN] {p.name}: {e}")
                continue
            rows.append(dfp)
            sig_cols = sig_cols or sig
            label_cols = label_cols or labs
            if i % 5 == 0 or i == len(xlsx_files):
                print(f"  [{i:3d}/{len(xlsx_files)}] featurized {p.name}")

        if not rows:
            raise RuntimeError("No file could be featurized.")
        dfw = pd.concat(rows, axis=0, ignore_index=True)
        save_csv_gz(dfw, cache_path)
        print(f"[OK] Cache: {cache_path} "
              f"windows={len(dfw)} patients={dfw['patient_id'].nunique()}")

    feat_cols = [c for c in dfw.columns if str(c).startswith("f__")]
    ctx_cols = [c for c in dfw.columns if str(c).startswith("ctx__")]
    feat_cols_with_ctx = feat_cols + ctx_cols
    if not feat_cols:
        raise RuntimeError("No 'f__'-prefixed feature columns found.")

    available_labels = [s for s in symptoms if s in dfw.columns]
    if not available_labels:
        raise RuntimeError(f"No label columns among {symptoms}.")

    # Per-label sanity summary
    summary = cody2.make_label_summary(dfw, available_labels)
    summary_path = args.out_dir / "training_label_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Label summary: {summary_path}")

    # Bundle metadata
    meta = {
        "pipeline": "sam3",
        "tier": int(args.tier),
        "symptoms": available_labels,
        "feature_columns": feat_cols_with_ctx,
        "fps": float(args.fps),
        "robust_norm": bool(args.robust_norm),
        "include_tags": include_tags,
        "cache_file": cache_path.name,
        "sampling": {"neg_pos_ratio": float(args.neg_pos_ratio),
                     "min_pos": int(args.min_pos)},
        "tabicl": {
            "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
            "n_estimators": int(args.n_estimators),
            "softmax_temperature": float(args.softmax_temperature),
            "batch_size": int(args.batch_size),
            "use_amp": str(args.use_amp),
            "device": str(device),
            "kv_cache": bool(args.kv_cache),
            "outlier_threshold": float(args.outlier_threshold),
            "average_logits": bool(args.average_logits),
        },
    }
    (args.out_dir / "bundle_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] Bundle meta: {args.out_dir / 'bundle_meta.json'}")

    # Train one classifier per label
    X_all = dfw[feat_cols_with_ctx].copy()
    pid_all = dfw["patient_id"].astype(str).to_numpy()

    for sym in available_labels:
        y_raw = pd.to_numeric(dfw[sym], errors="coerce").to_numpy(dtype=float)
        yb, known = cody2.label_to_known_binary(y_raw)
        if int(np.sum(known)) == 0:
            print(f"[SKIP] {sym}: no known labels.")
            continue
        X = X_all.loc[known].reset_index(drop=True)
        y = yb[known].astype(int)
        pid = pid_all[known]

        n_pos = int(np.sum(y == 1)); n_neg = int(np.sum(y == 0))
        n_unk = int(np.sum(~known))
        print(f"\n[LABEL] {sym} known={int(np.sum(known))} "
              f"pos={n_pos} neg={n_neg} unknown_dropped={n_unk}")

        if n_pos < int(args.min_pos):
            print(f"  [SKIP] too few positives (min_pos={args.min_pos}).")
            continue
        if n_neg == 0:
            print(f"  [SKIP] no negatives.")
            continue

        keep = cody2.sample_rows_ratio_keep_controls(
            y, pid, float(args.neg_pos_ratio), seed=int(args.seed)
        )
        Xs = X.iloc[keep].reset_index(drop=True)
        ys = y[keep]

        clf_kwargs = dict(
            checkpoint_version=meta["tabicl"]["checkpoint_version"],
            n_estimators=int(args.n_estimators),
            softmax_temperature=float(args.softmax_temperature),
            batch_size=int(args.batch_size),
            use_amp=str(args.use_amp),
            device=str(device),
            random_state=int(args.seed),
            outlier_threshold=float(args.outlier_threshold),
        )
        if bool(args.average_logits):
            clf_kwargs["average_logits"] = True

        clf = TabICLClassifier(**clf_kwargs)
        try:
            clf.fit(Xs, ys, kv_cache=bool(args.kv_cache))
        except TypeError:
            clf.fit(Xs, ys)

        label_dir = args.out_dir / "models" / f"label_{sym}"
        label_dir.mkdir(parents=True, exist_ok=True)
        try:
            clf.save(label_dir / "model.joblib")
        except Exception:
            import joblib
            joblib.dump(clf, label_dir / "model.joblib")
        print(f"  [OK] saved: {label_dir / 'model.joblib'}")

    print("\n[DONE] Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
