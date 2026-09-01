#!/usr/bin/env python3
"""
evaluate_external.py
====================

External-cohort evaluation of CODY-SAM3 model predictions against multi-rater
clinical ground truth.

This is the final step of the inference pipeline. After ``inference_sam3.py``
has produced per-patient predictions for an external dataset, this script
compares those predictions to the clinicians' annotations stored in
``dataset_inference.xlsx`` and reports model performance at several levels of
inter-rater consensus, together with the inter-rater agreement itself (the
human "ceiling").

Why several consensus levels?
-----------------------------
Hyperkinetic-movement-disorder phenomenology is judged by expert gestalt and
raters do not always agree. There is therefore no single unambiguous ground
truth. We evaluate against three increasingly strict definitions of a positive
case, plus an "agreement-restricted" view:

  - "any"        : positive if >= 1 rater scored the phenomenology present.
  - "majority"   : positive if > half of raters scored it present
                   (for 2 raters this requires both -> equivalent to unanimous).
  - "unanimous"  : positive only if ALL raters scored it present.
  - "restricted" : keep only patients where raters were UNANIMOUS (all-0 or
                   all-1) and evaluate the model on those unambiguous cases.
                   This mirrors the cody-2 "restricted/masked" protocol.

We also report inter-rater agreement as a reference: Fleiss' kappa when there
are >= 3 raters, Cohen's kappa for exactly 2 raters. The key paper message is
that, for well-captured phenomenologies, model-vs-consensus agreement
approaches inter-rater agreement.

Ground-truth file layout (``dataset_inference.xlsx``)
-----------------------------------------------------
One sheet per dataset (``dataset_1`` / ``dataset_2`` / ``dataset_3``). Two
header rows: row 0 holds rater initials at the start of each 8-column block,
row 1 holds the phenomenology names. Column 0 is the dataset tag (first data
row only), column 1 is the patient id. Each rater contributes a block of 8
binary columns (the 8 phenomenologies).

Inputs
------
  --pred_csv      ``inference_patient_predictions.csv`` produced by
                  inference_sam3.py for ONE dataset (has columns
                  ``patient_id``, ``score__<Sym>``, ``pred__<Sym>``, ...).
  --gt_xlsx       Path to dataset_inference.xlsx.
  --sheet         Which sheet to read (e.g. "dataset_2"). Defaults to the value
                  of --dataset_tag if that matches a sheet name.
  --dataset_tag   Label used in the output (e.g. "dataset_2").
  --out_dir       Where to write the evaluation tables and the figure.

Outputs (written into --out_dir)
--------------------------------
  external_eval_<tag>__per_consensus.csv   metrics per phenomenology x consensus
  external_eval_<tag>__interrater.csv      inter-rater kappa per phenomenology
  external_eval_<tag>__confusion.csv       TP/FP/FN/TN per phenomenology x consensus
  external_eval_<tag>__merged.csv          patient-level model preds + each rater
  external_eval_<tag>__summary.png         ROC / agreement figure

Usage
-----
    python evaluate_external.py \
        --pred_csv runs/sam3_tier2/infer_dataset_2/reports/tables/inference_patient_predictions.csv \
        --gt_xlsx  dataset_inference.xlsx \
        --sheet    dataset_2 \
        --dataset_tag dataset_2 \
        --out_dir  runs/sam3_tier2/infer_dataset_2/reports/eval
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PHENOMENOLOGIES = ["Dystonia", "Tremor", "Myoclonus", "Chorea",
                   "Athetosis", "Ballismus", "Stereotypies", "Tics"]


# ---------------------------------------------------------------------------
# Ground-truth parsing
# ---------------------------------------------------------------------------

def parse_ground_truth(xlsx_path: Path, sheet: str) -> pd.DataFrame:
    """Parse one sheet of dataset_inference.xlsx into a long tidy frame.

    Returns
    -------
    DataFrame with columns:
        patient_id, rater, <one column per phenomenology with 0/1 values>
    one row per (patient, rater).
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)

    # Row 0: rater initials at the start of each 8-col block (rest NaN).
    # Row 1: phenomenology names, repeated per rater block.
    # Data starts at row 2. Col 0 = dataset tag (sparse), col 1 = patient id.
    rater_row = raw.iloc[0].tolist()
    phenom_row = raw.iloc[1].tolist()

    # Identify rater blocks: a "block" starts at each column where rater_row is
    # a non-empty string. Each block is assumed to span 8 columns (the 8
    # phenomenologies, in the order given by phenom_row).
    block_starts = [j for j, v in enumerate(rater_row)
                    if isinstance(v, str) and v.strip() and v.strip().lower() != "nan"]

    records = []
    data = raw.iloc[2:].reset_index(drop=True)
    # Patient id is column 1
    patient_ids = data.iloc[:, 1].astype(str).str.strip()

    for bstart in block_starts:
        rater = str(rater_row[bstart]).strip()
        # The 8 phenomenology columns for this rater
        for off in range(8):
            col = bstart + off
            if col >= raw.shape[1]:
                break
            phenom = str(phenom_row[col]).strip()
            if phenom not in PHENOMENOLOGIES:
                continue
            for i, pid in enumerate(patient_ids):
                if pid in ("", "nan", "None"):
                    continue
                val = data.iloc[i, col]
                records.append({
                    "patient_id": pid,
                    "rater": rater,
                    "phenomenology": phenom,
                    "value": _to01(val),
                })

    long = pd.DataFrame(records)
    if long.empty:
        raise RuntimeError(f"No ground-truth records parsed from sheet {sheet!r}")

    # Pivot to one row per (patient, rater), columns = phenomenologies
    wide = long.pivot_table(index=["patient_id", "rater"],
                            columns="phenomenology", values="value",
                            aggfunc="first").reset_index()
    wide.columns.name = None
    # Ensure all phenomenology columns exist
    for ph in PHENOMENOLOGIES:
        if ph not in wide.columns:
            wide[ph] = np.nan
    return wide[["patient_id", "rater"] + PHENOMENOLOGIES]


def _to01(v) -> float:
    """Coerce a cell to 0/1; NaN if not interpretable."""
    if pd.isna(v):
        return np.nan
    try:
        f = float(str(v).replace(",", ".").strip())
    except ValueError:
        return np.nan
    if f == 1:
        return 1.0
    if f == 0:
        return 0.0
    # Some sheets may use 2 for "uncertain": treat as NaN (unknown)
    return np.nan


# ---------------------------------------------------------------------------
# Consensus construction
# ---------------------------------------------------------------------------

def build_consensus(gt_wide: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """From per-(patient,rater) labels, build patient-level consensus truths.

    Returns a dict {consensus_name: DataFrame[patient_id, <phenomenology>...]}
    for consensus_name in {"any","majority","unanimous"} plus a special
    "restricted_mask" frame that is 1 where ALL raters agreed (unambiguous),
    used to subset patients in the 'restricted' evaluation.
    """
    raters = sorted(gt_wide["rater"].unique())
    n_raters = len(raters)
    patients = sorted(gt_wide["patient_id"].unique())

    any_rows, maj_rows, unan_rows, agree_rows = [], [], [], []
    for pid in patients:
        sub = gt_wide[gt_wide["patient_id"] == pid]
        any_r = {"patient_id": pid}
        maj_r = {"patient_id": pid}
        unan_r = {"patient_id": pid}
        agree_r = {"patient_id": pid}
        for ph in PHENOMENOLOGIES:
            vals = pd.to_numeric(sub[ph], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                any_r[ph] = maj_r[ph] = unan_r[ph] = np.nan
                agree_r[ph] = 0
                continue
            n_pos = int((vals == 1).sum())
            n = len(vals)
            any_r[ph] = int(n_pos >= 1)
            maj_r[ph] = int(n_pos > n / 2.0)
            unan_r[ph] = int(n_pos == n)
            # raters unanimous (all 0 or all 1)?
            agree_r[ph] = int(n_pos == 0 or n_pos == n)
        any_rows.append(any_r)
        maj_rows.append(maj_r)
        unan_rows.append(unan_r)
        agree_rows.append(agree_r)

    return {
        "any": pd.DataFrame(any_rows),
        "majority": pd.DataFrame(maj_rows),
        "unanimous": pd.DataFrame(unan_rows),
        "restricted_mask": pd.DataFrame(agree_rows),
        "_meta": pd.DataFrame({"n_raters": [n_raters], "raters": [";".join(raters)]}),
    }


# ---------------------------------------------------------------------------
# Inter-rater agreement
# ---------------------------------------------------------------------------

def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' kappa for a (n_items x n_categories) count matrix."""
    n_items, n_cat = table.shape
    n_raters = table.sum(axis=1)
    if not np.all(n_raters == n_raters[0]) or n_raters[0] < 2:
        # Unequal rater counts per item -> fall back to NaN
        return np.nan
    n = n_raters[0]
    p_j = table.sum(axis=0) / (n_items * n)
    P_i = (np.sum(table ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = np.sum(p_j ** 2)
    if np.isclose(1 - P_e, 0):
        return np.nan
    return float((P_bar - P_e) / (1 - P_e))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two 0/1 raters."""
    from sklearn.metrics import cohen_kappa_score
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    aa, bb = a[mask].astype(int), b[mask].astype(int)
    if len(np.unique(np.concatenate([aa, bb]))) < 2:
        # Degenerate (all same) -> perfect agreement by convention
        return 1.0 if np.array_equal(aa, bb) else 0.0
    return float(cohen_kappa_score(aa, bb))


def interrater_agreement(gt_wide: pd.DataFrame) -> pd.DataFrame:
    """Per-phenomenology inter-rater agreement.

    Fleiss' kappa if >= 3 raters; mean pairwise Cohen's kappa if exactly 2.
    """
    raters = sorted(gt_wide["rater"].unique())
    n_raters = len(raters)
    rows = []
    for ph in PHENOMENOLOGIES:
        piv = gt_wide.pivot_table(index="patient_id", columns="rater",
                                  values=ph, aggfunc="first")
        piv = piv.dropna(how="any")  # keep patients rated by everyone
        if piv.shape[0] < 2:
            rows.append({"phenomenology": ph, "n_raters": n_raters,
                         "method": "n/a", "kappa": np.nan, "n_patients": int(piv.shape[0])})
            continue
        if n_raters >= 3:
            # Build count matrix (n_items x 2 categories)
            counts = np.zeros((piv.shape[0], 2), dtype=int)
            for i, (_, r) in enumerate(piv.iterrows()):
                vals = r.to_numpy()
                counts[i, 0] = int((vals == 0).sum())
                counts[i, 1] = int((vals == 1).sum())
            k = fleiss_kappa(counts)
            method = "fleiss"
        else:
            # exactly 2 raters
            cols = piv.columns.tolist()
            k = cohen_kappa(piv[cols[0]].to_numpy(), piv[cols[1]].to_numpy())
            method = "cohen"
        rows.append({"phenomenology": ph, "n_raters": n_raters,
                     "method": method, "kappa": k, "n_patients": int(piv.shape[0])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model-vs-consensus metrics
# ---------------------------------------------------------------------------

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   score: Optional[np.ndarray] = None) -> Dict[str, float]:
    from sklearn.metrics import confusion_matrix, roc_auc_score
    out = {"n": int(len(y_true)),
           "n_pos": int(np.sum(y_true == 1)),
           "n_neg": int(np.sum(y_true == 0))}
    if out["n_pos"] == 0 or out["n_neg"] == 0:
        out.update({"roc_auc": np.nan, "f1": np.nan, "sensitivity": np.nan,
                    "specificity": np.nan, "balanced_accuracy": np.nan,
                    "accuracy": np.nan, "tp": np.nan, "fp": np.nan,
                    "fn": np.nan, "tn": np.nan})
        # accuracy is still defined
        if len(y_true):
            out["accuracy"] = float(np.mean(y_true == y_pred))
        return out
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (2 * ppv * sens / (ppv + sens)
          if (np.isfinite(ppv) and np.isfinite(sens) and (ppv + sens) > 0) else 0.0)
    out.update({
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "sensitivity": float(sens) if np.isfinite(sens) else np.nan,
        "specificity": float(spec) if np.isfinite(spec) else np.nan,
        "ppv": float(ppv) if np.isfinite(ppv) else np.nan,
        "f1": float(f1),
        "balanced_accuracy": (float((sens + spec) / 2)
                              if np.isfinite(sens) and np.isfinite(spec) else np.nan),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    })
    if score is not None:
        m = np.isfinite(score)
        if m.sum() and len(np.unique(y_true[m])) == 2:
            out["roc_auc"] = float(roc_auc_score(y_true[m], score[m]))
        else:
            out["roc_auc"] = np.nan
    return out


def evaluate(pred_df: pd.DataFrame,
             consensus: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute metrics per phenomenology x consensus level.

    Returns (metrics_df, confusion_df).
    """
    metric_rows = []
    confusion_rows = []
    restricted_mask = consensus["restricted_mask"].set_index("patient_id")

    for cons_name in ["any", "majority", "unanimous", "restricted"]:
        if cons_name == "restricted":
            truth = consensus["unanimous"].set_index("patient_id")  # value where agreed
        else:
            truth = consensus[cons_name].set_index("patient_id")

        for ph in PHENOMENOLOGIES:
            pred_col = f"pred__{ph}"
            score_col = f"score__{ph}"
            if pred_col not in pred_df.columns:
                continue
            merged = pred_df[["patient_id", pred_col, score_col]].copy()
            merged = merged.merge(truth[[ph]].rename(columns={ph: "y_true"}),
                                  on="patient_id", how="inner")
            if cons_name == "restricted":
                # keep only patients where raters agreed for this phenomenology
                agreed = restricted_mask[ph].reindex(merged["patient_id"]).to_numpy()
                merged = merged[agreed == 1]

            y_true = pd.to_numeric(merged["y_true"], errors="coerce").to_numpy()
            y_pred = pd.to_numeric(merged[pred_col], errors="coerce").to_numpy()
            score = pd.to_numeric(merged[score_col], errors="coerce").to_numpy()
            ok = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true, y_pred, score = y_true[ok].astype(int), y_pred[ok].astype(int), score[ok]

            m = binary_metrics(y_true, y_pred, score)
            m.update({"phenomenology": ph, "consensus": cons_name})
            metric_rows.append(m)
            confusion_rows.append({
                "phenomenology": ph, "consensus": cons_name,
                "tp": m.get("tp"), "fp": m.get("fp"),
                "fn": m.get("fn"), "tn": m.get("tn"),
                "n": m["n"], "n_pos": m["n_pos"], "n_neg": m["n_neg"],
            })

    metrics_df = pd.DataFrame(metric_rows)
    # tidy column order
    front = ["phenomenology", "consensus", "n", "n_pos", "n_neg",
             "roc_auc", "f1", "sensitivity", "specificity",
             "balanced_accuracy", "accuracy", "ppv"]
    front = [c for c in front if c in metrics_df.columns]
    rest = [c for c in metrics_df.columns if c not in front]
    metrics_df = metrics_df[front + rest]
    return metrics_df, pd.DataFrame(confusion_rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(metrics_df: pd.DataFrame, interrater_df: pd.DataFrame,
                tag: str, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: balanced accuracy at majority consensus, per phenomenology
    maj = metrics_df[metrics_df["consensus"] == "majority"].set_index("phenomenology")
    maj = maj.reindex(PHENOMENOLOGIES)
    x = np.arange(len(PHENOMENOLOGIES))
    ax = axes[0]
    ax.bar(x, maj["balanced_accuracy"].to_numpy(), color="#3b6ea5")
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(PHENOMENOLOGIES, rotation=45, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{tag}: model vs majority consensus")
    ax.legend()

    # Right: inter-rater kappa per phenomenology (the human ceiling)
    ax = axes[1]
    ir = interrater_df.set_index("phenomenology").reindex(PHENOMENOLOGIES)
    ax.bar(x, ir["kappa"].to_numpy(), color="#a5673b")
    ax.axhline(0.6, color="green", ls=":", lw=1, label="substantial (0.6)")
    ax.axhline(0.4, color="orange", ls=":", lw=1, label="moderate (0.4)")
    ax.set_xticks(x); ax.set_xticklabels(PHENOMENOLOGIES, rotation=45, ha="right")
    method = ir["method"].dropna().unique()
    mlabel = "/".join(m for m in method if m != "n/a") or "kappa"
    ax.set_ylabel(f"Inter-rater {mlabel} kappa")
    ax.set_ylim(min(0, np.nanmin(ir["kappa"].to_numpy()) if len(ir) else 0), 1)
    ax.set_title(f"{tag}: inter-rater agreement (human ceiling)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_csv", type=Path, required=True)
    ap.add_argument("--gt_xlsx", type=Path, required=True)
    ap.add_argument("--sheet", type=str, default=None)
    ap.add_argument("--dataset_tag", type=str, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args(argv)

    sheet = args.sheet or args.dataset_tag
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Predictions
    pred_df = pd.read_csv(args.pred_csv)
    pred_df["patient_id"] = pred_df["patient_id"].astype(str).str.strip()
    print(f"[INFO] {len(pred_df)} patient predictions from {args.pred_csv.name}")

    # 2) Ground truth
    gt_wide = parse_ground_truth(args.gt_xlsx, sheet)
    raters = sorted(gt_wide["rater"].unique())
    print(f"[INFO] sheet {sheet!r}: {gt_wide['patient_id'].nunique()} patients, "
          f"{len(raters)} raters ({', '.join(raters)})")

    # Sanity: patient overlap
    gt_patients = set(gt_wide["patient_id"].unique())
    pred_patients = set(pred_df["patient_id"].unique())
    common = gt_patients & pred_patients
    print(f"[INFO] patient overlap: {len(common)}/{len(gt_patients)} GT, "
          f"{len(pred_patients)} predicted")
    if len(common) == 0:
        print("[ERROR] No patient IDs in common between predictions and ground "
              "truth. Check patient_id formatting (e.g. 'P1' vs 'P01').")
        # show a few from each side
        print("  GT ids   :", sorted(gt_patients)[:8])
        print("  Pred ids :", sorted(pred_patients)[:8])
        return 2
    missing = sorted(gt_patients - pred_patients)
    if missing:
        print(f"[WARN] {len(missing)} GT patients have no prediction: {missing}")

    # 3) Consensus + inter-rater
    consensus = build_consensus(gt_wide)
    interrater_df = interrater_agreement(gt_wide)
    interrater_df.insert(0, "dataset", args.dataset_tag)

    # 4) Metrics
    metrics_df, confusion_df = evaluate(pred_df, consensus)
    metrics_df.insert(0, "dataset", args.dataset_tag)
    confusion_df.insert(0, "dataset", args.dataset_tag)

    # 5) Merged patient-level table (preds + each rater) for transparency
    merged = pred_df[["patient_id"] +
                     [f"pred__{p}" for p in PHENOMENOLOGIES if f"pred__{p}" in pred_df.columns] +
                     [f"score__{p}" for p in PHENOMENOLOGIES if f"score__{p}" in pred_df.columns]].copy()
    gt_piv = gt_wide.pivot_table(index="patient_id", columns="rater",
                                 values=PHENOMENOLOGIES, aggfunc="first")
    gt_piv.columns = [f"gt__{ph}__{rater}" for ph, rater in gt_piv.columns]
    merged = merged.merge(gt_piv.reset_index(), on="patient_id", how="left")

    # 6) Save
    tag = args.dataset_tag
    metrics_df.to_csv(args.out_dir / f"external_eval_{tag}__per_consensus.csv", index=False)
    interrater_df.to_csv(args.out_dir / f"external_eval_{tag}__interrater.csv", index=False)
    confusion_df.to_csv(args.out_dir / f"external_eval_{tag}__confusion.csv", index=False)
    merged.to_csv(args.out_dir / f"external_eval_{tag}__merged.csv", index=False)
    make_figure(metrics_df, interrater_df, tag,
                args.out_dir / f"external_eval_{tag}__summary.png")

    # 7) Console summary
    print(f"\n=== {tag}: model vs consensus (balanced accuracy) ===")
    piv = metrics_df.pivot_table(index="phenomenology", columns="consensus",
                                 values="balanced_accuracy")
    piv = piv.reindex(PHENOMENOLOGIES)
    cols = [c for c in ["any", "majority", "unanimous", "restricted"] if c in piv.columns]
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(piv[cols].to_string())

    print(f"\n=== {tag}: inter-rater agreement (human ceiling) ===")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(interrater_df.set_index("phenomenology")[["method", "kappa", "n_patients"]]
              .reindex(PHENOMENOLOGIES).to_string())

    print(f"\n[DONE] Evaluation written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
