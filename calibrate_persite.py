#!/usr/bin/env python3
"""
calibrate_persite.py  (CODY-2 protocol)
=======================================

Per-site calibrate-then-deploy evaluation for CODY-SAM3, reproducing the
cody-2 calibration objective (calibrate_dataset2.py) for continuity between
the two papers.

        train once  ->  calibrate on a few local patients  ->  deploy

Calibration is a JOINT multi-label optimisation over all eight phenomenologies.
For each (aggregation, threshold) configuration the patient-level 8-vector is
predicted and scored as

    score = J(Y_true, Y_pred)
            - extra_penalty * fp_excess_norm
            - fn_penalty    * fn_key_rate(key_labels)

where J is the multi-label objective (jaccard | exact_match | macro_f1),
fp_excess_norm penalises predicting more positive labels than the GT cardinality
(over-calling), and fn_key_rate penalises missed positives on clinically
important labels (default Dystonia, Myoclonus, Chorea). Per-label threshold
guard-rails (min/max) prevent low-prevalence labels from collapsing to trivial
all-positive thresholds. Search is coordinate descent over an aggregation grid
(p70,p90,p95,max). These are the cody-2 defaults.

Rules are fitted on the clinician-selected CALIBRATION subset, then deployed on
the held-out patients. Results are reported in three views (held-out / all /
calibration) under the main present/absent and restrictive agreement-based label
definitions at agreement levels >=3/5, >=4/5, 5/5, with multi-label metrics
(Hamming, Jaccard), pooled confusion (TP/TN/FP/FN) and per-phenotype confusion.

Inputs
------
  --win_csv      inference_window_predictions.csv.gz for ONE dataset
  --gt_xlsx      dataset_inference.xlsx
  --sheet        e.g. "dataset_2"
  --dataset_tag  e.g. "dataset_2"
  --calib        comma-separated calibration patient ids
  --out_dir      output directory

Optional (cody-2 defaults)
--------------------------
  --objective jaccard|exact_match|macro_f1 (jaccard)
  --extra_penalty 0.35   --fn_penalty 0.35
  --fn_key_labels Dystonia,Myoclonus,Chorea
  --agg_grid p70,p90,p95,max
  --calib_family main|restricted (main)   --calib_level 3|4|5 (3)

Outputs (in --out_dir)
----------------------
  persite_<tag>__rules.json / __agg_map.json / __thresholds.json
  persite_<tag>__metrics.csv             Hamming/Jaccard + TP/TN/FP/FN per view x definition
  persite_<tag>__perphen_confusion.csv   per-phenotype confusion per view x definition
  persite_<tag>__summary.png

Usage
-----
    python calibrate_persite.py \
        --win_csv external_validation/dataset_2/reports/tables/inference_window_predictions.csv.gz \
        --gt_xlsx dataset_inference.xlsx --sheet dataset_2 --dataset_tag dataset_2 \
        --calib P3,P5,P8,P10,P11 \
        --out_dir external_validation/dataset_2/eval_persite
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PHENOMENOLOGIES = ["Dystonia", "Tremor", "Myoclonus", "Chorea",
                   "Athetosis", "Ballismus", "Stereotypies", "Tics"]
AGG_GRID_DEFAULT = ["p70", "p90", "p95", "max"]
THR_GRID = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33,
            0.35, 0.38, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
MIN_THR = {"Tremor": 0.35, "Tics": 0.35, "Ballismus": 0.30,
           "Stereotypies": 0.25, "Athetosis": 0.20}
MAX_THR = {"Dystonia": 0.55, "Myoclonus": 0.50, "Chorea": 0.55}
LABEL_DEFS = [("main", 3), ("main", 4), ("main", 5),
              ("restricted", 3), ("restricted", 4), ("restricted", 5)]


def agg_probs(probs: np.ndarray, method: str) -> float:
    p = probs[np.isfinite(probs)]
    if p.size == 0:
        return float("nan")
    method = method.lower()
    if method == "max":
        return float(np.max(p))
    if method == "mean":
        return float(np.mean(p))
    if method == "median":
        return float(np.median(p))
    if method == "noisy_or":
        return float(1.0 - np.prod(1.0 - p))
    m = re.match(r"^p(\d{2,3})$", method)
    if m:
        return float(np.percentile(p, float(m.group(1))))
    raise ValueError(f"Unknown aggregation: {method}")


def parse_votes(xlsx: Path, sheet: str) -> Dict[str, Dict[str, List[float]]]:
    raw = pd.read_excel(xlsx, sheet_name=sheet, header=None)
    rater_row = raw.iloc[0].tolist()
    phen_row = raw.iloc[1].tolist()
    starts = [j for j, v in enumerate(rater_row)
              if isinstance(v, str) and v.strip() and v.strip().lower() != "nan"]
    data = raw.iloc[2:].reset_index(drop=True)
    pids = data.iloc[:, 1].astype(str).str.strip()
    rec: Dict[str, Dict[str, List[float]]] = {}
    for i, pid in enumerate(pids):
        if pid in ("", "nan", "None"):
            continue
        rec[pid] = {}
        for ph in PHENOMENOLOGIES:
            vals = []
            for s in starts:
                for off in range(8):
                    c = s + off
                    if c < raw.shape[1] and str(phen_row[c]).strip() == ph:
                        try:
                            vals.append(float(str(data.iloc[i, c]).replace(",", ".")))
                        except ValueError:
                            pass
            rec[pid][ph] = [v for v in vals if v in (0.0, 1.0)]
    return rec


def label_of(votes: List[float], family: str, level: int) -> Optional[int]:
    if not votes:
        return None
    npos = sum(v == 1 for v in votes)
    present = npos >= level
    if family == "main":
        return 1 if present else 0
    if present:
        return 1
    if npos == 0:
        return 0
    return None


def jaccard(Yt: np.ndarray, Yp: np.ndarray) -> float:
    inter = np.sum(Yt & Yp, axis=1)
    union = np.sum(Yt | Yp, axis=1)
    return float(np.mean(np.where(union == 0, 1.0, inter / union)))


def exact_match(Yt: np.ndarray, Yp: np.ndarray) -> float:
    return float(np.mean(np.all(Yt == Yp, axis=1)))


def macro_f1(Yt: np.ndarray, Yp: np.ndarray) -> float:
    f1s = []
    for j in range(Yt.shape[1]):
        tp = int(np.sum((Yt[:, j] == 1) & (Yp[:, j] == 1)))
        fp = int(np.sum((Yt[:, j] == 0) & (Yp[:, j] == 1)))
        fn = int(np.sum((Yt[:, j] == 1) & (Yp[:, j] == 0)))
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(1.0)
        else:
            denom = 2 * tp + fp + fn
            f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def fp_excess_norm(Yt: np.ndarray, Yp: np.ndarray) -> float:
    kp = Yp.sum(1).astype(float)
    kt = Yt.sum(1).astype(float)
    return float(np.mean(np.maximum(0.0, kp - kt)) / max(1.0, float(Yt.shape[1])))


def fn_key_rate(Yt: np.ndarray, Yp: np.ndarray, labels: List[str],
                key: List[str]) -> float:
    idx = [labels.index(l) for l in key if l in labels]
    vals = []
    for j in idx:
        pos = int((Yt[:, j] == 1).sum())
        if pos == 0:
            continue
        fn = int(((Yt[:, j] == 1) & (Yp[:, j] == 0)).sum())
        vals.append(fn / max(1, pos))
    return float(np.mean(vals)) if vals else 0.0


OBJECTIVES = {"jaccard": jaccard, "exact_match": exact_match, "macro_f1": macro_f1}


def thr_candidates(label: str) -> List[float]:
    floor = MIN_THR.get(label, min(THR_GRID))
    ceil = MAX_THR.get(label, max(THR_GRID))
    vals = [t for t in THR_GRID if floor - 1e-9 <= t <= ceil + 1e-9]
    return vals if vals else [min(max(THR_GRID[0], floor), ceil)]


def calibrate(calib, rec, Pcache, obj_name, extra_penalty, fn_penalty,
              fn_key, agg_grid, family, level, max_iter=16):
    obj_fn = OBJECTIVES[obj_name]
    Yt = np.array([[(0 if label_of(rec[p][ph], family, level) is None
                     else label_of(rec[p][ph], family, level))
                    for ph in PHENOMENOLOGIES] for p in calib], dtype=int)

    def score(agg_sel, thr_sel):
        P = np.array([[Pcache[(PHENOMENOLOGIES[j], agg_sel[j])][p]
                       for j in range(len(PHENOMENOLOGIES))] for p in calib])
        Yp = (P >= np.asarray(thr_sel)[None, :]).astype(int)
        return (obj_fn(Yt, Yp)
                - extra_penalty * fp_excess_norm(Yt, Yp)
                - fn_penalty * fn_key_rate(Yt, Yp, PHENOMENOLOGIES, fn_key))

    agg_sel = ["p95"] * len(PHENOMENOLOGIES)
    thr_sel = [thr_candidates(l)[len(thr_candidates(l)) // 2] for l in PHENOMENOLOGIES]
    best = score(agg_sel, thr_sel)
    for _ in range(max_iter):
        improved = False
        for j, l in enumerate(PHENOMENOLOGIES):
            for m in agg_grid:
                for t in thr_candidates(l):
                    ca, ct = agg_sel.copy(), thr_sel.copy()
                    ca[j], ct[j] = m, t
                    sc = score(ca, ct)
                    if sc > best + 1e-12:
                        best, agg_sel, thr_sel = sc, ca, ct
                        improved = True
        if not improved:
            break
    return {PHENOMENOLOGIES[j]: (agg_sel[j], float(thr_sel[j]))
            for j in range(len(PHENOMENOLOGIES))}


def evaluate(rules, pats, rec, Pcache, family, level):
    TP = TN = FP = FN = 0
    ham, jac = [], []
    per = {ph: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for ph in PHENOMENOLOGIES}
    for p in pats:
        corr = tot = inter = union = 0
        for ph in PHENOMENOLOGIES:
            tl = label_of(rec[p][ph], family, level)
            if tl is None:
                continue
            am, t = rules[ph]
            pr = int(Pcache[(ph, am)][p] >= t)
            tot += 1
            corr += int(pr == tl)
            if pr == 1 and tl == 1:
                TP += 1; inter += 1; union += 1; per[ph]["tp"] += 1
            elif pr == 1 and tl == 0:
                FP += 1; union += 1; per[ph]["fp"] += 1
            elif pr == 0 and tl == 1:
                FN += 1; union += 1; per[ph]["fn"] += 1
            else:
                TN += 1; per[ph]["tn"] += 1
        if tot:
            ham.append(corr / tot)
        jac.append(inter / union if union > 0 else 1.0)
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "Hamming": float(np.mean(ham)) if ham else np.nan,
            "Jaccard": float(np.mean(jac)) if jac else np.nan,
            "per_phen": per}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--win_csv", type=Path, required=True)
    ap.add_argument("--gt_xlsx", type=Path, required=True)
    ap.add_argument("--sheet", type=str, required=True)
    ap.add_argument("--dataset_tag", type=str, required=True)
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--objective", choices=list(OBJECTIVES), default="jaccard")
    ap.add_argument("--extra_penalty", type=float, default=0.35)
    ap.add_argument("--fn_penalty", type=float, default=0.35)
    ap.add_argument("--fn_key_labels", type=str, default="Dystonia,Myoclonus,Chorea")
    ap.add_argument("--agg_grid", type=str, default=",".join(AGG_GRID_DEFAULT))
    ap.add_argument("--calib_family", choices=["main", "restricted"], default="main")
    ap.add_argument("--calib_level", type=int, default=3)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calib = [p.strip() for p in args.calib.split(",") if p.strip()]
    agg_grid = [a.strip() for a in args.agg_grid.split(",") if a.strip()]
    fn_key = [x.strip() for x in args.fn_key_labels.split(",") if x.strip()]

    if str(args.win_csv).endswith(".gz"):
        with gzip.open(args.win_csv, "rt", encoding="utf-8") as f:
            w = pd.read_csv(f)
    else:
        w = pd.read_csv(args.win_csv)
    w["patient_id"] = w["patient_id"].astype(str).str.strip()

    rec = parse_votes(args.gt_xlsx, args.sheet)
    patients = sorted([p for p in rec if p in set(w["patient_id"])])
    calib = [p for p in calib if p in patients]
    test = [p for p in patients if p not in calib]
    print(f"[INFO] {args.dataset_tag}: {len(patients)} patients "
          f"({len(calib)} calib, {len(test)} held-out)")
    if len(calib) < 2:
        print("[ERROR] need >= 2 calibration patients in data")
        return 2

    Pcache = {(ph, m): {p: agg_probs(
        w.loc[w.patient_id == p, f"prob__{ph}"].to_numpy(float), m) for p in patients}
        for ph in PHENOMENOLOGIES if f"prob__{ph}" in w.columns for m in agg_grid}

    rules = calibrate(calib, rec, Pcache, args.objective, args.extra_penalty,
                      args.fn_penalty, fn_key, agg_grid,
                      args.calib_family, args.calib_level)

    tag = args.dataset_tag
    (args.out_dir / f"persite_{tag}__rules.json").write_text(
        json.dumps({k: {"agg": v[0], "thr": round(v[1], 4)} for k, v in rules.items()}, indent=2))
    (args.out_dir / f"persite_{tag}__agg_map.json").write_text(
        json.dumps({k: v[0] for k, v in rules.items()}, indent=2))
    (args.out_dir / f"persite_{tag}__thresholds.json").write_text(
        json.dumps({k: round(v[1], 4) for k, v in rules.items()}, indent=2))

    metric_rows, perphen_rows = [], []
    for family, level in LABEL_DEFS:
        for view, pats in [("heldout", test), ("all", patients), ("calibration", calib)]:
            r = evaluate(rules, pats, rec, Pcache, family, level)
            metric_rows.append({"dataset": tag, "definition": family,
                                "agreement": f"{level}of5", "view": view,
                                "n_patients": len(pats), "Hamming": r["Hamming"],
                                "Jaccard": r["Jaccard"], "TP": r["TP"], "TN": r["TN"],
                                "FP": r["FP"], "FN": r["FN"]})
            for ph in PHENOMENOLOGIES:
                perphen_rows.append({"dataset": tag, "definition": family,
                                     "agreement": f"{level}of5", "view": view,
                                     "phenomenology": ph, **r["per_phen"][ph]})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.out_dir / f"persite_{tag}__metrics.csv", index=False)
    pd.DataFrame(perphen_rows).to_csv(
        args.out_dir / f"persite_{tag}__perphen_confusion.csv", index=False)
    _figure(metrics_df, tag, args.out_dir / f"persite_{tag}__summary.png")

    print(f"\n=== {tag}: calibrated rules (calibration set: {', '.join(calib)}) ===")
    for ph, (a, t) in rules.items():
        print(f"  {ph:<13} agg={a:<5} thr={t:.3f}")
    print(f"\n=== {tag}: held-out & all-patients (Hamming / Jaccard) ===")
    show = metrics_df[metrics_df.view.isin(["heldout", "all"])]
    piv = show.pivot_table(index=["definition", "agreement"], columns="view",
                           values=["Hamming", "Jaccard"])
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(piv.to_string())
    print(f"\n[DONE] -> {args.out_dir}")
    return 0


def _figure(metrics_df, tag, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ["Hamming", "Jaccard"]):
        sub_h = metrics_df[metrics_df.view == "heldout"]
        sub_a = metrics_df[metrics_df.view == "all"]
        labels = [f"{d}\n{a}" for d, a in zip(sub_h.definition, sub_h.agreement)]
        x = np.arange(len(sub_h))
        ax.bar(x - 0.2, sub_h[metric].to_numpy(), width=0.4, label="held-out", color="#3b6ea5")
        ax.bar(x + 0.2, sub_a[metric].to_numpy(), width=0.4, label="all", color="#a5673b")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_ylabel(metric); ax.set_title(f"{tag}: {metric}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
