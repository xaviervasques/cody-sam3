#!/usr/bin/env python3
"""
calibration_robustness.py
=========================

Robustness of the per-site calibrate-then-deploy protocol to the CHOICE of the
calibration subset.

Selecting one calibration set by hand invites a cherry-picking critique. This
script removes that ambiguity: for a given site it enumerates EVERY possible
calibration subset of size k (all C(n, k) combinations), and for each one it
calibrates the (aggregation, threshold) rule on that subset and deploys on the
remaining (held-out) patients. The output is the full DISTRIBUTION of held-out
performance across all calibration sets — not a single hand-picked number.

What to report in the paper: the distribution (median and 5-95 percentile
interval). A wide interval means the result depends on which patients are used
to calibrate (fragile, small-sample); a tight interval means the conclusion is
robust to the calibration choice.

The best / worst sets are also identified, but ONLY for diagnostic inspection
(which patients are "easy" vs "hard" to calibrate on). The best set is NOT a
publishable result — reporting it would be optimisation on the test data.

Metrics per phenomenology, computed on the held-out patients of each split:
  - balanced_accuracy and roc_auc   (when held-out has both classes)
  - sensitivity (detection rate)     (when held-out is all-positive)

Inputs
------
  --win_csv       inference_window_predictions.csv.gz for ONE dataset.
  --gt_xlsx       dataset_inference.xlsx
  --sheet         e.g. "dataset_3"
  --dataset_tag   e.g. "dataset_3"
  --k             calibration-set size (default 5)
  --out_dir       output directory

Optional
--------
  --objective     youden (default) | f1 | balanced
  --dev_thresholds JSON fallback for phenomenologies single-class in a calib set
  --default_agg   fallback aggregation (default p95)
  --aggs          aggregation candidates (default max,mean,median,noisy_or,p90,p95)
  --apriori       comma-separated a-priori calibration ids to locate in the
                  distribution (e.g. "P11,P13,P1,P10,P12")
  --max_combos    if C(n,k) exceeds this, sample this many random sets instead
                  of enumerating (default 50000; both our datasets are below it)
  --seed          RNG seed for sampling fallback (default 0)

Outputs (in --out_dir)
----------------------
  robustness_<tag>_k<k>__per_split.csv.gz   every split x phenomenology row
  robustness_<tag>_k<k>__distribution.csv   median/IQR/5-95/min/max per phenom
  robustness_<tag>_k<k>__best_worst.csv     best & worst set per phenom (diagnostic)
  robustness_<tag>_k<k>__distribution.png   violin/box of the distribution

Usage
-----
    python calibration_robustness.py \
        --win_csv external_validation/dataset_3/reports/tables/inference_window_predictions.csv.gz \
        --gt_xlsx dataset_inference.xlsx --sheet dataset_3 --dataset_tag dataset_3 \
        --k 5 --apriori P11,P13,P1,P10,P12 \
        --dev_thresholds runs/sam3_tier2/cv_25fold/thresholds_patient_youden.json \
        --out_dir external_validation/dataset_3/robustness
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PHENOMENOLOGIES = ["Dystonia", "Tremor", "Myoclonus", "Chorea",
                   "Athetosis", "Ballismus", "Stereotypies", "Tics"]
DEFAULT_AGGS = ["max", "mean", "median", "noisy_or", "p90", "p95"]


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


def gt_majority(xlsx: Path, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx, sheet_name=sheet, header=None)
    rater_row = raw.iloc[0].tolist()
    phen_row = raw.iloc[1].tolist()
    starts = [j for j, v in enumerate(rater_row)
              if isinstance(v, str) and v.strip() and v.strip().lower() != "nan"]
    data = raw.iloc[2:].reset_index(drop=True)
    pids = data.iloc[:, 1].astype(str).str.strip()
    rows = []
    for i, pid in enumerate(pids):
        if pid in ("", "nan", "None"):
            continue
        rec = {"patient_id": pid}
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
            vals = [v for v in vals if v in (0.0, 1.0)]
            rec[ph] = (1 if sum(v == 1 for v in vals) > len(vals) / 2 else 0) if vals else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).set_index("patient_id")


def best_threshold(scores: np.ndarray, labels: np.ndarray,
                   objective: str) -> Optional[float]:
    if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        return None
    grid = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best_thr, best_val = 0.5, -np.inf
    for t in grid:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        if objective == "youden":
            val = sens + spec - 1.0
        elif objective == "balanced":
            val = (sens + spec) / 2.0
        else:  # f1
            ppv = tp / (tp + fp) if (tp + fp) else 0.0
            val = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) > 0 else 0.0
        if val > best_val:
            best_val, best_thr = val, float(t)
    return best_thr


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--win_csv", type=Path, required=True)
    ap.add_argument("--gt_xlsx", type=Path, required=True)
    ap.add_argument("--sheet", type=str, required=True)
    ap.add_argument("--dataset_tag", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--objective", choices=["youden", "f1", "balanced"], default="youden")
    ap.add_argument("--dev_thresholds", type=Path, default=None)
    ap.add_argument("--default_agg", type=str, default="p95")
    ap.add_argument("--aggs", type=str, default=",".join(DEFAULT_AGGS))
    ap.add_argument("--apriori", type=str, default=None)
    ap.add_argument("--max_combos", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    aggs = [a.strip() for a in args.aggs.split(",") if a.strip()]
    dev_thr = {}
    if args.dev_thresholds and args.dev_thresholds.exists():
        dev_thr = {k: float(v) for k, v in json.loads(args.dev_thresholds.read_text()).items()}

    if str(args.win_csv).endswith(".gz"):
        with gzip.open(args.win_csv, "rt", encoding="utf-8") as f:
            win = pd.read_csv(f)
    else:
        win = pd.read_csv(args.win_csv)
    win["patient_id"] = win["patient_id"].astype(str).str.strip()

    gt = gt_majority(args.gt_xlsx, args.sheet)
    patients = [p for p in gt.index if p in set(win["patient_id"])]
    n = len(patients)
    k = args.k
    if k >= n:
        print(f"[ERROR] k={k} >= n={n}; need at least one held-out patient.")
        return 2

    # Pre-compute per-patient aggregated scores for every (phenomenology, agg)
    # so the inner loop is pure table lookups (fast over many combinations).
    score_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
    for ph in PHENOMENOLOGIES:
        col = f"prob__{ph}"
        if col not in win.columns:
            continue
        for am in aggs:
            d = {}
            for p in patients:
                pr = win.loc[win["patient_id"] == p, col].to_numpy(dtype=float)
                d[p] = agg_probs(pr, am)
            score_cache[(ph, am)] = d

    labels = {ph: {p: gt.loc[p, ph] for p in patients} for ph in PHENOMENOLOGIES}

    # Enumerate or sample calibration sets
    total = math.comb(n, k)
    if total <= args.max_combos:
        calib_sets = list(combinations(patients, k))
        mode = f"exhaustive ({total} sets)"
    else:
        rng = np.random.default_rng(args.seed)
        seen = set()
        calib_sets = []
        while len(calib_sets) < args.max_combos:
            s = tuple(sorted(rng.choice(patients, size=k, replace=False)))
            if s not in seen:
                seen.add(s); calib_sets.append(s)
        mode = f"sampled ({args.max_combos} of {total})"
    print(f"[INFO] {args.dataset_tag}: n={n} patients, k={k}, {mode}")

    from sklearn.metrics import roc_auc_score

    rows = []
    for ci, calib in enumerate(calib_sets):
        calib_set = set(calib)
        test = [p for p in patients if p not in calib_set]
        for ph in PHENOMENOLOGIES:
            if (ph, aggs[0]) not in score_cache:
                continue
            cal_lab = np.array([labels[ph][p] for p in calib], dtype=float)
            # choose best (agg, thr) on calibration subset
            best = None  # (agg, thr, val)
            for am in aggs:
                sc = score_cache[(ph, am)]
                s = np.array([sc[p] for p in calib], dtype=float)
                ok = np.isfinite(s) & np.isfinite(cal_lab)
                if ok.sum() < 2:
                    continue
                thr = best_threshold(s[ok], cal_lab[ok].astype(int), args.objective)
                if thr is None:
                    continue
                # objective value to pick best agg
                pred = (s[ok] >= thr).astype(int); yy = cal_lab[ok].astype(int)
                tp = ((pred == 1) & (yy == 1)).sum(); fn = ((pred == 0) & (yy == 1)).sum()
                tn = ((pred == 0) & (yy == 0)).sum(); fp = ((pred == 1) & (yy == 0)).sum()
                se = tp / (tp + fn) if (tp + fn) else 0; sp = tn / (tn + fp) if (tn + fp) else 0
                val = se + sp - 1
                if best is None or val > best[2]:
                    best = (am, thr, val)
            if best is not None:
                am, thr = best[0], best[1]; src = "site"
            else:
                am = args.default_agg; thr = dev_thr.get(ph, 0.5); src = "dev"

            sc = score_cache[(ph, am)]
            ts = np.array([sc[p] for p in test], dtype=float)
            ty = np.array([labels[ph][p] for p in test], dtype=float)
            ok = np.isfinite(ts) & np.isfinite(ty)
            ts, ty = ts[ok], ty[ok].astype(int)
            npos, nneg = int((ty == 1).sum()), int((ty == 0).sum())
            bal = roc = sens = np.nan
            if npos and nneg:
                pred = (ts >= thr).astype(int)
                tp = ((pred == 1) & (ty == 1)).sum(); fn = ((pred == 0) & (ty == 1)).sum()
                tn = ((pred == 0) & (ty == 0)).sum(); fp = ((pred == 1) & (ty == 0)).sum()
                se = tp / (tp + fn); sp = tn / (tn + fp)
                bal = (se + sp) / 2; sens = se
                try:
                    roc = roc_auc_score(ty, ts)
                except ValueError:
                    roc = np.nan
            elif npos:
                sens = float((ts >= thr).astype(int).mean())
            rows.append({"split": ci, "phenomenology": ph, "agg": am, "src": src,
                         "threshold": thr, "n_pos": npos, "n_neg": nneg,
                         "balanced_accuracy": bal, "roc_auc": roc, "sensitivity": sens,
                         "calib": ";".join(calib)})

    per = pd.DataFrame(rows)
    tag = args.dataset_tag
    per.to_csv(args.out_dir / f"robustness_{tag}_k{k}__per_split.csv.gz",
               index=False, compression="gzip")

    # Distribution summary per phenomenology
    def summarize(g, col):
        v = g[col].dropna().to_numpy()
        if v.size == 0:
            return {}
        return {f"{col}_median": float(np.median(v)),
                f"{col}_q05": float(np.percentile(v, 5)),
                f"{col}_q95": float(np.percentile(v, 95)),
                f"{col}_min": float(v.min()), f"{col}_max": float(v.max()),
                f"{col}_n": int(v.size)}

    dist_rows = []
    for ph, g in per.groupby("phenomenology"):
        rec = {"dataset": tag, "phenomenology": ph,
               "n_splits": int(g["split"].nunique())}
        for col in ["balanced_accuracy", "roc_auc", "sensitivity"]:
            rec.update(summarize(g, col))
        dist_rows.append(rec)
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(args.out_dir / f"robustness_{tag}_k{k}__distribution.csv", index=False)

    # Best / worst set per phenomenology (DIAGNOSTIC ONLY)
    bw_rows = []
    for ph, g in per.groupby("phenomenology"):
        for metric in ["balanced_accuracy", "roc_auc"]:
            gg = g.dropna(subset=[metric])
            if gg.empty:
                continue
            bi = gg[metric].idxmax(); wi = gg[metric].idxmin()
            bw_rows.append({"phenomenology": ph, "metric": metric,
                            "best_value": gg.loc[bi, metric], "best_calib": gg.loc[bi, "calib"],
                            "worst_value": gg.loc[wi, metric], "worst_calib": gg.loc[wi, "calib"]})
    pd.DataFrame(bw_rows).to_csv(args.out_dir / f"robustness_{tag}_k{k}__best_worst.csv", index=False)

    # A-priori set position
    apriori_note = ""
    if args.apriori:
        ap_set = ";".join(sorted([p.strip() for p in args.apriori.split(",")]))
        # match against sorted calib strings
        per["calib_sorted"] = per["calib"].apply(lambda s: ";".join(sorted(s.split(";"))))
        sub = per[per["calib_sorted"] == ap_set]
        if not sub.empty:
            apriori_note = f"\nA-priori set {args.apriori}:"
            for ph, g in sub.groupby("phenomenology"):
                b = g["balanced_accuracy"].mean(); r = g["roc_auc"].mean()
                if np.isfinite(b) or np.isfinite(r):
                    apriori_note += f"\n  {ph:<13} bal={b:.3f} roc={r:.3f}"

    _figure(per, tag, k, args.out_dir / f"robustness_{tag}_k{k}__distribution.png")

    # Console
    print(f"\n=== {tag}: distribution over all calibration sets (k={k}) ===")
    cols = ["phenomenology", "balanced_accuracy_median", "balanced_accuracy_q05",
            "balanced_accuracy_q95", "roc_auc_median", "roc_auc_q05", "roc_auc_q95",
            "sensitivity_median"]
    cols = [c for c in cols if c in dist.columns]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(dist[cols].to_string(index=False))
    if apriori_note:
        print(apriori_note)
    print(f"\n[DONE] -> {args.out_dir}")
    return 0


def _figure(per: pd.DataFrame, tag: str, k: int, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in [(axes[0], "balanced_accuracy", "Balanced accuracy"),
                              (axes[1], "roc_auc", "ROC-AUC")]:
        data, labels = [], []
        for ph in PHENOMENOLOGIES:
            v = per[per["phenomenology"] == ph][metric].dropna().to_numpy()
            if v.size >= 5:
                data.append(v); labels.append(ph)
        if data:
            bp = ax.boxplot(data, labels=labels, showmeans=True, vert=True)
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel(title)
        ax.set_title(f"{tag}: {title} over all C(n,{k}) calibration sets")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
