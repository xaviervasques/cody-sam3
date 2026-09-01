#!/usr/bin/env python3
"""
run_allsets_exploration.py  (CODY-SAM3 — Phase 3, v2, version Mac native)
=========================================================================

Exploration de TOUS les sets de calibration possibles, en local (CPU, pas de
GPU, pas de Colab, pas de Google Drive). Meme objectif de calibration que
`calibrate_persite.py` (continuite CODY-2) :

    score = J(Y_true, Y_pred)
            - extra_penalty * fp_excess_norm                          (0.35)
            - fn_penalty    * fn_key_rate(Dystonia, Myoclonus, Chorea) (0.35)

avec J = jaccard (defaut) | exact_match | macro_f1, optimise conjointement sur
les 8 phenomenologies (coordinate descent), garde-fous de seuils, grille
d'agregation p70/p90/p95/max.

Pour CHAQUE set de calibration de taille k (C(n,k), echantillonne si > MAX_SETS) :
  - calibration sur le set,
  - evaluation held-out (patients restants) ET all-patients,
  - sous main present/absent et restricted, niveaux >=3/5, >=4/5, 5/5,
  - metriques multi-label (Hamming, Jaccard) + TP/TN/FP/FN.

Sorties (dans --out_dir) :
  allsets_<dataset>.csv.gz            table complete par dataset
  SUMMARY_allsets_distribution.csv    distributions (median, 5-95) -> publishable
  TOP5_ho_Jaccard.csv / TOP5_ho_Hamming.csv         top 5 par metrique (diagnostique)
  TOP5_ho_TP.csv / TOP5_ho_TN.csv / TOP5_ho_FP.csv / TOP5_ho_FN.csv  top 5 par confusion
  cody_sam3_allsets_top5.xlsx         tout regroupe (une feuille par critere)

Example (from the working directory):
  python run_allsets_exploration.py \
      --results_dir 07_external_results \
      --gt_xlsx 05_external_videos/dataset_inference.xlsx \
      --datasets dataset_2 dataset_3 \
      --out_dir 07_external_results/allsets_exploration

Dependances : pandas, numpy, openpyxl  (matplotlib non requis).
"""

from __future__ import annotations

import argparse
import gzip
import math
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes (objectif CODY-2)
# ---------------------------------------------------------------------------
PHEN = ["Dystonia", "Tremor", "Myoclonus", "Chorea",
        "Athetosis", "Ballismus", "Stereotypies", "Tics"]
AGG_GRID = ["p70", "p90", "p95", "max"]
THR_GRID = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33,
            0.35, 0.38, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
MIN_THR = {"Tremor": 0.35, "Tics": 0.35, "Ballismus": 0.30,
           "Stereotypies": 0.25, "Athetosis": 0.20}
MAX_THR = {"Dystonia": 0.55, "Myoclonus": 0.50, "Chorea": 0.55}


def label_defs_for(n_raters: int) -> List[Tuple[str, int]]:
    """Agreement levels adapted to the number of raters R.

    With 5 raters: main/restricted at >=3, >=4, 5  (the cody-2 / dataset_2 grid).
    With 2 raters: main/restricted at >=1 (at least one) and 2 (consensus) ONLY,
                   because thresholds of 3+ can never be reached with 2 raters.
    Generic R    : a sensible majority grid (ceil(R/2) .. R).
    The 'level' is an absolute vote count; reporting tags it as '<level>of<R>'.
    """
    if n_raters >= 5:
        levels = [3, 4, 5]
    elif n_raters == 2:
        levels = [1, 2]
    elif n_raters == 1:
        levels = [1]
    else:  # 3 or 4 raters
        lo = (n_raters + 1) // 2  # strict majority
        levels = sorted(set([lo, n_raters]))
    levels = [l for l in levels if l <= n_raters]
    return [(fam, lv) for fam in ("main", "restricted") for lv in levels]


def count_raters(rec: Dict[str, Dict[str, List[float]]]) -> int:
    """Max number of votes seen for any phenomenology across patients."""
    return max((len(rec[p][ph]) for p in rec for ph in PHEN), default=0)


EXTRA_PEN_DEFAULT = 0.35
FN_PEN_DEFAULT = 0.35
FN_KEY_DEFAULT = ["Dystonia", "Myoclonus", "Chorea"]


# ---------------------------------------------------------------------------
# Aggregation / ground truth
# ---------------------------------------------------------------------------
def agg_probs(p: np.ndarray, m: str) -> float:
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    m = m.lower()
    if m == "max":
        return float(np.max(p))
    if m == "mean":
        return float(np.mean(p))
    if m == "median":
        return float(np.median(p))
    if m == "noisy_or":
        return float(1.0 - np.prod(1.0 - p))
    mm = re.match(r"^p(\d{2,3})$", m)
    if mm:
        return float(np.percentile(p, float(mm.group(1))))
    raise ValueError(f"Unknown aggregation: {m}")


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
        for ph in PHEN:
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


# ---------------------------------------------------------------------------
# Objectif multi-label + penalites
# ---------------------------------------------------------------------------
def jaccard(Yt, Yp):
    inter = np.sum(Yt & Yp, axis=1)
    union = np.sum(Yt | Yp, axis=1)
    return float(np.mean(np.where(union == 0, 1.0, inter / union)))


def exact_match(Yt, Yp):
    return float(np.mean(np.all(Yt == Yp, axis=1)))


def macro_f1(Yt, Yp):
    f = []
    for j in range(Yt.shape[1]):
        tp = int(((Yt[:, j] == 1) & (Yp[:, j] == 1)).sum())
        fp = int(((Yt[:, j] == 0) & (Yp[:, j] == 1)).sum())
        fn = int(((Yt[:, j] == 1) & (Yp[:, j] == 0)).sum())
        f.append(1.0 if (tp == 0 and fp == 0 and fn == 0)
                 else (2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0))
    return float(np.mean(f))


def fp_excess_norm(Yt, Yp):
    return float(np.mean(np.maximum(0, Yp.sum(1) - Yt.sum(1))) / max(1.0, Yt.shape[1]))


def fn_key_rate(Yt, Yp, key):
    idx = [PHEN.index(l) for l in key if l in PHEN]
    vals = []
    for j in idx:
        pos = int((Yt[:, j] == 1).sum())
        if pos == 0:
            continue
        vals.append(int(((Yt[:, j] == 1) & (Yp[:, j] == 0)).sum()) / max(1, pos))
    return float(np.mean(vals)) if vals else 0.0


OBJ = {"jaccard": jaccard, "exact_match": exact_match, "macro_f1": macro_f1}


def thr_candidates(label: str) -> List[float]:
    floor = MIN_THR.get(label, min(THR_GRID))
    ceil = MAX_THR.get(label, max(THR_GRID))
    vals = [t for t in THR_GRID if floor - 1e-9 <= t <= ceil + 1e-9]
    return vals if vals else [min(max(THR_GRID[0], floor), ceil)]


# ---------------------------------------------------------------------------
# Calibration (coordinate descent) + evaluation
# ---------------------------------------------------------------------------
def calibrate(calib, rec, Pcache, obj_name, extra_pen, fn_pen, fn_key,
              family, level, max_iter=16):
    obj = OBJ[obj_name]
    Yt = np.array([[(0 if label_of(rec[p][ph], family, level) is None
                     else label_of(rec[p][ph], family, level))
                    for ph in PHEN] for p in calib], dtype=int)

    def score(asel, tsel):
        P = np.array([[Pcache[(PHEN[j], asel[j])][p] for j in range(len(PHEN))]
                      for p in calib])
        Yp = (P >= np.array(tsel)[None, :]).astype(int)
        return (obj(Yt, Yp) - extra_pen * fp_excess_norm(Yt, Yp)
                - fn_pen * fn_key_rate(Yt, Yp, fn_key))

    asel = ["p95"] * len(PHEN)
    tsel = [thr_candidates(l)[len(thr_candidates(l)) // 2] for l in PHEN]
    best = score(asel, tsel)
    for _ in range(max_iter):
        improved = False
        for j, l in enumerate(PHEN):
            for m in AGG_GRID:
                for t in thr_candidates(l):
                    ca, ct = asel.copy(), tsel.copy()
                    ca[j], ct[j] = m, t
                    sc = score(ca, ct)
                    if sc > best + 1e-12:
                        best, asel, tsel = sc, ca, ct
                        improved = True
        if not improved:
            break
    return {PHEN[j]: (asel[j], float(tsel[j])) for j in range(len(PHEN))}


def evaluate(rules, pats, rec, Pcache, family, level):
    TP = TN = FP = FN = 0
    ham, jac = [], []
    for p in pats:
        corr = tot = inter = union = 0
        for ph in PHEN:
            tl = label_of(rec[p][ph], family, level)
            if tl is None:
                continue
            am, t = rules[ph]
            pr = int(Pcache[(ph, am)][p] >= t)
            tot += 1
            corr += int(pr == tl)
            if pr == 1 and tl == 1:
                TP += 1; inter += 1; union += 1
            elif pr == 1 and tl == 0:
                FP += 1; union += 1
            elif pr == 0 and tl == 1:
                FN += 1; union += 1
            else:
                TN += 1
        if tot:
            ham.append(corr / tot)
        jac.append(inter / union if union > 0 else 1.0)
    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                Hamming=float(np.mean(ham)) if ham else np.nan,
                Jaccard=float(np.mean(jac)) if jac else np.nan)


# ---------------------------------------------------------------------------
# Run one dataset
# ---------------------------------------------------------------------------
def run_dataset(ds, results_dir, gt_xlsx, sheet, k, obj_name, extra_pen,
                fn_pen, fn_key, calib_family, calib_level, max_sets, seed):
    win = results_dir / ds / "reports" / "tables" / "inference_window_predictions.csv.gz"
    if not win.exists():
        print(f"[SKIP] {ds}: window file introuvable -> {win}")
        return None
    with gzip.open(win, "rt", encoding="utf-8") as f:
        w = pd.read_csv(f)
    w["patient_id"] = w["patient_id"].astype(str).str.strip()
    rec = parse_votes(gt_xlsx, sheet)
    patients = sorted([p for p in rec if p in set(w["patient_id"])])

    # Detect number of raters and adapt the agreement-level grid + calibration level.
    R = count_raters(rec)
    defs = label_defs_for(R)
    levels = sorted({lv for _, lv in defs})
    # calibration level: requested if reachable, else strict majority of R
    eff_calib_level = calib_level if calib_level in levels else max(levels)
    print(f"[{ds}] raters detectes = {R}  ->  niveaux = {[f'{l}of{R}' for l in levels]}  "
          f"(calibration: {calib_family} {eff_calib_level}of{R})")

    Pcache = {(ph, m): {p: agg_probs(
        w.loc[w.patient_id == p, f"prob__{ph}"].to_numpy(float), m) for p in patients}
        for ph in PHEN if f"prob__{ph}" in w.columns for m in AGG_GRID}

    total = math.comb(len(patients), k)
    if total <= max_sets:
        sets = list(combinations(patients, k))
        mode = f"exhaustif ({total})"
    else:
        rng = np.random.default_rng(seed)
        seen, sets = set(), []
        while len(sets) < max_sets:
            s = tuple(sorted(rng.choice(patients, k, replace=False)))
            if s not in seen:
                seen.add(s); sets.append(s)
        mode = f"echantillon ({max_sets}/{total})"

    import time
    print(f"[{ds}] {len(patients)} patients, k={k}, sets={mode}")
    t0 = time.time()
    rows = []
    for n_done, calib in enumerate(sets, 1):
        calib = list(calib)
        test = [p for p in patients if p not in set(calib)]
        rules = calibrate(calib, rec, Pcache, obj_name, extra_pen, fn_pen,
                          fn_key, calib_family, eff_calib_level)
        for family, level in defs:
            ho = evaluate(rules, test, rec, Pcache, family, level)
            al = evaluate(rules, patients, rec, Pcache, family, level)
            rows.append(dict(
                dataset=ds, calib=";".join(calib), heldout=";".join(test),
                n_raters=R, definition=family, agreement=f"{level}of{R}",
                ho_Hamming=ho["Hamming"], ho_Jaccard=ho["Jaccard"],
                ho_TP=ho["TP"], ho_TN=ho["TN"], ho_FP=ho["FP"], ho_FN=ho["FN"],
                all_Hamming=al["Hamming"], all_Jaccard=al["Jaccard"],
                all_TP=al["TP"], all_TN=al["TN"], all_FP=al["FP"], all_FN=al["FN"]))
        if n_done % 200 == 0:
            print(f"   ... {n_done}/{len(sets)} sets ({time.time()-t0:.0f}s)")
    print(f"   -> {len(rows)} lignes en {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", type=Path, required=True,
                    help="directory containing dataset_N/reports/tables/inference_window_predictions.csv.gz")
    ap.add_argument("--gt_xlsx", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["dataset_2", "dataset_3"])
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--k", type=str, default="dataset_1:1,dataset_2:5,dataset_3:5",
                    help="taille du set de calibration par dataset")
    ap.add_argument("--objective", choices=list(OBJ), default="jaccard")
    ap.add_argument("--extra_penalty", type=float, default=EXTRA_PEN_DEFAULT)
    ap.add_argument("--fn_penalty", type=float, default=FN_PEN_DEFAULT)
    ap.add_argument("--fn_key_labels", type=str, default=",".join(FN_KEY_DEFAULT))
    ap.add_argument("--calib_family", choices=["main", "restricted"], default="main")
    ap.add_argument("--calib_level", type=int, default=3,
                    help="vote count used FOR calibration; auto-adapted if unreachable "
                         "(e.g. falls back to the consensus level for 2-rater datasets)")
    ap.add_argument("--max_sets", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fn_key = [x.strip() for x in args.fn_key_labels.split(",") if x.strip()]
    kmap = {}
    for tok in args.k.split(","):
        name, val = tok.split(":")
        kmap[name.strip()] = int(val)

    frames = []
    for ds in args.datasets:
        k = kmap.get(ds, 5)
        df = run_dataset(ds, args.results_dir, args.gt_xlsx, ds, k,
                         args.objective, args.extra_penalty, args.fn_penalty,
                         fn_key, args.calib_family, args.calib_level,
                         args.max_sets, args.seed)
        if df is None:
            continue
        df.to_csv(args.out_dir / f"allsets_{ds}.csv.gz", index=False, compression="gzip")
        frames.append(df)

    if not frames:
        print("[ERROR] no dataset processed (check --results_dir / window files).")
        return 2
    BIG = pd.concat(frames, ignore_index=True)

    # --- Distributions (publishable) ---
    drows = []
    for (ds, dfam, agr), g in BIG.groupby(["dataset", "definition", "agreement"]):
        for metric in ["ho_Jaccard", "ho_Hamming"]:
            v = g[metric].dropna().to_numpy()
            if v.size == 0:
                continue
            drows.append(dict(dataset=ds, definition=dfam, agreement=agr,
                              metric=metric.replace("ho_", ""),
                              median=np.median(v), q05=np.percentile(v, 5),
                              q95=np.percentile(v, 95), n_sets=g["calib"].nunique()))
    dist = pd.DataFrame(drows)
    dist.to_csv(args.out_dir / "SUMMARY_allsets_distribution.csv", index=False)

    # --- Top 5 par metrique + par confusion (diagnostique) ---
    def top5(by, ascending):
        out = []
        for _, g in BIG.groupby(["dataset", "definition", "agreement"]):
            out.append(g.sort_values(by, ascending=ascending).head(5))
        return pd.concat(out, ignore_index=True)

    metric_cols = ["dataset", "definition", "agreement", "calib", "ho_Hamming",
                   "ho_Jaccard", "ho_TP", "ho_TN", "ho_FP", "ho_FN",
                   "all_Hamming", "all_Jaccard"]
    conf_cols = ["dataset", "definition", "agreement", "calib", "ho_TP", "ho_TN",
                 "ho_FP", "ho_FN", "ho_Hamming", "ho_Jaccard"]
    tops = {}
    for metric in ["ho_Jaccard", "ho_Hamming"]:
        t = top5(metric, ascending=False)[metric_cols]
        t.to_csv(args.out_dir / f"TOP5_{metric}.csv", index=False)
        tops[f"TOP5_{metric}"] = t
    for counter, asc in [("ho_TP", False), ("ho_TN", False),
                         ("ho_FP", True), ("ho_FN", True)]:
        t = top5(counter, ascending=asc)[conf_cols]
        t.to_csv(args.out_dir / f"TOP5_{counter}.csv", index=False)
        tops[f"TOP5_{counter}"] = t

    # --- Excel consolide ---
    xlsx_out = args.out_dir / "cody_sam3_allsets_top5.xlsx"
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as xl:
        readme = pd.DataFrame({"Notice": [
            "CODY-SAM3 — Exploration de tous les sets de calibration (objectif CODY-2).",
            "Vues: ho_ = held-out (validation externe), all_ = tous les patients.",
            "Definitions: main present/absent et restricted, niveaux 3/4/5 sur 5.",
            "SUMMARY_distribution = median + 5-95 percentile (publishable result).",
            "TOP5_* = diagnostique (quels patients sont representatifs); NE PAS rapporter comme performance.",
            "Pour le papier: set a priori clinicien (notebook/script v1), comme CODY-2.",
        ]})
        readme.to_excel(xl, sheet_name="README", index=False)
        dist.round(3).to_excel(xl, sheet_name="distribution", index=False)
        for name, t in tops.items():
            tt = t.copy()
            for c in ["ho_Hamming", "ho_Jaccard", "all_Hamming", "all_Jaccard"]:
                if c in tt.columns:
                    tt[c] = tt[c].round(3)
            tt.to_excel(xl, sheet_name=name[:31], index=False)

    print(f"\n[DONE] -> {args.out_dir}")
    print("  allsets_<dataset>.csv.gz, SUMMARY_allsets_distribution.csv, TOP5_*.csv")
    print(f"  {xlsx_out.name}")
    # quick console preview
    print("\n=== Distribution Jaccard held-out (apercu) ===")
    prev = dist[dist.metric == "Jaccard"]
    with pd.option_context("display.width", 200):
        print(prev.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
