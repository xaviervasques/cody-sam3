#!/usr/bin/env python3
"""
run_external_validation.py
==========================

End-to-end EXTERNAL VALIDATION pipeline for CODY-SAM3, implementing the cody-2
clinical-deployment protocol:

        train once  ->  calibrate on a few local patients  ->  deploy

This single script runs the whole external-site workflow for one or more
datasets, so a third party can reproduce the paper's external results without
any manual step:

    Step 1  INFERENCE        run the trained Tier-2 models on every patient of
            (GPU)            the site -> window-level probabilities + raw
                             per-patient scores.   [inference_sam3.py]

    Step 2  CALIBRATION      on a small, clinician-defined CALIBRATION subset of
            (CPU, light)     patients, pick per phenomenology the
                             (aggregation, threshold) pair maximising Youden's J.
                             [calibrate_persite.py]

    Step 3  EVALUATION       apply the calibrated rule to the remaining (TEST)
            (CPU, light)     patients and report two views:
                               - held-out : TEST patients only (generalisation)
                               - all      : every patient (operational).

GPU is required only for Step 1 (the foundation-model inference). Steps 2-3 are
light CPU post-processing of the saved window probabilities. If inference has
already been run, pass --skip_inference to re-do only calibration/evaluation.

Expected layout
---------------
    <root>/
      cody_sam3_pipeline/      inference_sam3.py, calibrate_persite.py, ...
      runs/sam3_tier2/         models/, bundle_meta.json,
                               cv_25fold/thresholds_patient_youden.json
      outputs_inference/       dataset_1/, dataset_2/, dataset_3/   (timeseries)
      dataset_inference.xlsx   multi-rater ground truth
      external_validation/     <- created: all outputs land here

Per-site calibration patients are given via --calib (one entry per dataset).

Usage (PC, conda env with tabicl + GPU)
---------------------------------------
    python cody_sam3_pipeline/run_external_validation.py \
        --root /path/to/sam_3 \
        --datasets 1 2 3 \
        --calib 1:P28  2:P1,P2,P3,P10,P9  3:P11,P13,P1,P10,P12

To re-run only calibration/evaluation (inference already done):
    ... --skip_inference
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_calib(entries: List[str]) -> Dict[int, str]:
    """Parse ['1:P28', '2:P1,P2'] -> {1:'P28', 2:'P1,P2'}."""
    out = {}
    for e in entries:
        if ":" not in e:
            raise ValueError(f"--calib entry must be 'N:ids', got {e!r}")
        n, ids = e.split(":", 1)
        out[int(n)] = ids
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--datasets", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--calib", type=str, nargs="+", required=True,
                    help="Per-dataset calibration patients, e.g. "
                         "1:P28 2:P1,P2,P3,P10,P9 3:P11,P13,P1,P10,P12")
    ap.add_argument("--pipeline_dir", type=Path, default=None)
    ap.add_argument("--bundle_dir", type=Path, default=None)
    ap.add_argument("--infer_root", type=Path, default=None)
    ap.add_argument("--gt_xlsx", type=Path, default=None)
    ap.add_argument("--out_root", type=Path, default=None)
    ap.add_argument("--objective", choices=["jaccard", "exact_match", "macro_f1"],
                    default="jaccard")
    ap.add_argument("--skip_inference", action="store_true",
                    help="Reuse existing inference outputs; only calibrate+evaluate.")
    args = ap.parse_args()

    root = args.root
    pipe = args.pipeline_dir or (root / "cody_sam3_pipeline")
    bundle = args.bundle_dir or (root / "runs" / "sam3_tier2")
    infer_root = args.infer_root or (root / "outputs_inference")
    gt_xlsx = args.gt_xlsx or (root / "dataset_inference.xlsx")
    out_root = args.out_root or (root / "external_validation")
    out_root.mkdir(parents=True, exist_ok=True)
    calib_map = parse_calib(args.calib)
    py = sys.executable

    # --- sanity ---
    print("=" * 70)
    print("CODY-SAM3 — External validation (train once / calibrate / deploy)")
    print("=" * 70)
    for label, p, required in [
        ("inference_sam3.py", pipe / "inference_sam3.py", True),
        ("calibrate_persite.py", pipe / "calibrate_persite.py", True),
        ("Tier-2 bundle", bundle / "bundle_meta.json", True),
        ("ground truth", gt_xlsx, True),
        ]:
        ok = p.exists()
        print(f"  [{'OK' if ok else 'MISSING'}] {label}: {p}")
        if required and not ok:
            print(f"[ERROR] required: {p}")
            return 2
    for n in args.datasets:
        ds_dir = infer_root / f"dataset_{n}"
        # The raw inference source folder is only needed when actually running inference.
        if not args.skip_inference and not ds_dir.is_dir():
            print(f"\n[SKIP] dataset_{n}: {ds_dir} not found")
            continue
        if n not in calib_map:
            print(f"\n[SKIP] dataset_{n}: no --calib entry given")
            continue

        out_dir = out_root / f"dataset_{n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        win_csv = out_dir / "reports" / "tables" / "inference_window_predictions.csv.gz"

        # ---------- Step 1: inference (GPU) ----------
        if not args.skip_inference:
            print(f"\n{'='*70}\n[dataset_{n}] STEP 1/3 — inference (GPU)\n{'='*70}")
            cmd = [py, str(pipe / "inference_sam3.py"),
                   "--infer_root", str(ds_dir),
                   "--bundle_dir", str(bundle),
                   "--out_dir", str(out_dir),
                   "--save_windows"]
            if subprocess.run(cmd).returncode != 0:
                print(f"[WARN] inference failed for dataset_{n}; skipping")
                continue
        else:
            print(f"\n[dataset_{n}] STEP 1/3 — inference SKIPPED (reusing existing)")
            if not win_csv.exists():
                print(f"[ERROR] --skip_inference but no window file at {win_csv}")
                continue

        # ---------- Step 2+3: calibration + dual evaluation (CPU) ----------
        print(f"\n{'='*70}\n[dataset_{n}] STEP 2-3/3 — per-site calibration + evaluation\n{'='*70}")
        cmd = [py, str(pipe / "calibrate_persite.py"),
               "--win_csv", str(win_csv),
               "--gt_xlsx", str(gt_xlsx),
               "--sheet", f"dataset_{n}",
               "--dataset_tag", f"dataset_{n}",
               "--calib", calib_map[n],
               "--objective", args.objective,
               "--out_dir", str(out_dir / "eval_persite")]
        subprocess.run(cmd)

    print(f"\n[DONE] External validation complete. Outputs under {out_root}")
    print("  Per dataset: reports/ (inference) + eval_persite/ (calibrated metrics + figure)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
