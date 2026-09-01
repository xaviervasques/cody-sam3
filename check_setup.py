#!/usr/bin/env python3
"""
check_setup.py
==============

Full environment check before launching CODY-SAM3 training.

The pipeline is self-contained: it does not need the external cody-2
repository (the helpers are vendored in ``cody2_utils.py`` next to this file).

Usage
-----
    python check_setup.py [--merged_root PATH]

    --merged_root   Optional. Root of the merged-by-patient training data
                    (e.g. 03_train_merged_by_patient). When given, the
                    expected dataset_* subfolders and file counts are checked.
"""

import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description="CODY-SAM3 environment check")
parser.add_argument("--merged_root", type=Path, default=None,
                    help="Root of the merged-by-patient training data (optional)")
args = parser.parse_args()

print("=" * 70)
print("SETUP CHECK — CODY-SAM3 (self-contained pipeline)")
print("=" * 70)

# [1] Python
print("\n[1] Python environment")
print(f"  Python      : {sys.version.split()[0]}")
print(f"  Executable  : {sys.executable}")

# [2] PyTorch + CUDA
print("\n[2] PyTorch + CUDA")
try:
    import torch
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA avail. : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM        : {vram_gb:.1f} GB")
        cap = torch.cuda.get_device_capability(0)
        print(f"  Compute cap : sm_{cap[0]}{cap[1]}")
except Exception as e:
    print(f"  ERROR: {e}")

# [3] Required packages
print("\n[3] Required packages")
required = ["tabicl", "antropy", "pandas", "numpy", "scipy",
            "sklearn", "openpyxl", "joblib", "matplotlib", "seaborn"]
missing = []
for pkg in required:
    try:
        m = __import__(pkg)
        v = getattr(m, "__version__", "?")
        print(f"  {pkg:<14s} {v}")
    except ImportError:
        print(f"  {pkg:<14s} MISSING")
        missing.append(pkg)

# [4] Pipeline imports (self-contained: only this directory is needed)
print("\n[4] Pipeline imports (self-contained)")
PIPELINE = Path(__file__).resolve().parent
print(f"  {PIPELINE}: {'OK' if PIPELINE.exists() else 'MISSING'}")
sys.path.insert(0, str(PIPELINE))

try:
    import sam3_features  # noqa: F401
    print("  sam3_features.py        OK")
except Exception as e:
    print(f"  sam3_features.py        ERROR: {e}")

try:
    import cody2_utils
    print("  cody2_utils.py          OK")
    for fn in ['compute_features_1d', 'resolve_device',
               'sample_rows_ratio_keep_controls', 'to_numeric_any',
               'label_to_known_binary', 'make_label_summary']:
        print(f"    has {fn:<36s}: {hasattr(cody2_utils, fn)}")
except Exception as e:
    print(f"  cody2_utils.py          ERROR: {e}")

try:
    from tabicl import TabICLClassifier
    print("  tabicl                  OK")
except Exception as e:
    print(f"  tabicl                  ERROR: {e}")

# [5] Merged training data (optional)
if args.merged_root is not None:
    print("\n[5] Merged training data")
    MERGED = args.merged_root
    print(f"  Path: {MERGED}")
    print(f"  Exists: {MERGED.exists()}")

    if MERGED.exists():
        expected = ["dataset_lc", "dataset_dd", "dataset_consensus",
                    "dataset_action", "dataset_posture", "dataset_rest"]
        for ds in expected:
            d = MERGED / ds
            if d.exists():
                n = len(list(d.glob("*.xlsx")))
                print(f"    {ds:<20s}  {n} files")
            else:
                print(f"    {ds:<20s}  MISSING")
        total = len(list(MERGED.rglob("*.xlsx")))
        print(f"  TOTAL : {total} files (expected: 137)")
else:
    print("\n[5] Merged training data: skipped (pass --merged_root to check)")

# [6] TabICL GPU smoke test
print("\n[6] TabICL GPU smoke test")
try:
    import numpy as np
    from tabicl import TabICLClassifier
    X = np.random.rand(80, 10).astype("float32")
    y = np.random.randint(0, 2, 80)
    clf = TabICLClassifier(device="cuda", n_estimators=4, batch_size=16)
    clf.fit(X, y)
    proba = clf.predict_proba(X[:5])
    print("  TabICL fit + predict_proba OK on GPU")
    print(f"  Sample probas: {proba[0]}")
except Exception as e:
    print(f"  ERROR: {e}")

# [7] Smoke test of cody2_utils.compute_features_1d
print("\n[7] cody2_utils.compute_features_1d smoke test")
try:
    import cody2_utils
    import numpy as np
    sig = np.random.randn(300)
    feats = cody2_utils.compute_features_1d(sig, fps=30.0, robust_norm=True)
    print(f"  Returned {len(feats)} descriptors (expected 19)")
    print(f"  Keys: {sorted(feats.keys())[:5]} ... {sorted(feats.keys())[-3:]}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 70)
if missing:
    print(f"MISSING PACKAGES: {missing}")
    print(f"  Run: pip install {' '.join(missing)}")
else:
    print("Setup OK — ready for training")
print("=" * 70)
