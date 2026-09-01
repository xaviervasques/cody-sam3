#!/usr/bin/env python3
"""
merge_sam3_labels.py
====================

Merge SAM 3 dense per-frame time-series with the cody-2 clinician annotations
(labels + From/To window bounds) into a single training-ready XLSX file per
video.

Inputs
------
  --sam3_root       Directory containing SAM 3 outputs, one subdir per subject
                    (e.g. outputs/P1_RPA/<stem>_sam3_timeseries.xlsx).
  --labels_root     Directory containing cody-2 labelled files. Standard
                    cody-2 layout, with at least:
                       <labels_root>/dataset_lc/<subject>/*.xlsx
                       <labels_root>/dataset_dd/<subject>/*.xlsx
                    Optionally also dataset_action / dataset_rest /
                    dataset_posture (short clips, single rater).
  --output_root     Where to write the merged files. Will produce one subdir
                    per dataset tag (dataset_lc, dataset_dd, dataset_consensus,
                    dataset_action, dataset_rest, dataset_posture).

Pairing rule
------------
The SAM 3 file `<dir>/<stem>_sam3_timeseries.xlsx` is paired with the cody-2
file `<dir2>/<stem>_merged.xlsx` (where <stem> is identical). The two files
are guaranteed to have the same number of rows because they were processed
from the same source video frame-by-frame.

What is propagated
------------------
From the SAM 3 file we keep:
  - subject_id, subject_type, video_relpath, time_s, frame_idx,
    patient_detected
  - the 14 geometric descriptors
  - all 64 contour points x 2 (= 128 columns)
  - all 96 grid points x 2 (= 192 columns)

From the cody-2 file we add:
  - From, To (cleaned via to_numeric_any: French "0,1" -> 0.1)
  - the eight target labels (Dystonia, Tremor, Myoclonus, Chorea,
    Athetosis, Ballismus, Stereotypies, Tics) with values in {0, 1, 2}
  - the optional extra labels (Bradykinesia, FOG/Fog, Ataxia)

The merged file is written WITHOUT derived signals (posture / regional).
Those are recomputed on the fly by train_sam3 / inference_sam3 according to
the user-selected tier (1, 2, 3). This keeps the merged file as a single
source of truth: the same merged file can be used for any tier.

Consensus output
----------------
When both `dataset_lc/<sid>/<stem>_merged.xlsx` and
`dataset_dd/<sid>/<stem>_merged.xlsx` exist, we also produce a third merged
file in `dataset_consensus/`. For each (window, label):
    - LC==DD==1  -> 1
    - LC==DD==0  -> 0
    - any other case (including disagreement, NaN, or any value of 2)
                 -> 2 (uncertain), so train_sam3 will exclude that
                   (window, label) pair from training. Other labels of the
                   same window where LC and DD agree remain usable.

Context tag deduction
---------------------
The dataset tag is taken from the parent directory of the cody-2 file
(dataset_lc / dataset_dd / dataset_action / dataset_rest / dataset_posture).
This is the same convention as cody-2's `dataset_tag_from_path`.

Usage
-----
    python merge_sam3_labels.py \\
        --sam3_root   /path/to/sam3/outputs \\
        --labels_root /path/to/cody2/training \\
        --output_root /path/to/merged
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_SYMPTOMS = [
    "Dystonia", "Tremor", "Myoclonus", "Chorea",
    "Athetosis", "Ballismus", "Stereotypies", "Tics",
]
EXTRA_LABELS = ["Bradykinesia", "FOG", "Fog", "Ataxia"]

LABEL_COLUMNS = DEFAULT_SYMPTOMS + EXTRA_LABELS
META_COLS = ["subject_id", "subject_type", "video_relpath",
             "time_s", "frame_idx", "patient_detected"]

# Cody-2 dataset folder conventions
SUPPORTED_DATASET_TAGS = {
    "dataset_lc", "dataset_dd",
    "dataset_action", "dataset_rest", "dataset_posture",
}

SAM3_SUFFIX = "_sam3_timeseries.xlsx"
LABEL_SUFFIX = "_merged.xlsx"

log = logging.getLogger("merge_sam3_labels")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def to_numeric_any(x: pd.Series) -> pd.Series:
    """Convert numeric-like strings (with optional comma decimals) to float.

    Mirrors cody-2's `to_numeric_any` so that French-locale "0,1" -> 0.1.
    """
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce")
    return pd.to_numeric(
        x.astype("string").str.replace(",", ".", regex=False),
        errors="coerce",
    )


def find_sam3_files(sam3_root: Path) -> Dict[str, Path]:
    """Index SAM 3 timeseries files by stem (without the suffix)."""
    by_stem: Dict[str, Path] = {}
    for p in sam3_root.rglob(f"*{SAM3_SUFFIX}"):
        if "__MACOSX" in str(p) or p.name.startswith("~$"):
            continue
        stem = p.name[:-len(SAM3_SUFFIX)]
        by_stem.setdefault(stem, p)  # keep the first match
    return by_stem


def find_label_files(labels_root: Path) -> List[Tuple[str, Path, str]]:
    """Find cody-2 labelled files.

    Returns a list of (stem, path, dataset_tag) tuples.
    The dataset tag is the name of the nearest ancestor folder that matches
    one of SUPPORTED_DATASET_TAGS; "dataset_unknown" if none is found.
    """
    out: List[Tuple[str, Path, str]] = []
    for p in labels_root.rglob(f"*{LABEL_SUFFIX}"):
        if "__MACOSX" in str(p) or p.name.startswith("~$"):
            continue
        stem = p.name[:-len(LABEL_SUFFIX)]
        tag = "dataset_unknown"
        for parent in p.parents:
            if parent.name.lower() in SUPPORTED_DATASET_TAGS:
                tag = parent.name.lower()
                break
        out.append((stem, p, tag))
    return out


def safe_read_excel(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl", **kwargs)


def normalize_label_value(v) -> float:
    """Coerce a label value into a float in {0, 1, 2}. Anything else -> NaN."""
    if pd.isna(v):
        return np.nan
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    if x not in (0.0, 1.0, 2.0):
        return np.nan
    return x


def consensus_label_pair(v_lc: float, v_dd: float) -> float:
    """
    Combine two label values from two raters according to the consensus rule:
        agree on 0 -> 0
        agree on 1 -> 1
        disagree, or any 2/NaN -> 2 (uncertain)
    Returns float for consistency with NaN handling.
    """
    a = normalize_label_value(v_lc)
    b = normalize_label_value(v_dd)
    if a == 0 and b == 0:
        return 0.0
    if a == 1 and b == 1:
        return 1.0
    return 2.0


# -----------------------------------------------------------------------------
# Core merge
# -----------------------------------------------------------------------------

def merge_one_file(
    sam3_path: Path, label_path: Path, dataset_tag: str,
) -> pd.DataFrame:
    """Pair one SAM 3 file with one cody-2 file by row index.

    The two files were produced from the same source video and must have the
    same number of rows; if they do not, an error is raised.
    """
    df_sam3 = safe_read_excel(sam3_path)
    df_lab = safe_read_excel(label_path)

    if len(df_sam3) != len(df_lab):
        raise ValueError(
            f"Row count mismatch between SAM 3 file ({len(df_sam3)}) "
            f"and label file ({len(df_lab)}) for stem "
            f"'{sam3_path.name}' / '{label_path.name}'."
        )

    # ------------------------------------------------------------------
    # Start from the SAM 3 dataframe (it already has all signals + meta).
    # ------------------------------------------------------------------
    merged = df_sam3.copy()

    # ------------------------------------------------------------------
    # Add From / To, cleaned of French decimals.
    # ------------------------------------------------------------------
    for required in ("From", "To"):
        if required not in df_lab.columns:
            raise ValueError(
                f"Required column '{required}' is missing from {label_path}"
            )
    merged["From"] = to_numeric_any(df_lab["From"]).values
    merged["To"] = to_numeric_any(df_lab["To"]).values

    # ------------------------------------------------------------------
    # Add label columns. Coerce values to {0, 1, 2}; everything else -> NaN.
    # ------------------------------------------------------------------
    for col in LABEL_COLUMNS:
        if col in df_lab.columns:
            merged[col] = pd.to_numeric(df_lab[col], errors="coerce").values
        # We do not invent missing labels; downstream `train_sam3` will
        # simply skip them.

    # ------------------------------------------------------------------
    # Normalise the case of Fog -> FOG so downstream code does not have to
    # worry about the alternative spelling we observed in some files.
    # ------------------------------------------------------------------
    if "Fog" in merged.columns and "FOG" not in merged.columns:
        merged.rename(columns={"Fog": "FOG"}, inplace=True)

    # ------------------------------------------------------------------
    # Inject the dataset tag (informational only; the parent folder is the
    # source of truth for train_sam3's context flags).
    # ------------------------------------------------------------------
    merged["__dataset_tag"] = dataset_tag

    return merged


def build_consensus(
    df_lc: pd.DataFrame, df_dd: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the consensus merged dataframe from a paired LC / DD merged df.
    The two inputs must come from the same SAM 3 file (so they share the
    SAM 3 signals); only the labels and From/To are reconciled.

    For each label column present in either input:
      - both agree on 0 -> 0
      - both agree on 1 -> 1
      - everything else -> 2 (uncertain)

    For From / To, we trust the LC file's values (they should be identical
    to DD anyway since the window definitions are typically shared; if they
    are not, the LC values are kept for reproducibility).
    """
    if len(df_lc) != len(df_dd):
        raise ValueError("LC and DD merged dfs have different row counts")

    # Start from LC's signals + From/To (they share SAM 3 signals).
    df = df_lc.copy()
    df["__dataset_tag"] = "dataset_consensus"

    # Reconcile labels label-by-label.
    for col in LABEL_COLUMNS:
        col_norm = "FOG" if col == "Fog" else col
        if col_norm not in df_lc.columns and col_norm not in df_dd.columns:
            continue
        v_lc = df_lc[col_norm] if col_norm in df_lc.columns else pd.Series(
            [np.nan] * len(df_lc)
        )
        v_dd = df_dd[col_norm] if col_norm in df_dd.columns else pd.Series(
            [np.nan] * len(df_dd)
        )
        out = np.array([
            consensus_label_pair(a, b) for a, b in zip(v_lc, v_dd)
        ], dtype=float)
        df[col_norm] = out

    return df


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge SAM 3 timeseries with cody-2 labels."
    )
    parser.add_argument("--sam3_root", type=Path, required=True,
                        help="Directory with SAM 3 *_sam3_timeseries.xlsx files.")
    parser.add_argument("--labels_root", type=Path, required=True,
                        help="Directory with cody-2 *_merged.xlsx files.")
    parser.add_argument("--output_root", type=Path, required=True,
                        help="Where to write the merged output files.")
    parser.add_argument("--no_consensus", action="store_true",
                        help="Skip the consensus dataset_consensus/ output.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Index SAM 3 files.
    # ------------------------------------------------------------------
    sam3_by_stem = find_sam3_files(args.sam3_root)
    log.info(f"Found {len(sam3_by_stem)} SAM 3 files in {args.sam3_root}")

    # ------------------------------------------------------------------
    # 2) Index cody-2 label files.
    # ------------------------------------------------------------------
    label_files = find_label_files(args.labels_root)
    log.info(f"Found {len(label_files)} cody-2 label files in {args.labels_root}")

    # Group by stem -> by dataset_tag
    by_stem_tag: Dict[str, Dict[str, Path]] = {}
    for stem, p, tag in label_files:
        by_stem_tag.setdefault(stem, {})[tag] = p

    # ------------------------------------------------------------------
    # 3) Process each stem.
    # ------------------------------------------------------------------
    summary_rows = []
    n_total, n_ok, n_missing_sam3, n_failed = 0, 0, 0, 0
    n_consensus = 0

    for stem in sorted(by_stem_tag.keys()):
        sam3_path = sam3_by_stem.get(stem)
        if sam3_path is None:
            n_missing_sam3 += 1
            log.warning(f"No SAM 3 file for stem '{stem}'; skipping.")
            for tag in by_stem_tag[stem]:
                summary_rows.append(dict(
                    stem=stem, tag=tag,
                    status="missing_sam3", output=""
                ))
            continue

        # Cache merged frames per tag so we can compute the consensus later
        # without re-reading the SAM 3 file.
        merged_per_tag: Dict[str, pd.DataFrame] = {}

        for tag, label_path in sorted(by_stem_tag[stem].items()):
            n_total += 1
            try:
                merged = merge_one_file(sam3_path, label_path, tag)
            except Exception as e:
                n_failed += 1
                log.error(f"FAILED stem={stem} tag={tag}: {e}")
                summary_rows.append(dict(
                    stem=stem, tag=tag,
                    status=f"failed: {e}", output=""
                ))
                continue

            merged_per_tag[tag] = merged
            out_dir = args.output_root / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}{LABEL_SUFFIX}"
            try:
                merged.to_excel(out_path, index=False)
                n_ok += 1
                summary_rows.append(dict(
                    stem=stem, tag=tag, status="ok", output=str(out_path)
                ))
                log.info(f"WROTE {out_path}")
            except Exception as e:
                n_failed += 1
                log.error(f"Could not write {out_path}: {e}")
                summary_rows.append(dict(
                    stem=stem, tag=tag,
                    status=f"write_failed: {e}", output=""
                ))

        # ----------------------------------------------------------------
        # Consensus: produce only when BOTH dataset_lc and dataset_dd
        # variants exist for this stem.
        # ----------------------------------------------------------------
        if (not args.no_consensus
                and "dataset_lc" in merged_per_tag
                and "dataset_dd" in merged_per_tag):
            try:
                cons = build_consensus(
                    merged_per_tag["dataset_lc"],
                    merged_per_tag["dataset_dd"],
                )
                out_dir = args.output_root / "dataset_consensus"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{stem}{LABEL_SUFFIX}"
                cons.to_excel(out_path, index=False)
                n_consensus += 1
                summary_rows.append(dict(
                    stem=stem, tag="dataset_consensus",
                    status="ok", output=str(out_path)
                ))
                log.info(f"WROTE {out_path}  (consensus)")
            except Exception as e:
                log.error(f"Consensus FAILED stem={stem}: {e}")
                summary_rows.append(dict(
                    stem=stem, tag="dataset_consensus",
                    status=f"failed: {e}", output=""
                ))

    # ------------------------------------------------------------------
    # 4) Summary
    # ------------------------------------------------------------------
    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_root / "_merge_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n=== merge_sam3_labels summary ===")
    print(f"Total label files processed : {n_total}")
    print(f"  Merged OK                 : {n_ok}")
    print(f"  Consensus extras          : {n_consensus}")
    print(f"  Missing SAM 3 file        : {n_missing_sam3}")
    print(f"  Failed                    : {n_failed}")
    print(f"Summary saved to            : {summary_path}")
    if n_failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
