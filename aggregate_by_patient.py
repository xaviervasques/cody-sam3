#!/usr/bin/env python3
"""
aggregate_by_patient.py
=======================

After ``merge_sam3_labels.py`` has produced one merged XLSX per source video,
this script concatenates all the videos of a same patient into a single
per-patient file, matching the cody-2 paper's data layout.

For each ``merged/<tag>/<stem>_merged.xlsx`` file we infer the patient ID from
its filename or from its ``subject_id`` column (preferred), then concatenate
all per-video files sharing the same patient ID into one
``merged_by_patient/<tag>/<patient_id>.xlsx``.

Important: to keep every ``(patient_id, From, To)`` triplet unique across
concatenated videos (so cody-2's ``groupby([patient_id, From, To])`` works
unambiguously), we apply a **time offset** to ``From`` and ``To`` of every
successive video for the same patient. The first video keeps its original
times; the second is shifted by (max_To_of_video_1 + 1.0); etc.

A ``__source_video`` column is added to every row so the original video
can still be traced.

Naming
------
- For files whose stem comes from a date-time (``20231005_114736``) we use
  the ``subject_id`` field from the SAM 3 part of the merged file (which is
  the parent folder name, e.g. ``P1``).
- For files whose stem already contains the patient ID
  (``P1_1Repos``, ``C_1``, etc.) we extract it via a regex on the stem
  (``P\\d+`` / ``C\\d+``).

Inputs
------
  --merged_root      Directory produced by ``merge_sam3_labels.py``
                     (contains ``dataset_lc/``, ``dataset_dd/`` etc.).
  --output_root      Where to write the per-patient files
                     (mirrors the dataset_* layout of ``--merged_root``).

Outputs
-------
  output_root/<tag>/<patient_id>.xlsx          one per (tag, patient)
  output_root/_aggregation_summary.csv         what went into each file
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("aggregate_by_patient")

# --- patient ID extraction ---------------------------------------------------

# Matches "P1", "P12", "C1", "C_1", "C12", "P1_RPA", "P1_1Repos", etc.
# Anchored: the patient marker must be at the START of the string. After the
# digits, we accept either end-of-string, an underscore, or a word boundary.
PATIENT_ID_RE = re.compile(r"^([PC])_?(\d+)(?:_|$|\b)", re.IGNORECASE)


def patient_id_from_row(df: pd.DataFrame, fallback_stem: str) -> str:
    """Find the patient ID from the merged file: prefer subject_id column."""
    if "subject_id" in df.columns and df["subject_id"].notna().any():
        sid_raw = str(df["subject_id"].dropna().iloc[0]).strip()
        m = PATIENT_ID_RE.search(sid_raw)
        if m:
            return f"{m.group(1).upper()}{int(m.group(2))}"
    # Fallback: match the stem (filename without _merged.xlsx)
    m = PATIENT_ID_RE.search(fallback_stem)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2))}"
    return fallback_stem.upper()  # last resort, used as-is


# --- core ---------------------------------------------------------------------

def list_merged_files_per_tag(merged_root: Path) -> Dict[str, List[Path]]:
    """Map dataset_tag -> list of merged xlsx file paths."""
    out: Dict[str, List[Path]] = defaultdict(list)
    for p in merged_root.rglob("*_merged.xlsx"):
        if "__MACOSX" in str(p) or p.name.startswith("~$"):
            continue
        tag = p.parent.name.lower()
        if not tag.startswith("dataset_"):
            log.warning(f"Skipping {p} (parent {tag!r} is not a dataset_* folder)")
            continue
        out[tag].append(p)
    return out


def group_by_patient(file_list: List[Path]) -> Dict[str, List[Tuple[str, Path]]]:
    """Group merged files by inferred patient_id; return {pid: [(stem, path), ...]}."""
    g: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for p in file_list:
        stem = p.name.replace("_merged.xlsx", "")
        # Quick peek: read first row only to grab subject_id (saves time)
        try:
            df = pd.read_excel(p, nrows=2, engine="openpyxl")
        except Exception as e:
            log.error(f"Could not read {p}: {e}")
            continue
        pid = patient_id_from_row(df, stem)
        g[pid].append((stem, p))
    # Sort the videos within each patient by stem (lexicographic, typically
    # chronological since stems start with timestamps like 20231005_HHMMSS)
    for pid in g:
        g[pid].sort(key=lambda x: x[0])
    return g


def aggregate_one_patient(
    pid: str,
    pairs: List[Tuple[str, Path]],
    video_offset: float = 1000.0,
) -> pd.DataFrame:
    """Concatenate all merged dataframes of one patient with per-video offsets.

    cody-2 stores window boundaries in a ``M.SS`` "minute-and-window" format
    (e.g. ``From=0.01`` means minute 0 / window 1 / 0-10s ; ``From=1.21`` means
    minute 1 / window 3 / 80-90s). This is NOT a simple seconds-based time
    axis. Adding ``max_to + 1`` would produce values like ``16.52`` that are
    invalid in the ``M.SS`` grammar.

    Instead we apply a **large per-video offset** (default 1000) that keeps
    the ``M.SS`` semantics intact: video k's windows live in the
    ``k * video_offset .. (k+1) * video_offset`` band. Since cody-2 only uses
    ``(patient_id, From, To)`` for groupby (never for time arithmetic), the
    triplets remain unique and the downstream pipeline is unaffected.
    """
    out_chunks = []

    for k, (stem, path) in enumerate(pairs):
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            log.error(f"  [{pid}] could not read {path}: {e}")
            continue
        if len(df) == 0:
            log.warning(f"  [{pid}] empty file: {path}")
            continue

        # Robust numeric coercion for From / To (cody-2 stores French
        # "0,1" sometimes).
        for c in ("From", "To"):
            if c in df.columns:
                df[c] = pd.to_numeric(
                    df[c].astype("string").str.replace(",", ".", regex=False),
                    errors="coerce",
                )
            else:
                df[c] = np.nan

        # Apply the per-video offset (preserves M.SS semantics within the
        # band, just shifts each video into its own band)
        offset = k * float(video_offset)
        if offset > 0:
            df["From"] = df["From"] + offset
            df["To"]   = df["To"] + offset

        # Pre-allocate the two metadata columns at once (avoids the
        # "DataFrame is highly fragmented" warning that pandas emits when
        # you insert columns one by one into a wide dataframe).
        meta = pd.DataFrame({
            "__source_video": [stem] * len(df),
            "__patient_id":   [pid]  * len(df),
        }, index=df.index)
        df = pd.concat([df, meta], axis=1).copy()

        out_chunks.append(df)

    if not out_chunks:
        return pd.DataFrame()
    return pd.concat(out_chunks, axis=0, ignore_index=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate per-video merged files into per-patient files."
    )
    ap.add_argument("--merged_root", type=Path, required=True,
                    help="Directory produced by merge_sam3_labels.py.")
    ap.add_argument("--output_root", type=Path, required=True,
                    help="Where to write the per-patient files.")
    ap.add_argument("--video_offset", type=float, default=1000.0,
                    help="Numeric offset added to From/To for the k-th video "
                         "of a patient (default: 1000, which preserves the "
                         "M.SS minute.window format that cody-2 uses).")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)

    files_per_tag = list_merged_files_per_tag(args.merged_root)
    log.info(f"Found {sum(len(v) for v in files_per_tag.values())} "
             f"merged files across {len(files_per_tag)} dataset tags")

    summary_rows = []

    for tag, files in sorted(files_per_tag.items()):
        log.info(f"\n=== {tag}: {len(files)} files ===")
        per_patient = group_by_patient(files)
        log.info(f"   -> grouped into {len(per_patient)} patients")

        tag_dir = args.output_root / tag
        tag_dir.mkdir(parents=True, exist_ok=True)

        for pid in sorted(per_patient.keys()):
            pairs = per_patient[pid]
            merged = aggregate_one_patient(pid, pairs, args.video_offset)
            if merged.empty:
                log.warning(f"  [{pid}] no rows after aggregation; skipped")
                continue
            out_path = tag_dir / f"{pid}.xlsx"
            try:
                merged.to_excel(out_path, index=False)
                log.info(f"  WROTE {out_path}  ({len(merged)} rows from "
                         f"{len(pairs)} video(s))")
            except Exception as e:
                log.error(f"  [{pid}] could not write {out_path}: {e}")
                summary_rows.append(dict(
                    tag=tag, patient_id=pid,
                    n_videos=len(pairs), n_rows=int(len(merged)),
                    status=f"write_failed: {e}",
                    output=""
                ))
                continue

            summary_rows.append(dict(
                tag=tag, patient_id=pid,
                n_videos=len(pairs), n_rows=int(len(merged)),
                source_videos=";".join(s for s, _ in pairs),
                status="ok",
                output=str(out_path),
            ))

    if summary_rows:
        s = pd.DataFrame(summary_rows)
        sp = args.output_root / "_aggregation_summary.csv"
        s.to_csv(sp, index=False)
        print(f"\n=== aggregate_by_patient summary ===")
        print(s.groupby(["tag", "status"]).size().unstack(fill_value=0))
        print(f"Summary saved to: {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
