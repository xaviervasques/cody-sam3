#!/usr/bin/env python3
"""
sam3_features.py
================

Feature engineering for SAM 3 segmentation-based time-series, designed for
hyperkinetic movement disorder phenotyping with TabICLv2.

This module defines what is meant by *signal* in the SAM 3 setting and provides
three feature schemes ("tiers") with increasing capacity:

Tier 1 (minimal, ~26 signals)
    14 raw geometric descriptors of the patient silhouette
    + 12 posture-derived signals computed from the silhouette contour.

Tier 2 (recommended, ~60-70 signals)
    Tier 1 + anatomical regional aggregations of the contour and interior grid.
    The contour is partitioned into 6 anatomical zones (head, L/R shoulder area,
    L/R arm area, pelvis area, legs area). The interior grid is partitioned
    into 4 regions (top, middle, bottom-left, bottom-right). For each region we
    compute regional kinetic energy, radial dispersion, mean velocity and
    dominant angle.

Tier 3 (kitchen sink, ~334 signals)
    All raw SAM 3 signals: 14 geometric + 128 contour coordinates + 192 grid
    coordinates. Maximum information, highest overfitting risk on small cohorts.

Each "signal" here is a per-frame time-series. The downstream module (train_sam3)
then runs the standard 19 cody-2 statistical descriptors (mean, std, min, max,
median, range, skew, kurtosis, energy, slope, iqr, entropy, var, fft_peak_freq,
fft_peak_amp, zero_crossings_delta, abs_accel_mean, higuchi_fd, perm_entropy)
on every signal within every window.

Clinical rationale for each feature category is documented inline next to its
construction.
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ===========================================================================
# Constants: column conventions in SAM 3 output XLSX files
# ===========================================================================

META_COLS = {
    "subject_id", "subject_type", "video_relpath",
    "time_s", "frame_idx", "patient_detected",
}

GEOMETRIC_COLS = [
    "centroid_x", "centroid_y",
    "area_px", "perimeter_px",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "aspect_ratio", "solidity", "extent",
    "orientation_deg", "major_axis_px", "minor_axis_px",
]

CONTOUR_COL_RE = re.compile(r"^contour_(\d+)_([xy])$")
GRID_COL_RE = re.compile(r"^grid_r(\d+)_c(\d+)_([xy])$")


def list_contour_points(df: pd.DataFrame) -> List[int]:
    """Return sorted list of contour point indices present in df."""
    idx = set()
    for c in df.columns:
        m = CONTOUR_COL_RE.match(c)
        if m:
            idx.add(int(m.group(1)))
    return sorted(idx)


def list_grid_cells(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Return sorted list of (row, col) grid cell coordinates present in df."""
    cells = set()
    for c in df.columns:
        m = GRID_COL_RE.match(c)
        if m:
            cells.add((int(m.group(1)), int(m.group(2))))
    return sorted(cells)


# ===========================================================================
# Anatomical region definitions
# ===========================================================================
# The SAM 3 contour is 64 points sampled by arc length, anchored at the top
# of the head (index 0). Going clockwise around the body, the indices roughly
# map onto: head -> right side (down the right arm, hip, leg) -> feet ->
# left leg, hip, arm -> back to head.
#
# We split the 64 indices into 6 anatomical bins. Boundaries are approximate
# (the actual proportions depend on body posture) but stable enough to be
# clinically meaningful: changes in any one bin correspond to specific body
# regions.
#
# Index 0 = top of head.
# Going CW around the silhouette:
#   - head:           indices around 0 (modulo wrap-around)
#   - right shoulder/arm
#   - right hip and leg
#   - feet area (bottom-most)
#   - left hip and leg
#   - left shoulder/arm
# The contour is closed: index 63 is adjacent to index 0.
# Boundaries below assume N=64 (the default in our SAM 3 notebook).

CONTOUR_REGIONS_DEFAULT: Dict[str, Tuple[int, int]] = {
    # name           : (start_idx_inclusive, end_idx_exclusive)
    "head":             (60, 4),     # wraps around 0
    "right_shoulder":   (4, 12),
    "right_arm":        (12, 22),
    "feet_right":       (22, 32),
    "feet_left":        (32, 42),
    "left_arm":         (42, 52),
    "left_shoulder":    (52, 60),
}


def _expand_wrap_range(start: int, end: int, n: int) -> List[int]:
    """Expand a (start, end) range that may wrap around 0..n-1."""
    if start < end:
        return list(range(start, end))
    return list(range(start, n)) + list(range(0, end))


def get_contour_region_indices(
    point_indices: List[int],
    regions: Dict[str, Tuple[int, int]] = CONTOUR_REGIONS_DEFAULT,
) -> Dict[str, List[int]]:
    """Map each region name to the list of contour point indices it covers."""
    if not point_indices:
        return {k: [] for k in regions}
    n = max(point_indices) + 1
    mapped = {}
    for name, (a, b) in regions.items():
        idxs = _expand_wrap_range(a, b, n)
        mapped[name] = [i for i in idxs if i in point_indices]
    return mapped


# Grid: 12 rows x 8 cols = 96 cells. Top row = upper body, bottom row = lower.
GRID_REGIONS_DEFAULT = {
    # name      : (row_min, row_max_excl, col_min, col_max_excl)
    "top":          (0, 3, 0, 8),    # upper third (head + shoulders)
    "middle":       (3, 8, 0, 8),    # trunk + upper arms + hips
    "bottom_left":  (8, 12, 0, 4),   # left leg
    "bottom_right": (8, 12, 4, 8),   # right leg
}


def get_grid_region_cells(
    grid_cells: List[Tuple[int, int]],
    regions: Dict[str, Tuple[int, int, int, int]] = GRID_REGIONS_DEFAULT,
) -> Dict[str, List[Tuple[int, int]]]:
    """Map each region name to the list of (row, col) cells it covers."""
    mapped = {}
    for name, (rmin, rmax, cmin, cmax) in regions.items():
        mapped[name] = [
            (r, c) for (r, c) in grid_cells
            if rmin <= r < rmax and cmin <= c < cmax
        ]
    return mapped


# ===========================================================================
# Posture-derived signals (Tier 1)
# ===========================================================================
# The contour gives us a discretised body outline; we derive simple posture
# proxies that are robust to keypoint absence (which YOLO would have provided).
#
# Clinical rationale, per signal:
#   - shoulder_proxy_height_asymmetry: surrogate for the cody-2 "shoulder
#     height diff" feature. Sensitive to dystonic postures with shoulder
#     elevation on one side, or to head tilt towards one shoulder.
#   - hip_proxy_height_asymmetry: surrogate for pelvic tilt, indirect signal
#     for trunk dystonia.
#   - vertical_extent: total head-to-foot extent of the silhouette in pixels.
#     Sensitive to global posture decomposition (slumped or collapsed posture)
#     and to crouching, particularly relevant for dystonia and parkinsonism.
#   - body_axis_orientation: principal axis angle (similar to SAM 3's
#     `orientation_deg` but computed from contour-region centroids rather than
#     pixel moments). Sensitive to trunk inclination.
#   - silhouette_lateral_swing: horizontal distance between the centroid of
#     the head region and the centroid of the feet region. Trace of titubation,
#     swaying, ataxic movements.
#   - head_movement_amplitude: instantaneous distance of the head-region
#     centroid from its own median position over the whole video; will burst
#     during head tics, head bobbing, head dystonia.
#   - arm_swing_left / arm_swing_right: distance of left/right "arm region"
#     centroid from its own median. Trace of arm tremor, chorea, ballism,
#     dystonic arm posturing.
#   - leg_movement_left / leg_movement_right: same for legs. Trace of
#     restless-leg, lower-limb chorea, foot dystonia.
#   - relative_torsion: angle between the head-region axis and the feet-region
#     axis, an approximate trunk torsion measure (proxy for dystonia and
#     ataxia).
#   - silhouette_complexity: ratio of (perimeter^2) / (4 * pi * area). Equals
#     1 for a perfect disc; grows as the silhouette becomes more concave or
#     fragmented (sensitive to limb spreading, asymmetric dystonic postures).

POSTURE_SIGNAL_NAMES = [
    "posture__shoulder_proxy_height_asym",
    "posture__hip_proxy_height_asym",
    "posture__vertical_extent",
    "posture__body_axis_orientation",
    "posture__silhouette_lateral_swing",
    "posture__head_movement_amplitude",
    "posture__arm_swing_left",
    "posture__arm_swing_right",
    "posture__leg_movement_left",
    "posture__leg_movement_right",
    "posture__relative_torsion",
    "posture__silhouette_complexity",
]


def _region_centroid_xy(
    df: pd.DataFrame,
    point_idxs: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean (x, y) over a list of contour points, per frame, ignoring NaN."""
    if not point_idxs:
        n = len(df)
        return np.full(n, np.nan), np.full(n, np.nan)
    xs = np.column_stack([df[f"contour_{i:03d}_x"].to_numpy(dtype=float)
                          for i in point_idxs])
    ys = np.column_stack([df[f"contour_{i:03d}_y"].to_numpy(dtype=float)
                          for i in point_idxs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cx = np.nanmean(xs, axis=1)
        cy = np.nanmean(ys, axis=1)
    return cx, cy


def add_posture_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the 12 Tier-1 posture-derived signals to df. The input dataframe must
    contain SAM 3 contour columns (`contour_NNN_x/y`) and the geometric
    columns; columns that cannot be computed are filled with NaN.
    """
    d = df.copy()
    pts = list_contour_points(d)
    if not pts:
        for name in POSTURE_SIGNAL_NAMES:
            d[name] = np.nan
        return d

    regions = get_contour_region_indices(pts)

    # Per-region centroids
    centroids = {
        name: _region_centroid_xy(d, idxs)
        for name, idxs in regions.items()
    }

    head_x, head_y       = centroids["head"]
    rsh_x, rsh_y         = centroids["right_shoulder"]
    lsh_x, lsh_y         = centroids["left_shoulder"]
    rarm_x, rarm_y       = centroids["right_arm"]
    larm_x, larm_y       = centroids["left_arm"]
    rfeet_x, rfeet_y     = centroids["feet_right"]
    lfeet_x, lfeet_y     = centroids["feet_left"]

    feet_x = np.nanmean(np.column_stack([rfeet_x, lfeet_x]), axis=1)
    feet_y = np.nanmean(np.column_stack([rfeet_y, lfeet_y]), axis=1)

    # ------------------------------------------------------------------
    # Shoulder/hip asymmetry surrogates
    # The contour does not give us the true joint position, but it gives the
    # most lateral point at the shoulder altitude. The y-difference between
    # the two "shoulder region" centroids is a reasonable surrogate.
    # ------------------------------------------------------------------
    # Normalize by bbox height to make the signal comparable across patients
    # and recording distances.
    bbox_h = pd.to_numeric(d.get("bbox_h"), errors="coerce").to_numpy(float)
    bbox_h = np.where(np.isfinite(bbox_h) & (bbox_h > 1e-6), bbox_h, np.nan)
    bbox_w = pd.to_numeric(d.get("bbox_w"), errors="coerce").to_numpy(float)
    bbox_w = np.where(np.isfinite(bbox_w) & (bbox_w > 1e-6), bbox_w, np.nan)

    d["posture__shoulder_proxy_height_asym"] = (rsh_y - lsh_y) / bbox_h
    # The "hip" proxy uses the topmost of feet regions, which approximates
    # the lower trunk - a rough but consistent surrogate of pelvic tilt.
    d["posture__hip_proxy_height_asym"] = (rfeet_y - lfeet_y) / bbox_h

    # ------------------------------------------------------------------
    # Whole-body geometry
    # ------------------------------------------------------------------
    d["posture__vertical_extent"] = bbox_h
    d["posture__body_axis_orientation"] = pd.to_numeric(
        d.get("orientation_deg"), errors="coerce"
    )

    # ------------------------------------------------------------------
    # Lateral swing of trunk axis
    # ------------------------------------------------------------------
    d["posture__silhouette_lateral_swing"] = (head_x - feet_x) / bbox_w

    # ------------------------------------------------------------------
    # Region-level "movement amplitude" = displacement of a region centroid
    # from its own median position. Per-frame, NOT a time-derivative. The
    # 19 statistical descriptors will pick up its distribution and rhythm.
    # ------------------------------------------------------------------
    def _amp(cx, cy):
        cx_med = np.nanmedian(cx)
        cy_med = np.nanmedian(cy)
        return np.sqrt((cx - cx_med) ** 2 + (cy - cy_med) ** 2) / bbox_h

    d["posture__head_movement_amplitude"] = _amp(head_x, head_y)
    d["posture__arm_swing_left"]          = _amp(larm_x, larm_y)
    d["posture__arm_swing_right"]         = _amp(rarm_x, rarm_y)
    d["posture__leg_movement_left"]       = _amp(lfeet_x, lfeet_y)
    d["posture__leg_movement_right"]      = _amp(rfeet_x, rfeet_y)

    # ------------------------------------------------------------------
    # Relative torsion: angle of head -> feet vector
    # ------------------------------------------------------------------
    d["posture__relative_torsion"] = np.degrees(
        np.arctan2(feet_y - head_y, feet_x - head_x)
    )

    # ------------------------------------------------------------------
    # Silhouette complexity (perimeter^2 / (4 pi area))
    # ------------------------------------------------------------------
    area = pd.to_numeric(d.get("area_px"), errors="coerce").to_numpy(float)
    perim = pd.to_numeric(d.get("perimeter_px"), errors="coerce").to_numpy(float)
    area_safe = np.where(np.isfinite(area) & (area > 1.0), area, np.nan)
    d["posture__silhouette_complexity"] = (perim ** 2) / (4 * np.pi * area_safe)

    return d


# ===========================================================================
# Regional aggregations (Tier 2)
# ===========================================================================
# Idea: instead of feeding 64 contour x/y signals or 96 grid x/y signals to
# the downstream descriptor extractor, we summarise each anatomical region
# with a handful of physically meaningful aggregates:
#
#   - region_centroid_x / region_centroid_y: where the region sits, on average.
#   - region_radial_dispersion: how "spread out" the region's points are
#       around the centroid (sensitive to chorea, ballism, athetosis).
#   - region_kinetic_energy: sum of squared frame-to-frame displacements
#       across the region's points (sensitive to tremor and rapid movements).
#   - region_dominant_angle: angle of the longest axis of the region's
#       point cloud (sensitive to dystonic posturing).
#
# 6 contour regions x 4 aggregates + 4 grid regions x 4 aggregates
#  = 24 + 16 = 40 regional signals, on top of Tier 1.
#
# Per signal we again compute the 19 cody-2 statistical descriptors.

REGIONAL_AGGREGATES = [
    "centroid_x", "centroid_y",
    "radial_dispersion", "kinetic_energy",
    "dominant_angle",
]


def _stack_xy(df: pd.DataFrame, x_cols: List[str], y_cols: List[str]
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Stack groups of x/y columns into (T, K) arrays."""
    X = np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in x_cols
    ]) if x_cols else np.zeros((len(df), 0))
    Y = np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in y_cols
    ]) if y_cols else np.zeros((len(df), 0))
    return X, Y


def _regional_aggregates(
    df: pd.DataFrame, x_cols: List[str], y_cols: List[str], region_name: str,
    prefix: str,
) -> Dict[str, np.ndarray]:
    """
    Compute per-frame regional aggregates over a set of points.

    Returns 5 time-series per region: centroid_x, centroid_y,
    radial_dispersion, kinetic_energy, dominant_angle.
    """
    T = len(df)
    out: Dict[str, np.ndarray] = {}
    if not x_cols or not y_cols or len(x_cols) != len(y_cols):
        for agg in REGIONAL_AGGREGATES:
            out[f"{prefix}__{region_name}__{agg}"] = np.full(T, np.nan)
        return out

    X, Y = _stack_xy(df, x_cols, y_cols)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cx = np.nanmean(X, axis=1)
        cy = np.nanmean(Y, axis=1)
        rx = X - cx[:, None]
        ry = Y - cy[:, None]
        radial = np.sqrt(np.nanmean(rx ** 2 + ry ** 2, axis=1))

    # Kinetic energy: sum of squared inter-frame displacements of each point
    # in the region. The 19 descriptors will summarise its time profile.
    KE = np.full(T, np.nan)
    if T > 1:
        dX = np.diff(X, axis=0)
        dY = np.diff(Y, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ke_step = np.nanmean(dX ** 2 + dY ** 2, axis=1)
        KE[1:] = ke_step
        KE[0] = ke_step[0] if ke_step.size else np.nan

    # Dominant angle: PCA on (X - cx, Y - cy) per frame. Vectorised version.
    ang = np.full(T, np.nan)
    for t in range(T):
        xt = rx[t]; yt = ry[t]
        m = np.isfinite(xt) & np.isfinite(yt)
        if m.sum() < 2:
            continue
        xt = xt[m]; yt = yt[m]
        # Covariance components
        sxx = float(np.mean(xt * xt))
        syy = float(np.mean(yt * yt))
        sxy = float(np.mean(xt * yt))
        # Principal direction (angle of dominant eigenvector)
        ang[t] = 0.5 * np.degrees(np.arctan2(2 * sxy, sxx - syy))

    out[f"{prefix}__{region_name}__centroid_x"] = cx
    out[f"{prefix}__{region_name}__centroid_y"] = cy
    out[f"{prefix}__{region_name}__radial_dispersion"] = radial
    out[f"{prefix}__{region_name}__kinetic_energy"] = KE
    out[f"{prefix}__{region_name}__dominant_angle"] = ang
    return out


def add_regional_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add Tier-2 contour-region and grid-region aggregates to df."""
    d = df.copy()

    # ---- contour regions ----
    pts = list_contour_points(d)
    if pts:
        regions = get_contour_region_indices(pts)
        for name, idxs in regions.items():
            x_cols = [f"contour_{i:03d}_x" for i in idxs]
            y_cols = [f"contour_{i:03d}_y" for i in idxs]
            for k, v in _regional_aggregates(
                d, x_cols, y_cols, name, prefix="contour_region",
            ).items():
                d[k] = v

    # ---- grid regions ----
    cells = list_grid_cells(d)
    if cells:
        grid_regions = get_grid_region_cells(cells)
        for name, region_cells in grid_regions.items():
            x_cols = [f"grid_r{r:02d}_c{c:02d}_x" for (r, c) in region_cells]
            y_cols = [f"grid_r{r:02d}_c{c:02d}_y" for (r, c) in region_cells]
            for k, v in _regional_aggregates(
                d, x_cols, y_cols, name, prefix="grid_region",
            ).items():
                d[k] = v
    return d


# ===========================================================================
# Tier selection: enumerate signal columns for a given tier
# ===========================================================================

def get_tier_signal_cols(df: pd.DataFrame, tier: int) -> List[str]:
    """
    Return the list of column names in df to be used as time-series signals
    for the given feature tier. The downstream module (train_sam3) will then
    compute its 19 statistical descriptors on each.

    Tier 1: geometric + posture-derived (~26)
    Tier 2: Tier 1 + regional aggregates (~66)
    Tier 3: all raw SAM 3 signals (~334)
    """
    if tier not in (1, 2, 3):
        raise ValueError(f"tier must be 1, 2 or 3 (got {tier})")

    geom = [c for c in GEOMETRIC_COLS if c in df.columns]
    posture = [c for c in df.columns if c.startswith("posture__")]
    region = [c for c in df.columns
              if c.startswith("contour_region__")
              or c.startswith("grid_region__")]
    raw_contour = sorted(
        c for c in df.columns if CONTOUR_COL_RE.match(c)
    )
    raw_grid = sorted(
        c for c in df.columns if GRID_COL_RE.match(c)
    )

    if tier == 1:
        return geom + posture
    if tier == 2:
        return geom + posture + region
    return geom + raw_contour + raw_grid  # tier 3


def add_derived_signals(df: pd.DataFrame, tier: int) -> pd.DataFrame:
    """
    Add to df the derived signals required by the given tier.

    Tier 1 / 2: posture signals are always added (they are needed by Tier 1).
    Tier 2: regional aggregates are also added.
    Tier 3: nothing is added (uses raw signals only).
    """
    if tier not in (1, 2, 3):
        raise ValueError(f"tier must be 1, 2 or 3 (got {tier})")
    d = df
    if tier in (1, 2):
        d = add_posture_signals(d)
    if tier == 2:
        d = add_regional_signals(d)
    return d


# ===========================================================================
# Convenience: per-signal clinical descriptions (used by analyze_features.py)
# ===========================================================================

SIGNAL_DESCRIPTIONS: Dict[str, str] = {
    # --- Geometric (raw SAM 3) ---
    "centroid_x": "Horizontal centroid of the patient silhouette (pixels). "
                  "Sensitive to whole-body horizontal displacement; useful for "
                  "tremor at the torso level and for ataxic swaying.",
    "centroid_y": "Vertical centroid of the patient silhouette (pixels). "
                  "Sensitive to crouching, slumping or jumping; relevant to "
                  "dystonia, parkinsonism and myoclonus.",
    "area_px":    "Total area of the patient silhouette in pixels. Sensitive "
                  "to posture: shrinks when the patient folds (dystonia), "
                  "grows when limbs are extended.",
    "perimeter_px": "Perimeter of the silhouette. Sensitive to limb position: "
                    "extended limbs add to the perimeter without much area "
                    "change; useful for distinguishing limb-spreading "
                    "dystonia from compact posturing.",
    "bbox_x":     "Top-left x of the bounding box. Tracks horizontal "
                  "position of the patient; mirrors centroid_x but in extreme "
                  "values.",
    "bbox_y":     "Top-left y of the bounding box. Tracks vertical extent of "
                  "the top of the silhouette (head movement, raised arms).",
    "bbox_w":     "Bounding box width. Sensitive to lateral limb spreading; "
                  "useful for ballism and broad dystonic postures.",
    "bbox_h":     "Bounding box height. Tracks crouching vs upright posture, "
                  "extension of limbs upward (raised hand) etc.",
    "aspect_ratio": "Width / height of the bounding box. Captures global body "
                    "shape: tall-thin (standard upright) vs wide (limbs out). "
                    "Sensitive to chorea, ballism, and broad dystonic postures.",
    "solidity":   "Ratio of silhouette area to convex hull area. Close to 1 "
                  "for compact postures; drops when limbs make 'gaps' (raised "
                  "arms, spread legs). Sensitive to dystonic limb posturing.",
    "extent":     "Ratio of silhouette area to bounding box area. Drops as "
                  "the bounding box gets stretched by extreme limb positions; "
                  "relevant to ballism and dystonia.",
    "orientation_deg": "Principal axis orientation of the silhouette (degrees). "
                       "Sensitive to trunk inclination; relevant to dystonia "
                       "(camptocormia, axial tilt) and ataxia.",
    "major_axis_px":   "Major axis length of the silhouette's fitted ellipse. "
                       "Roughly tracks the longest body extent.",
    "minor_axis_px":   "Minor axis length of the silhouette's fitted ellipse.",
    # --- Posture-derived (Tier 1) ---
    "posture__shoulder_proxy_height_asym":
        "Difference of mean y between the right-shoulder and left-shoulder "
        "contour regions, normalised by silhouette height. Sensitive to "
        "shoulder-elevation dystonia and head tilts.",
    "posture__hip_proxy_height_asym":
        "Analogous asymmetry for the hip / lower trunk region. Sensitive to "
        "pelvic tilt and trunk dystonia.",
    "posture__vertical_extent":
        "Height of the silhouette in pixels. Tracks global posture: drops in "
        "stooping/crouching (Parkinsonism, camptocormia) and increases when "
        "raising arms.",
    "posture__body_axis_orientation":
        "Body principal-axis orientation, mirrors orientation_deg.",
    "posture__silhouette_lateral_swing":
        "Horizontal distance between head-region centroid and feet-region "
        "centroid, normalised by silhouette width. Captures trunk-to-feet "
        "lateral offset, a proxy for swaying and lateral truncal dystonia.",
    "posture__head_movement_amplitude":
        "Instantaneous distance of the head-region centroid from its median "
        "position. Will burst in head tics, head bobbing, cervical dystonia.",
    "posture__arm_swing_left":
        "Distance of the left-arm-region centroid from its median position. "
        "Tracks tremor, chorea, dystonic arm posturing on the left side.",
    "posture__arm_swing_right":
        "Symmetric to arm_swing_left, on the right side.",
    "posture__leg_movement_left":
        "Distance of the left-foot-region centroid from its median position. "
        "Tracks lower-limb chorea, restless-leg, foot dystonia on the left.",
    "posture__leg_movement_right":
        "Symmetric to leg_movement_left, on the right side.",
    "posture__relative_torsion":
        "Angle of the head-to-feet vector. A rough but consistent measure of "
        "axial torsion, sensitive to dystonia.",
    "posture__silhouette_complexity":
        "Perimeter^2 / (4 pi area), normalised so a disc has value 1. Grows "
        "as the silhouette becomes more star-shaped (limbs sticking out); "
        "useful for distinguishing limb-spreading postures from compact ones.",
}


def get_signal_description(name: str) -> str:
    """Return the clinical description for a signal name."""
    if name in SIGNAL_DESCRIPTIONS:
        return SIGNAL_DESCRIPTIONS[name]

    # Auto-generate for regional aggregates
    if name.startswith("contour_region__"):
        _, region, agg = name.split("__")
        return (f"Contour region '{region}': {agg}. "
                f"Region-level summary of the silhouette contour points "
                f"belonging to this anatomical zone.")
    if name.startswith("grid_region__"):
        _, region, agg = name.split("__")
        return (f"Interior grid region '{region}': {agg}. "
                f"Region-level summary of the interior grid points falling "
                f"in this body zone.")

    # Raw contour / grid coordinates
    m = CONTOUR_COL_RE.match(name)
    if m:
        return (f"Raw contour point {int(m.group(1))} {m.group(2).upper()}-"
                "coordinate. Tracks a single sampled point along the patient's "
                "silhouette boundary.")
    m = GRID_COL_RE.match(name)
    if m:
        return (f"Raw interior grid point (row {int(m.group(1))}, "
                f"col {int(m.group(2))}) {m.group(3).upper()}-coordinate. "
                "Tracks a single sampled point inside the patient's silhouette.")

    return "(no description available)"
