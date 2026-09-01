#!/usr/bin/env python3
"""
cody2_utils.py
==============

Self-contained utility functions used by ``train_sam3.py`` and
``inference_sam3.py``. The function bodies are reproduced verbatim from the
cody-2 paper repository (Cif & Vasques, 2026), specifically from
``cody-pipeline/train.py``. They are vendored here so that the SAM 3 pipeline
is a stand-alone, self-contained package that does not require the cody-2
repository to be present at run time.

Provenance: cody-pipeline/train.py, commit corresponding to the cody-2
preprint. Functions kept identical so SAM 3 results remain bit-comparable
with the cody-2 protocol.

Vendored functions:
    - to_numeric_any
    - label_to_known_binary
    - patient_truth_from_windows
    - is_control_patient_id
    - resolve_device
    - print_env
    - robust_normalize
    - higuchi_fd
    - permutation_entropy
    - compute_features_1d
    - sample_rows_ratio_keep_controls
    - make_label_summary

Constants:
    - DEFAULT_SYMPTOMS
    - EXTRA_LABELS
    - NON_SIGNAL_COLUMNS
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Symptom / column conventions
# ---------------------------------------------------------------------------

DEFAULT_SYMPTOMS = ["Dystonia", "Tremor", "Myoclonus", "Chorea",
                    "Athetosis", "Ballismus", "Stereotypies", "Tics"]
EXTRA_LABELS = ["Bradykinesia", "FOG", "Ataxia"]
NON_SIGNAL_COLUMNS = {"Fog", "FOG", "Bradykinesia", "Ataxia"}


# ---------------------------------------------------------------------------
# Numeric coercion
# ---------------------------------------------------------------------------

def to_numeric_any(x: pd.Series) -> pd.Series:
    """Convert numeric-like strings (with optional comma decimals) to float.

    Handles the French-locale "0,1" -> 0.1 case found in the cody-2 dataset.
    """
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce")
    return pd.to_numeric(
        x.astype("string").str.replace(",", ".", regex=False),
        errors="coerce",
    )


# ---------------------------------------------------------------------------
# Label handling: clinician 0/1/2 -> known binary mask
# ---------------------------------------------------------------------------

def label_to_known_binary(y_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map clinician labels:
      0 -> known 0
      1 -> known 1
      2 -> unknown (excluded)

    Returns:
      y_bin (0/1) for known entries (others arbitrary 0)
      known_mask (True for y in {0, 1})
    """
    y_raw = np.asarray(y_raw)
    known = (y_raw == 0) | (y_raw == 1)
    yb = np.zeros_like(y_raw, dtype=int)
    yb[y_raw == 1] = 1
    return yb.astype(int), known.astype(bool)


def patient_truth_from_windows(vals: np.ndarray) -> int:
    """
    Patient-level ground truth from window labels (values in {0, 1, 2, NaN}).

      any 1 -> 1
      no 1 but any 2 -> -1 (unknown)
      otherwise -> 0
    """
    v = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0
    if np.any(v == 1):
        return 1
    if np.any(v == 2):
        return -1
    return 0


def is_control_patient_id(pid: str) -> bool:
    """Heuristic: patient IDs that start with 'C' are healthy controls."""
    if pid is None:
        return False
    s = str(pid).strip().upper()
    return bool(re.match(r"^C(\b|[_\-]|\d)", s))


# ---------------------------------------------------------------------------
# Environment / device
# ---------------------------------------------------------------------------

def resolve_device(user_device: Optional[str]) -> str:
    if user_device is not None and str(user_device).strip():
        return str(user_device)
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def print_env(selected_device: str) -> None:
    try:
        import torch
        print(f"[ENV] torch={torch.__version__} cuda={torch.cuda.is_available()} "
              f"selected_device={selected_device}")
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"[ENV] cuda:{i} name={name} mem_gb={mem_gb:.1f}")
    except Exception as e:
        print(f"[ENV] gpu info unavailable: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Robust normalization, fractal dimension, permutation entropy
# ---------------------------------------------------------------------------

def robust_normalize(sig: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Median / IQR normalization, falling back to std when IQR is degenerate."""
    sig = np.asarray(sig, dtype=float)
    sig = sig[~np.isnan(sig)]
    if sig.size == 0:
        return sig
    med = np.median(sig)
    q75, q25 = np.percentile(sig, [75, 25])
    iqr = float(q75 - q25)
    scale = iqr if iqr > eps else float(np.std(sig) + eps)
    return (sig - med) / scale


def higuchi_fd(x: np.ndarray, kmax: int = 6) -> float:
    """Higuchi fractal dimension."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < (kmax + 2):
        return np.nan
    lk = []
    lnk = []
    for k in range(1, kmax + 1):
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if idx.size < 2:
                continue
            diffs = np.abs(np.diff(x[idx]))
            length = np.sum(diffs)
            norm = (n - 1) / (k * (idx.size - 1))
            lm.append(length * norm)
        if not lm:
            continue
        Lk = float(np.mean(lm))
        if Lk > 0:
            lk.append(np.log(Lk))
            lnk.append(np.log(1.0 / k))
    if len(lk) < 2:
        return np.nan
    D = np.polyfit(lnk, lk, 1)[0]
    return float(D)


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Normalized permutation entropy."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < (order * delay + 1):
        return np.nan
    patterns: Dict[tuple, int] = {}
    count = 0
    for i in range(n - delay * (order - 1)):
        window = x[i:(i + delay * order):delay]
        if window.size != order:
            continue
        key = tuple(np.argsort(window))
        patterns[key] = patterns.get(key, 0) + 1
        count += 1
    if count == 0:
        return np.nan
    p = np.array(list(patterns.values()), dtype=float) / float(count)
    pe = stats.entropy(p, base=np.e)
    pe_norm = pe / np.log(math.factorial(order))
    return float(pe_norm)


def _safe_skew(sig: np.ndarray) -> float:
    sig = np.asarray(sig, dtype=float)
    sig = sig[~np.isnan(sig)]
    if sig.size < 3 or np.nanstd(sig) < 1e-12:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = stats.skew(sig, bias=False)
    return float(v) if np.isfinite(v) else 0.0


def _safe_kurtosis(sig: np.ndarray) -> float:
    sig = np.asarray(sig, dtype=float)
    sig = sig[~np.isnan(sig)]
    if sig.size < 4 or np.nanstd(sig) < 1e-12:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = stats.kurtosis(sig, bias=False)
    return float(v) if np.isfinite(v) else 0.0


# ---------------------------------------------------------------------------
# Per-window feature extraction (the 19 descriptors used in cody-2)
# ---------------------------------------------------------------------------

def compute_features_1d(signal: np.ndarray, fps: float,
                        robust_norm: bool) -> Dict[str, float]:
    """Compute statistical, temporal, spectral and non-linear descriptors of
    a 1-D signal.

    Returns a dict with 19 keys:
      mean, std, min, max, median, range, skew, kurtosis, energy,
      slope, iqr, entropy, var, fft_peak_freq, fft_peak_amp,
      zero_crossings_delta, abs_accel_mean, higuchi_fd, perm_entropy.
    """
    sig = np.asarray(signal, dtype=float)
    sig = sig[~np.isnan(sig)]
    if sig.size == 0:
        return {k: np.nan for k in [
            "mean", "std", "min", "max", "median", "range", "skew",
            "kurtosis", "energy",
            "slope", "iqr", "entropy", "var", "fft_peak_freq", "fft_peak_amp",
            "zero_crossings_delta", "abs_accel_mean", "higuchi_fd", "perm_entropy"
        ]}

    if robust_norm:
        sig = robust_normalize(sig)

    mean = float(np.mean(sig))
    var = float(np.var(sig))
    std = float(np.std(sig, ddof=1)) if sig.size > 1 else 0.0
    mn = float(np.min(sig))
    mx = float(np.max(sig))
    med = float(np.median(sig))
    rng = float(np.ptp(sig))
    energy = float(np.sum(sig ** 2))

    skew = _safe_skew(sig)
    kurt = _safe_kurtosis(sig)

    slope = (float(np.polyfit(np.arange(sig.size, dtype=float), sig, 1)[0])
             if sig.size >= 2 else 0.0)
    q75, q25 = np.percentile(sig, [75, 25])
    iqr = float(q75 - q25)

    hist, _ = np.histogram(sig, bins=20, density=False)
    hist = hist.astype(float)
    ent = (float(stats.entropy((hist / hist.sum()) + 1e-12, base=np.e))
           if hist.sum() > 0 else np.nan)

    if sig.size >= 4:
        s0 = sig - np.mean(sig)
        fft_vals = np.abs(np.fft.rfft(s0))
        freqs = np.fft.rfftfreq(s0.size, d=1.0 / float(fps))
        if fft_vals.size > 1:
            peak_idx = int(np.argmax(fft_vals[1:]) + 1)
            fft_peak_freq = float(freqs[peak_idx])
            fft_peak_amp = float(fft_vals[peak_idx])
        else:
            fft_peak_freq = np.nan
            fft_peak_amp = np.nan
    else:
        fft_peak_freq = np.nan
        fft_peak_amp = np.nan

    if sig.size >= 3:
        delta = np.diff(sig)
        zc = int(((delta[:-1] * delta[1:]) < 0).sum())
        accel = np.diff(sig, n=2)
        abs_accel_mean = float(np.mean(np.abs(accel))) if accel.size else 0.0
    else:
        zc = 0
        abs_accel_mean = 0.0

    hfd = higuchi_fd(sig, kmax=6)
    pe = permutation_entropy(sig, order=3, delay=1)

    return {
        "mean": mean, "std": std, "min": mn, "max": mx, "median": med,
        "range": rng, "skew": skew, "kurtosis": kurt, "energy": energy,
        "slope": slope, "iqr": iqr, "entropy": ent, "var": var,
        "fft_peak_freq": fft_peak_freq, "fft_peak_amp": fft_peak_amp,
        "zero_crossings_delta": zc, "abs_accel_mean": abs_accel_mean,
        "higuchi_fd": hfd, "perm_entropy": pe,
    }


# ---------------------------------------------------------------------------
# Negative-subsampling that preserves controls
# ---------------------------------------------------------------------------

def sample_rows_ratio_keep_controls(
    y: np.ndarray,
    patient_ids: np.ndarray,
    neg_pos_ratio: float,
    seed: int,
    ctrl_frac: float = 0.30,
) -> np.ndarray:
    """Stratified negative subsampling with a cap on healthy controls.

    Returns the sorted indices of the rows to keep, preserving all positives
    plus up to ``neg_pos_ratio * n_pos`` negatives, with at most ``ctrl_frac``
    of those drawn from healthy controls. Controls are identified by
    ``is_control_patient_id``.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    patient_ids = np.asarray(patient_ids).astype(str)

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return np.arange(len(y))

    is_ctrl = np.array([is_control_patient_id(pid) for pid in patient_ids],
                       dtype=bool)
    idx_neg_ctrl = idx_neg[is_ctrl[idx_neg]]
    idx_neg_pat = idx_neg[~is_ctrl[idx_neg]]

    max_neg = int(np.ceil(float(neg_pos_ratio) * len(idx_pos)))

    # If the budget already covers every negative, keep all of them
    if max_neg >= len(idx_neg):
        keep_neg = idx_neg
        keep = np.sort(np.concatenate([idx_pos, keep_neg]))
        return keep

    # Otherwise: stratified sampling with a cap on healthy controls
    n_ctrl = int(np.floor(ctrl_frac * max_neg))
    n_ctrl = min(n_ctrl, len(idx_neg_ctrl))

    n_pat = max_neg - n_ctrl
    n_pat = min(n_pat, len(idx_neg_pat))

    # If patient negatives are insufficient, top up with controls
    n_ctrl = min(max_neg - n_pat, len(idx_neg_ctrl))

    keep_ctrl = (rng.choice(idx_neg_ctrl, size=n_ctrl, replace=False)
                 if n_ctrl > 0 else np.array([], dtype=int))
    keep_pat = (rng.choice(idx_neg_pat, size=n_pat, replace=False)
                if n_pat > 0 else np.array([], dtype=int))

    return np.sort(np.concatenate([idx_pos, keep_ctrl, keep_pat]))


# ---------------------------------------------------------------------------
# Label summary helper
# ---------------------------------------------------------------------------

def make_label_summary(dfw: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    """Per-label sanity summary at window and patient level (counts of 0/1/2)."""
    rows = []
    # Window level
    for lab in labels:
        v = pd.to_numeric(dfw[lab], errors="coerce").to_numpy(dtype=float)
        rows.append({
            "level": "window",
            "label": lab,
            "n0": int(np.sum(v == 0)),
            "n1": int(np.sum(v == 1)),
            "n2": int(np.sum(v == 2)),
            "n_nan": int(np.sum(~np.isfinite(v))),
            "n_total": int(len(v)),
        })
    # Patient level
    g = dfw.groupby("patient_id", sort=True)
    for lab in labels:
        pt = g[lab].apply(
            lambda s: patient_truth_from_windows(s.values)
        ).to_numpy(dtype=int)
        rows.append({
            "level": "patient",
            "label": lab,
            "n0": int(np.sum(pt == 0)),
            "n1": int(np.sum(pt == 1)),
            "n_unknown": int(np.sum(pt == -1)),
            "n_total": int(len(pt)),
        })
    return pd.DataFrame(rows)
