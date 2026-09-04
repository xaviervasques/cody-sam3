# CODY-SAM3

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22232609.svg)](https://doi.org/10.5281/zenodo.22232609)

**Fully foundation-model-based phenotyping of combined hyperkinetic movement disorders.**

Markerless phenotyping of eight hyperkinetic movement disorders (Dystonia,
Tremor, Myoclonus, Chorea, Athetosis, Ballismus, Stereotypies, Tics) from
routine clinical video, using **SAM 3** dense body-surface segmentation and a
**TabICLv2** tabular foundation model. Third study of the CODY series
(CODY-1 adult, CODY-2 pediatric, CODY-3 = this SAM 3 successor).

Strategy: **train once → calibrate per site → deploy**. A shared predictive
backbone (segmentation + kinematic featurization + TabICLv2 per-symptom
prediction) is held fixed; only the subject-level decision layer (aggregation +
thresholds) is recalibrated per dataset. Calibration uses the **CODY-2
objective** for continuity between the two papers.

This repository contains the full analysis code of the paper:

> Cif L, Souei Z, Demailly D, Castro-Jimenez M, Ortigoza Escobar JD,
> Ur Rehman MM, Dornadic M, Huby S, Hariz G-M, Hubsch C, Moraud EM, Bloch J,
> Horvath G, Oullier O, Vasques X.
> *A fully foundation-model-based phenotyping of combined hyperkinetic
> movement disorders.* (submitted, 2026)

The CODY series:

1. **CODY-1 (adult)** — Cif L, et al. *Deep Learning Pose Estimation for
   Phenotyping of Co-Occurring Hyperkinetic Movement Disorders.*
   Annals of Clinical and Translational Neurology (2026), 1–22.
   [doi:10.1002/acn3.70474](https://doi.org/10.1002/acn3.70474) —
   code: [xaviervasques/CODY](https://github.com/xaviervasques/CODY)
2. **CODY-2 (pediatric transfer)** — Cif L, Demailly D, Souei Z, et al.
   *Simultaneous hyperkinetic movement disorders phenotyping: a cross-cohort
   pediatric transfer study using routine videos, markerless pose estimation
   and a tabular foundation model.*
   [arXiv:2606.07674](https://doi.org/10.48550/arXiv.2606.07674) (2026),
   accepted in Frontiers in Neurology. —
   code: [xaviervasques/cody-pipeline](https://github.com/xaviervasques/cody-pipeline)
3. **CODY-3 (this repository)** — the SAM 3 paper above (submitted).

---

## Repository layout

Scripts are kept flat (self-contained imports: each reads `cody2_utils.py`
next to it, and the orchestrator locates its siblings directly). The phase
structure lives in `notebooks/`, named by phase.

```
cody-sam3/
├── *.py   (flat — grouped by phase below)
│
│   # phase 1 — training on the adult development cohort (GPU)
│   merge_sam3_labels.py        merge rater labels -> per-video tables
│   aggregate_by_patient.py     per-video -> per-patient consensus
│   sam3_features.py            SAM 3 dense features (kinematic descriptors)
│   train_sam3.py               fit TabICLv2 backbone (per-symptom)
│   cv_sam3.py                  subject-grouped CV / LOPO (out-of-fold probs)
│   cv_aggregate_oof.py         aggregate OOF window probs -> patient scores
│   tune_threshold.py           internal patient-level thresholds (dev)
│
│   # phase 2 — external inference (no backbone retraining) (GPU)
│   inference_sam3.py           SAM 3 extraction + TabICLv2 inference -> window probs
│
│   # phase 3 — calibrate / deploy / evaluate (CODY-2 objective) (CPU)
│   calibrate_persite.py        JOINT multi-label calibration (cody-2 objective)
│   evaluate_external.py        multi-consensus eval + inter-rater kappa
│   calibration_robustness.py   exhaustive C(n,k) robustness distributions
│   run_allsets_exploration.py  exploration of all calibration sets
│   run_external_validation.py  orchestrator (inference -> calibrate -> eval)
│
│   # shared
│   cody2_utils.py              vendored cody-2 helpers (no external dep)
│   check_setup.py              environment / dependency check
│
└── notebooks/
    ├── phase_0_master_pipeline.ipynb                 end-to-end overview
    ├── phase_1a_sam3_feature_extraction.ipynb        SAM 3 dense feature extraction
    ├── phase_1b_training_cv_lopo.ipynb               training + CV/LOPO (tier selection)
    ├── phase_2_external_inference.ipynb              external SAM 3 extraction + inference
    ├── phase_3_external_validation_v1_manual.ipynb        calibration set chosen MANUALLY
    ├── phase_3_external_validation_v2_exploration.ipynb   ALL calibration sets + top 5
    └── phase_4_feature_analysis.ipynb                feature analysis (clinical interpretability)
```

Notebook outputs have been cleared before release (patient privacy); the
notebooks are meant to be re-run on your own data.

---

## Installation

Python ≥ 3.10 with an NVIDIA GPU is required for phases 1–2 (feature
extraction, training, inference). Phases 3–4 (calibration, evaluation, feature
analysis) run on any CPU.

```bash
# 1) fresh environment
conda create -n cody-sam3 python=3.11 -y
conda activate cody-sam3

# 2) PyTorch with CUDA — pick the command for your CUDA version at
#    https://pytorch.org/get-started/locally/  (example: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) everything else
pip install -r requirements.txt

# 4) the CLIP package required by Ultralytics SAM 3 (NOT the pypi 'clip')
pip uninstall -y clip
pip install git+https://github.com/ultralytics/CLIP.git

# 5) sanity check
python check_setup.py
```

## Pretrained models (download procedure)

Model weights are **not** included in this repository. Four sets of weights
are involved:

### 1. SAM 3 segmentation weights (`sam3.pt`, ~3.45 GB) — required

Access to `facebook/sam3` is gated on Hugging Face:

1. Create a Hugging Face account and request access at
   <https://huggingface.co/facebook/sam3> (accept the model licence).
2. Create an access token (Settings → Access Tokens).
3. Download the weights:

```bash
pip install -U huggingface_hub
hf auth login          # paste your token
hf download facebook/sam3 sam3.pt --local-dir .
```

or from Python (as done in `notebooks/phase_2_external_inference.ipynb`):

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt", local_dir=".")
```

SAM 3 inference runs through **Ultralytics ≥ 8.3.237** (already in
`requirements.txt`). Point the notebooks' `MODEL_PATH` at the downloaded
`sam3.pt`.

### 2. TabICLv2 tabular foundation model — required

Installed via the `tabicl` package (in `requirements.txt`); its checkpoint is
downloaded automatically on first use.

### 3. SAM 3D Body (`facebook/sam-3d-body-dinov3`) — optional

Only needed for the optional 3D reconstruction section (section 10 of
`notebooks/phase_1a_sam3_feature_extraction.ipynb`); it is not part of the
quantitative pipeline. Follow the notebook's instructions:

```bash
git clone https://github.com/facebookresearch/sam-3d-body.git
cd sam-3d-body
pip install -r requirements.txt
pip install -e .
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
```

(also gated: accept the licence on the Hugging Face model page first).

### 4. CODY-SAM3 trained model bundles — to reproduce the paper's external validation

The TabICLv2 bundles trained on the development cohort (Tier 1/2/3, with the
retained **Tier 2** configuration: 81 signals, 1560 features) are not
distributed with the Zenodo record. Retrain them from the archived
time-series with `phase_1b_training_cv_lopo.ipynb`, then pass
`--bundle_dir` to the inference/CV scripts — or contact the corresponding
author.

---

## Data availability

The de-identified data are archived in the **CODY** record on Zenodo
(CC BY 4.0, v1.1.0 or later):
**DOI (v1.1.0, the published version): [10.5281/zenodo.22275553](https://doi.org/10.5281/zenodo.22275553)**
(concept DOI for all versions: [10.5281/zenodo.22232609](https://doi.org/10.5281/zenodo.22232609))

`dataset_0` is the paper's *training dataset*; `dataset_1/2/3` are the three
evaluation datasets. Archive names carry the study and pipeline
(`*_cody_3_sam3` for this study); each archive extracts to a folder named
without that suffix:

| Role                                              | Archive                                       |
|---------------------------------------------------|-----------------------------------------------|
| Raw rater labels + YOLOv8 keypoints (dataset_0)   | `dataset_0_labels_raw_cody_1_2_yolo.zip`      |
| Merged-by-video labels + SAM 3 (dataset_0)        | `dataset_0_merged_by_video_cody_3_sam3.zip`   |
| Merged-by-patient consensus (dataset_0)           | `dataset_0_merged_by_patient_cody_3_sam3.zip` |
| SAM 3 kinematic time-series (dataset_0)           | `dataset_0_timeseries_cody_3_sam3.zip`        |
| SAM 3 kinematic time-series (datasets 1/2/3)      | `datasets_1_2_3_timeseries_cody_3_sam3.zip`   |
| Results (inference/calibration/robustness)        | `datasets_1_2_3_results_cody_3_sam3.zip`      |

The record also carries `dataset_2_timeseries_cody_2_yolo.zip` (the pediatric
YOLOv8 keypoints of the earlier CODY-2 study), not used by this pipeline.

The **raw clinical videos cannot be shared publicly** (identifiable patient
data). The Zenodo archive contains the de-identified SAM 3 kinematic
time-series extracted from them (`*_sam3_timeseries.xlsx`), which are the
direct input of every downstream analysis; all results of the paper can be
reproduced from phase 1b onward without the videos. Access to the videos may
be considered by the corresponding author upon reasonable request, subject to
ethics approval.

Scripts take all paths as arguments (`--infer_root`, `--out_root`, …), so any
local layout works. The Colab notebooks read from `MyDrive/sam_3_infer/` with
`external_validation/` for results.

---

## Reproducing the paper

| Phase | Notebook | GPU | What it does |
|---|---|---|---|
| 0 | `phase_0_master_pipeline.ipynb` | — | End-to-end overview (sections A–F) |
| 1a | `phase_1a_sam3_feature_extraction.ipynb` | yes | SAM 3 dense feature extraction on the development cohort |
| 1b | `phase_1b_training_cv_lopo.ipynb` | yes | TabICLv2 training + 5-fold CV / LOPO (tier selection) |
| 2 | `phase_2_external_inference.ipynb` | yes | SAM 3 extraction + TabICLv2 inference on the 3 external datasets |
| 3 | `phase_3_external_validation_v1_manual.ipynb` | no | Calibration on the clinician's a priori set → **paper results** |
| 3 | `phase_3_external_validation_v2_exploration.ipynb` | no | All C(n,k) calibration sets → robustness distributions |
| 4 | `phase_4_feature_analysis.ipynb` | no | Feature analysis (clinical interpretability) |

The whole external-site workflow can also be run as a single command:

```bash
python run_external_validation.py --help
```

Tier 2 (81 signals, 1560 features) is the retained configuration; LOPO
`n_estimators=8` is the main internal result.

---

## Calibration objective (CODY-2, verbatim)

`calibrate_persite.py` reproduces `calibrate_dataset2.py` from the CODY-2 repo.
Calibration is a **joint multi-label optimisation** over the eight
phenomenologies (coordinate descent), maximising

```
score = J(Y_true, Y_pred)
        − extra_penalty · fp_excess_norm                       (default 0.35)
        − fn_penalty    · fn_key_rate(Dystonia, Myoclonus, Chorea)  (default 0.35)
```

* `J` = multi-label objective: `jaccard` (default) | `exact_match` | `macro_f1`
* `fp_excess_norm` penalises predicting more positive labels than the GT
  cardinality (discourages over-calling)
* `fn_key_rate` penalises missed positives on clinically important labels
* aggregation grid: `p70, p90, p95, max`
* per-label threshold guard-rails — min: Tremor/Tics 0.35, Ballismus 0.30,
  Stereotypies 0.25, Athetosis 0.20 ; max: Dystonia 0.55, Myoclonus 0.50,
  Chorea 0.55

Rules are fitted on the clinician-selected **calibration subset** only, then
deployed on the held-out patients. Results are reported in three views
(held-out / all / calibration) under **main present/absent** and **restrictive
agreement-based** definitions at agreement levels ≥3/5, ≥4/5, 5/5, with
multi-label metrics (Hamming, Jaccard) and pooled + per-phenotype confusion
(TP/TN/FP/FN).

## Two flavours of external validation (phase 3)

* **v1 — manual** (`..._v1_manual.ipynb`): you pick one calibration set per
  dataset (e.g. the CODY-2 a priori set `P3,P5,P8,P10,P11` for dataset_2).
  This is what goes in the paper.
* **v2 — exploration** (`..._v2_exploration.ipynb`): tests **all** C(n,k)
  calibration sets (sampled if too many), reports the distribution of metrics
  (publishable robustness result) plus top-5 sets by metric and by confusion
  counter (diagnostic only — do not report tops as performance).

## Hardware

| Stage                         | GPU | Where                                    |
|-------------------------------|-----|------------------------------------------|
| phase 1 — training, CV/LOPO   | yes | PC RTX 5080 / Colab RTX PRO 6000         |
| phase 2 — external inference  | yes | same                                     |
| phase 3 — calibrate/evaluate  | no  | any CPU (post-processing of window probs)|
| phase 4 — feature analysis    | no  | any CPU                                  |

---

## Citation

If you use this code, please cite the paper (see `CITATION.cff`):

```bibtex
@article{cif2026codysam3,
  title   = {A fully foundation-model-based phenotyping of combined
             hyperkinetic movement disorders},
  author  = {Cif, Laura and Souei, Zohra and Demailly, Diane and
             Castro-Jimenez, Mayt{\'e} and Ortigoza Escobar, Juan Dario and
             Ur Rehman, Muhammad Mushhood and Dornadic, Morgan and
             Huby, Sophie and Hariz, Gun-Marie and Hubsch, C{\'e}cile and
             Moraud, Eduardo M. and Bloch, Jocelyne and Horvath, Gabriella and
             Oullier, Olivier and Vasques, Xavier},
  year    = {2026},
  note    = {Submitted}
}
```

## License

Code released under the [MIT License](LICENSE). The SAM 3 and SAM 3D Body
weights are subject to their own licences (accept them on the Hugging Face
model pages); the clinical data are covered by the terms of the Zenodo record.
