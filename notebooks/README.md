# Notebooks — CODY-SAM3 (by phase)

| Notebook | Phase | GPU | Role |
|---|---|---|---|
| `phase_0_master_pipeline.ipynb` | overview | — | End-to-end overview |
| `phase_1a_sam3_feature_extraction.ipynb` | 1 | yes | SAM 3 dense feature extraction (development cohort) |
| `phase_1b_training_cv_lopo.ipynb` | 1 | yes | TabICLv2 training + CV/LOPO (tier selection) |
| `phase_2_external_inference.ipynb` | 2 | yes | SAM 3 extraction + inference on the external datasets |
| `phase_3_external_validation_v1_manual.ipynb` | 3 | no | **Manual** calibration (one set per dataset), CODY-2 objective |
| `phase_3_external_validation_v2_exploration.ipynb` | 3 | no | **All** calibration sets + distributions + top 5, CODY-2 objective |
| `phase_4_feature_analysis.ipynb` | 4 | no | Feature analysis (clinical interpretability) |

**v1 vs v2**: v1 produces the paper's results (clinician's a priori set).
v2 explores all calibration sets — the **distribution** is publishable
(robustness), the **top 5** lists are diagnostic (do not report them as
performance).

The Colab notebooks read from `MyDrive/sam_3_infer/` with
`external_validation/` for the results, and call the scripts from the
repository root.

Executed outputs were cleared before release (patient privacy): re-run the
notebooks on your own data.
