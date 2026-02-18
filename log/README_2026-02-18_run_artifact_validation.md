# 2026-02-18 — Run Artifact Validation Snapshot

## Purpose
Record which modality pipelines have run successfully and produced validation outputs, based on on-disk artifacts.

## Validation Criteria Used
- `trained checkpoint exists`: expected model file(s) are present.
- `validation output exists`: `predictions/<modality>/Subject_{1..10}_Story_1.csv` present with `split=val` and `manifest_id=current_repo_v1`.

## Status by Modality
- `landmarks`: `validated`
  - Checkpoints: `landmarks/experiments/experiment_*/best_model.h5` (4 runs found)
  - Validation outputs: `predictions/landmarks/Subject_1_Story_1.csv` ... `Subject_10_Story_1.csv` (10 files)
  - Legacy NPY outputs also present: `landmarks/predictions_test_FINAL/*.npy`
- `speech`: `validated`
  - Checkpoint: `speech/models/bigru_PROVA2.keras`
  - Validation outputs: `predictions/speech/Subject_1_Story_1.csv` ... `Subject_10_Story_1.csv` (10 files)
- `transcript`: `validated`
  - Checkpoint: `transcript/tmp_weights.h5`
  - Validation outputs: `predictions/transcript/Subject_1_Story_1.csv` ... `Subject_10_Story_1.csv` (10 files)
- `raw_face`: `trained_only`
  - Checkpoints present: `raw_face/model/conv_3D_raw_face.*.keras` (multiple best epochs)
  - No canonical validation CSV set under `predictions/rawface/` in this snapshot.
- `fullbody`: `not_validated_in_snapshot`
  - No checkpoint files found under repo `models/` at snapshot time.

## Notes
- This snapshot is artifact-based; it does not claim metric quality, only run completion evidence.
- `predictions_test_FINAL/` at repo root is empty; landmarks legacy NPY outputs are under `landmarks/predictions_test_FINAL/`.
