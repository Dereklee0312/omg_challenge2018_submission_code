# Historical Note
This file is archived and kept for project history.
It may reference scripts, paths, or workflow assumptions that are no longer current.
Use active docs in `docs/` and root/modality READMEs for current operational guidance.

# 2026-02-17 — Split Consistency + Default-First Runtime

## Scope
Implemented default-first split consistency and shared configuration so scripts run with built-in settings and optional overrides only.

## Core decisions implemented
- Canonical split manifest: `train={2,4,5,8}`, `val={1}`.
- Strict fail-fast split validation.
- Shared prediction CSV schema for cross-modality/fusion exchange.
- Default-first behavior intended for `uv run <script>.py` usage.

## New shared infrastructure
- `configs/splits/current_repo.yaml`
- `configs/defaults.yaml`
- `shared_utils/config_loader.py`
- `shared_utils/split_validation.py`
- `shared_utils/pred_schema.py`
- `tools/preflight_split_check.py`

## Scripts updated
- Raw face:
  - `raw_face/raw_face_main.py`
  - `raw_face/utils.py`
- Landmarks:
  - `landmarks/utils.py`
  - `landmarks/landmarks_preprocessing.py`
- Full-body:
  - `fullbody/fullbody_main.py`
- Speech:
  - `speech/config/configOMG.ini`
  - `speech/src/loadconfig.py`
  - `speech/src/preprocessing_seq.py`
  - `speech/src/build_model_rnn_seq.py`
  - `speech/src/evaluate_model_seq.py`
  - `speech/src/convert_predictions2npy.py`
- Transcript:
  - `transcript/preprocessing_pipeline.py`
  - `transcript/transcript_LSTM.py`
  - `transcript/model_predictions.py`
- Fusion/prediction scripts:
  - `Multimodal - Transcript + Raw Face + Speech/late_fusion.py`
  - `Multimodal - Transcript + Raw Face + Speech/rawface_model_predictions.py`

## Validation performed
- Python syntax checks passed on modified scripts.
- Preflight check passed:
  - `python3 tools/preflight_split_check.py`
  - manifest: `current_repo_v1`
  - train stories: `[2, 4, 5, 8]`
  - val stories: `[1]`

## Known follow-up
- Ensure all modalities produce schema CSVs before running strict 5-modality fusion.
- Add/confirm fullbody prediction exporter in schema format if not already present.
