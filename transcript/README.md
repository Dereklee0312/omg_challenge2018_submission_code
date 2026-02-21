## Transcript Model

This modality generates valence predictions from transcript-derived features.

## Source of Truth
1. Global paths and split settings come from `configs/defaults.yaml` and `configs/splits/current_repo.yaml`.
2. Canonical prediction CSV output is under `predictions/transcript`.
3. Legacy transcript `.npy` output path is configured via `paths.transcript_legacy_predictions` in defaults.

## Dependencies
1. numpy
2. keras
3. tensorflow
4. pandas

## Active Prediction Entrypoint
1. `transcript/model_predictions.py`

Run from repo root:

```bash
uv run python transcript/model_predictions.py
```

## Outputs
1. Canonical CSV predictions: `predictions/transcript/`
2. Legacy `.npy` predictions: `predictions/transcript_legacy_npy/`

## Notes
1. Transcript vectors and model weights must be present for prediction generation.
2. Historical transcript pipeline details in root README are legacy context, not the current operational reference.
3. For cross-modality conventions, see `docs/team_codebase_playbook.md`.
