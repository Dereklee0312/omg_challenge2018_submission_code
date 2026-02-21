## Speech Model

This modality predicts valence sequences from audio features using a recurrent network pipeline.

## Source of Truth
1. Speech configuration comes from `configs/defaults.yaml` under the `speech` section.
2. Path resolution is handled by shared config loader utilities.
3. Canonical prediction CSV outputs are written under `predictions/speech`.

## Dependencies
1. numpy
2. essentia
3. scipy
4. keras
5. tensorflow
6. pandas

## Active Scripts
1. Preprocessing: `speech/src/preprocessing_seq_1.py`
2. Training: `speech/src/build_model_rnn_seq_2.py`
3. Evaluation + prediction export: `speech/src/evaluate_model_seq_3.py`
4. Legacy conversion helper: `speech/src/convert_predictions2npy_4.py`

## Typical Workflow
From repo root:

```bash
uv run python speech/src/preprocessing_seq_1.py
uv run python speech/src/build_model_rnn_seq_2.py
uv run python speech/src/evaluate_model_seq_3.py
```

## Output Locations
1. Canonical CSV predictions: `predictions/speech/`
2. Model output CSVs: `speech/model_output/`
3. Latent feature outputs: `speech/last_latent_dim/`

## Notes
1. ffmpeg may be required to generate `.wav` files from `.mp4` depending on your data state.
2. For team onboarding and cross-modality conventions, see `docs/team_codebase_playbook.md`.
