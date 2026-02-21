## Raw Face Model

The raw face model predicts valence from sequences of subject face cropped frames using a 3d convolutional architecture. This model additionally takes into account the identity of the subject as an auxiliary input added after the convolutional layers.

### Dependencies

1. numpy
2. keras
3. tensorflow
4. scikit-image
 

### Preprocessing

Face crops are produced by landmarks preprocessing and stored in configured landmarks output directories (see defaults path keys).


### Training

Run from repo root:

```bash
uv run python raw_face/raw_face_main.py
```

The training script resolves data and annotation paths from `configs/defaults.yaml`.

### Prediction Generation

Raw-face prediction export used by multimodal flow:
1. `Multimodal - Transcript + Raw Face + Speech/rawface_model_predictions.py`

Run from repo root:

```bash
uv run python "Multimodal - Transcript + Raw Face + Speech/rawface_model_predictions.py"
```

### Outputs
1. Canonical CSV predictions: `predictions/rawface/`
2. Legacy `.npy` predictions: `predictions/rawface_legacy_npy/`

### Notes
1. For cross-modality conventions and path/source-of-truth rules, see `docs/team_codebase_playbook.md`.
