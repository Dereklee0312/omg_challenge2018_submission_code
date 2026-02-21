"""Convert speech per-sample CSV predictions into legacy `.npy` artifacts."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from shared_utils.config_loader import load_defaults, resolve_manifest
from speech_config import speech_paths

defaults = load_defaults()
manifest = resolve_manifest(defaults)
paths_cfg = speech_paths()
subjects = manifest["subjects_val"]
stories = manifest["stories_val"]
model_output_dir = Path(paths_cfg["model_output_folder"]).resolve()
speech_preds_dir = Path(paths_cfg["speech_predictions_dir"]).resolve()
speech_preds_dir.mkdir(parents=True, exist_ok=True)

for i in subjects:
    for j in stories:
        # Restrict conversion to current manifest validation coverage.
        df = pd.read_csv(model_output_dir / f"Subject_{i}_Story_{j}.csv")
        data = df.to_numpy()
        
        np.save(speech_preds_dir / f"Subject_{i}_Story_{j}_predictions.npy", data)
