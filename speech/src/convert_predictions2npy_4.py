import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from shared_utils.config_loader import load_defaults, resolve_manifest

defaults = load_defaults()
manifest = resolve_manifest(defaults)
subjects = manifest["subjects_val"]
stories = manifest["stories_val"]
SCRIPT_DIR = Path(__file__).resolve().parent
model_output_dir = (SCRIPT_DIR / "../model_output").resolve()
speech_preds_dir = (SCRIPT_DIR / "../speech_predictions").resolve()
speech_preds_dir.mkdir(parents=True, exist_ok=True)

for i in subjects:
    for j in stories:
        df = pd.read_csv(model_output_dir / f"Subject_{i}_Story_{j}.csv")
        data = df.to_numpy()
        
        np.save(speech_preds_dir / f"Subject_{i}_Story_{j}_predictions.npy", data)
