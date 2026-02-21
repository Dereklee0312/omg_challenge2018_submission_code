"""Generate landmarks predictions from an existing trained model.

This script avoids retraining by loading a saved `best_model.h5` and reusing
the existing `save_predictions` pipeline from `landmarks/utils.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
from shared_utils.config_loader import resolve_repo_path

import utils as lmu


def _parse_args() -> argparse.Namespace:
    """Parse CLI args for selecting model path and prediction mode."""
    parser = argparse.ArgumentParser(
        description="Generate landmarks predictions from a saved model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional model path. If omitted, latest experiments/*/best_model.h5 is used.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default=str(BASE_DIR / "experiments"),
        help="Directory containing experiment_* subfolders.",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "validation_predictions",
            "training_predictions",
            "validation_latent",
            "training_latent",
        ),
        default="validation_predictions",
        help="Select which artifact type/split `save_predictions` should generate.",
    )
    return parser.parse_args()


def _find_latest_best_model(experiments_dir: Path) -> Path:
    """Return most recently modified `best_model.h5` under experiment folders."""
    candidates = sorted(
        experiments_dir.glob("experiment_*/best_model.h5"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No best_model.h5 found under {experiments_dir}/experiment_*/"
        )
    return candidates[0]


def _set_mode(mode: str) -> None:
    """Set prediction/latent flags consumed by `utils.save_predictions`."""
    lmu.save_latent_training = False
    lmu.save_predictions_training = False
    lmu.save_latent_test = False
    lmu.save_predictions_test = False

    if mode == "validation_predictions":
        lmu.save_predictions_test = True
    elif mode == "training_predictions":
        lmu.save_predictions_training = True
    elif mode == "validation_latent":
        lmu.save_latent_test = True
    elif mode == "training_latent":
        lmu.save_latent_training = True


def main() -> None:
    """Load chosen model and run standalone predictions via existing utils flow."""
    args = _parse_args()
    experiments_dir = resolve_repo_path(args.experiments_dir)
    model_path = (
        resolve_repo_path(args.model)
        if args.model
        else _find_latest_best_model(experiments_dir)
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Using landmarks model: {model_path}")
    _set_mode(args.mode)

    # Recompute training labels for f_trick normalization used in save_predictions.
    y_training, _ = lmu.create_Y(
        lmu.subjects_training,
        lmu.stories_training,
        base_path_Y=lmu.base_path_Y_training,
    )

    model = load_model(str(model_path), custom_objects={"ccc_error": lmu.ccc_error})
    lmu.save_predictions(model, y_training)


if __name__ == "__main__":
    main()
