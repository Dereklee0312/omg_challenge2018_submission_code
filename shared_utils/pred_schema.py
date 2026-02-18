"""Helpers for canonical prediction CSV outputs and schema validation."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from shared_utils.config_loader import resolve_repo_path


REQUIRED_COLUMNS = [
    "frame_idx",
    "timestamp_s",
    "y_pred",
    "y_true",
    "subject_id",
    "story_id",
    "split",
    "manifest_id",
]


def ensure_prediction_dir(base_dir: str | Path, modality: str) -> Path:
    """Create and return modality-specific prediction directory under base path."""
    # Always anchor relative base dirs to repo root for CWD-invariant outputs.
    out_dir = resolve_repo_path(base_dir) / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_prediction_csv(
    out_dir: str | Path,
    subject: int,
    story: int,
    frame_idx,
    y_pred,
    manifest_id: str,
    split: str,
    y_true=None,
    fps: float = 25.0,
) -> Path:
    """Write per-subject/story predictions to canonical CSV format."""
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = resolve_repo_path(out_path)
    out_path = out_path / f"Subject_{subject}_Story_{story}.csv"
    n = len(y_pred)
    if y_true is None:
        y_true = [None] * n
    df = pd.DataFrame(
        {
            "frame_idx": frame_idx,
            "timestamp_s": [float(i) / fps for i in frame_idx],
            "y_pred": y_pred,
            "y_true": y_true,
            "subject_id": [subject] * n,
            "story_id": [story] * n,
            "split": [split] * n,
            "manifest_id": [manifest_id] * n,
        }
    )
    df.to_csv(out_path, index=False)
    return out_path


def validate_prediction_csv(path: str | Path) -> None:
    """Validate required columns for a generated prediction CSV file."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction schema missing columns in {path}: {missing}")
