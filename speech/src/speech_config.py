"""Typed accessors for speech-related configuration in `configs/defaults.yaml`."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from shared_utils.config_loader import load_defaults


def load_speech_config() -> dict:
    """Return the top-level `speech` config section from shared defaults."""
    defaults = load_defaults()
    speech_cfg = defaults.get("speech")
    if not isinstance(speech_cfg, dict):
        raise KeyError("Missing 'speech' section in configs/defaults.yaml")
    return speech_cfg


def speech_stft() -> dict:
    """Return speech STFT parameters required for feature extraction."""
    cfg = load_speech_config().get("stft")
    if not isinstance(cfg, dict):
        raise KeyError("Missing speech.stft in configs/defaults.yaml")
    return cfg


def speech_sampling() -> dict:
    """Return speech sampling parameters (e.g., sampling rate)."""
    cfg = load_speech_config().get("sampling")
    if not isinstance(cfg, dict):
        raise KeyError("Missing speech.sampling in configs/defaults.yaml")
    return cfg


def speech_preprocessing() -> dict:
    """Return speech preprocessing settings used by dataset builders."""
    cfg = load_speech_config().get("preprocessing")
    if not isinstance(cfg, dict):
        raise KeyError("Missing speech.preprocessing in configs/defaults.yaml")
    return cfg


def speech_paths() -> dict:
    """Return speech input/output/model path configuration values."""
    cfg = load_speech_config().get("paths")
    if not isinstance(cfg, dict):
        raise KeyError("Missing speech.paths in configs/defaults.yaml")
    return cfg
