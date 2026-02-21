"""Load and normalize project configuration relative to repository root.

This module centralizes defaults/split loading and ensures path-like config
values are resolved against the repo root so scripts behave consistently
regardless of current working directory.
"""

from __future__ import annotations

import copy
import os
import json
from pathlib import Path
from typing import Any


DEFAULTS_REL_PATH = Path("configs/defaults.yaml")
DEFAULT_SPLIT_REL_PATH = Path("configs/splits/current_repo.yaml")


def get_repo_root() -> Path:
    """Return the effective repository root used for path resolution."""
    # Allow explicit override for non-standard execution environments.
    env_root = os.environ.get("OMG_REPO_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"OMG_REPO_ROOT does not exist: {root}")
        return root
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve a potentially relative path against the repository root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (get_repo_root() / path).resolve()


def _load_jsonish(path: Path) -> dict[str, Any]:
    """Load a JSON-compatible config file and raise a contextual error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path} (repo_root={get_repo_root()})"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_defaults_paths(defaults: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy defaults and normalize configured path fields to absolute paths."""
    normalized = copy.deepcopy(defaults)

    manifest_path = normalized.get("manifest_path")
    if isinstance(manifest_path, str):
        normalized["manifest_path"] = str(resolve_repo_path(manifest_path))

    paths_cfg = normalized.get("paths")
    if isinstance(paths_cfg, dict):
        for key, value in paths_cfg.items():
            if isinstance(value, str):
                paths_cfg[key] = str(resolve_repo_path(value))

    pred_cfg = normalized.get("predictions")
    if isinstance(pred_cfg, dict):
        base_dir = pred_cfg.get("base_dir")
        if isinstance(base_dir, str):
            pred_cfg["base_dir"] = str(resolve_repo_path(base_dir))

    speech_cfg = normalized.get("speech")
    if isinstance(speech_cfg, dict):
        speech_paths = speech_cfg.get("paths")
        if isinstance(speech_paths, dict):
            for key, value in speech_paths.items():
                if isinstance(value, str):
                    speech_paths[key] = str(resolve_repo_path(value))

    return normalized


def load_defaults(path: str | Path | None = None) -> dict[str, Any]:
    """Load `configs/defaults.yaml` (JSON content) and normalize path fields."""
    cfg_path = resolve_repo_path(path) if path else resolve_repo_path(DEFAULTS_REL_PATH)
    return _normalize_defaults_paths(_load_jsonish(cfg_path))


def load_split_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load a split manifest from explicit path or default split config path."""
    split_path = (
        resolve_repo_path(path)
        if path
        else resolve_repo_path(DEFAULT_SPLIT_REL_PATH)
    )
    return _load_jsonish(split_path)


def resolve_manifest(defaults: dict[str, Any], override_path: str | Path | None = None) -> dict[str, Any]:
    """Resolve active split manifest with precedence: override, defaults, fallback."""
    if override_path:
        return load_split_manifest(override_path)
    manifest_path = defaults.get("manifest_path")
    if manifest_path:
        return load_split_manifest(manifest_path)
    return load_split_manifest()
