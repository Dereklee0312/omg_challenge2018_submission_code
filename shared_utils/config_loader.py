from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULTS_PATH = Path("configs/defaults.yaml")
DEFAULT_SPLIT_PATH = Path("configs/splits/current_repo.yaml")


def _load_jsonish(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_defaults(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULTS_PATH
    return _load_jsonish(cfg_path)


def load_split_manifest(path: str | Path | None = None) -> dict[str, Any]:
    split_path = Path(path) if path else DEFAULT_SPLIT_PATH
    return _load_jsonish(split_path)


def resolve_manifest(defaults: dict[str, Any], override_path: str | Path | None = None) -> dict[str, Any]:
    if override_path:
        return load_split_manifest(override_path)
    manifest_path = defaults.get("manifest_path")
    if manifest_path:
        return load_split_manifest(manifest_path)
    return load_split_manifest()
