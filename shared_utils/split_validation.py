from __future__ import annotations

from pathlib import Path


def ensure_disjoint(train_stories: list[int], val_stories: list[int]) -> None:
    overlap = set(train_stories).intersection(val_stories)
    if overlap:
        raise ValueError(f"Train/val story overlap detected: {sorted(overlap)}")


def assert_annotation_files_exist(annotations_dir: str | Path, subjects: list[int], stories: list[int]) -> None:
    root = Path(annotations_dir)
    missing = []
    for subject in subjects:
        for story in stories:
            p = root / f"Subject_{subject}_Story_{story}.csv"
            if not p.exists():
                missing.append(str(p))
    if missing:
        sample = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing annotation files ({len(missing)} total). Sample:\n{sample}")


def split_for_story(story: int, train_stories: list[int], val_stories: list[int]) -> str:
    if story in train_stories:
        return "train"
    if story in val_stories:
        return "val"
    return "unknown"
