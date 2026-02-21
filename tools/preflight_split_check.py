from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared_utils.config_loader import load_defaults, resolve_manifest
from shared_utils.split_validation import ensure_disjoint, assert_annotation_files_exist


def main() -> None:
    defaults = load_defaults()
    manifest = resolve_manifest(defaults)
    ensure_disjoint(manifest["stories_train"], manifest["stories_val"])

    assert_annotation_files_exist(
        defaults["paths"]["train_annotations"],
        manifest["subjects_train"],
        manifest["stories_train"],
    )
    assert_annotation_files_exist(
        defaults["paths"]["val_annotations"],
        manifest["subjects_val"],
        manifest["stories_val"],
    )

    print("Preflight OK")
    print(f"manifest_id={manifest['manifest_id']}")
    print(f"train_stories={manifest['stories_train']}")
    print(f"val_stories={manifest['stories_val']}")


if __name__ == "__main__":
    main()
