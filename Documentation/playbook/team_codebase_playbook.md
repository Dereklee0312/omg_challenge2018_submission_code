# Team Codebase Playbook

## 1) Purpose of This Document
This playbook is the single onboarding and execution reference for the 4-person team working in this repository.

Use this document to:
1. Understand how the codebase is organized.
2. Know which files are source-of-truth for paths/config/splits.
3. Run the correct scripts with `uv` from any directory.
4. Avoid conflicting edits across team members.
5. Complete weekly tasks with clear deliverables and validation steps.

This guide is written to be operational, not conceptual only.

---

## 2) Project Context (What This Repo Does)
The repository implements a multimodal empathy/valence pipeline for OMG-style data using multiple modality models and fusion:
1. `raw_face` model
2. `landmarks` model
3. `fullbody` model
4. `speech` model
5. `transcript` model
6. multimodal fusion under `Multimodal - Transcript + Raw Face + Speech/`

Most scripts are legacy-first code modernized with shared config and split checks.

---

## 3) Environment and Tooling

### 3.1 Python and Dependency Manager
1. Python version target is defined in `pyproject.toml`: `>=3.13`.
2. Package/dependency manager and script runner is `uv`.

### 3.2 Core Dependencies
Defined in `pyproject.toml`:
1. `tensorflow`, `keras`
2. `numpy`, `pandas`, `matplotlib`
3. `opencv-python`, `dlib`, `scikit-image`
4. `essentia`

### 3.3 Standard Command Pattern
Always prefer:
1. `uv run <command>`

Examples:
```bash
uv run tools/preflight_split_check.py
uv run python -m py_compile shared_utils/config_loader.py
```

---

## 4) Top-Level Repository Map

### 4.1 Core Folders
1. `configs/`: defaults and split manifests
2. `shared_utils/`: shared config/split/path/prediction utilities
3. `data/`: train/validation annotations, videos, audio
4. `raw_face/`, `landmarks/`, `fullbody/`, `speech/`, `transcript/`: modality code
5. `Multimodal - Transcript + Raw Face + Speech/`: fusion and cross-modality scripts
6. `predictions/`: canonical per-modality prediction CSV outputs
7. `tools/`: repo-level utility checks
8. `plans/weekly_team_split/`: team execution plans
9. `log/`: evidence, reports, handoffs
10. `docs/`: detailed operational docs (this file)

### 4.2 Important Existing Files for Team Work
1. `configs/defaults.yaml`
2. `configs/splits/current_repo.yaml`
3. `shared_utils/config_loader.py`
4. `shared_utils/pred_schema.py`
5. `tools/preflight_split_check.py`

---

## 5) Source-of-Truth Rules (Must Follow)

### 5.1 Config Source-of-Truth
The canonical config is:
1. `configs/defaults.yaml`

Do not hardcode new relative paths in modality scripts if a defaults key exists.

### 5.2 Split Source-of-Truth
The canonical split manifest is:
1. `configs/splits/current_repo.yaml`

Current split values:
1. `subjects_train`: `[1..10]`
2. `subjects_val`: `[1..10]`
3. `stories_train`: `[2, 4, 5, 8]`
4. `stories_val`: `[1]`

### 5.3 Path Resolution Contract
All relative config paths should resolve via shared path logic:
1. `shared_utils/config_loader.py`
2. `resolve_repo_path(...)`

Meaning:
1. Running from repo root or subdirectory should resolve to same absolute target paths.

### 5.4 Prediction Output Contract
Canonical prediction CSV contract is implemented in:
1. `shared_utils/pred_schema.py`

Canonical CSV columns:
1. `frame_idx`
2. `timestamp_s`
3. `y_pred`
4. `y_true`
5. `subject_id`
6. `story_id`
7. `split`
8. `manifest_id`

---

## 6) Data and Output Contracts

### 6.1 Input Data Expectations
From defaults:
1. annotations:
   1. `data/Training/Annotations`
   2. `data/Validation/Annotations`
2. videos:
   1. `data/Training/Videos`
   2. `data/Validation/Videos`
3. audio:
   1. `data/Training/audio`
   2. `data/Validation/audio`

### 6.2 Canonical Prediction Outputs
Base dir from defaults: `predictions`

Modality subdirs:
1. `predictions/rawface`
2. `predictions/landmarks`
3. `predictions/fullbody`
4. `predictions/speech`
5. `predictions/transcript`

Each contains per-subject/story CSV prediction files.

### 6.3 Legacy Outputs (Still Used)
1. rawface legacy `.npy`: `predictions/rawface_legacy_npy`
2. transcript legacy `.npy`: `predictions/transcript_legacy_npy`
3. speech legacy-style model output folder: `speech/model_output`
4. speech latent output folder: `speech/last_latent_dim`

---

## 7) Shared Utilities Explained

### 7.1 `shared_utils/config_loader.py`
Responsibilities:
1. Load defaults and split manifests.
2. Resolve relative paths to absolute repo-root paths.
3. Normalize nested path sections (`paths`, `predictions.base_dir`, `speech.paths`).

Key functions:
1. `get_repo_root()`
2. `resolve_repo_path(path_value)`
3. `load_defaults(...)`
4. `resolve_manifest(...)`

### 7.2 `shared_utils/pred_schema.py`
Responsibilities:
1. Create modality prediction output directories safely.
2. Write canonical prediction CSV files.
3. Validate required columns.

Key functions:
1. `ensure_prediction_dir(base_dir, modality)`
2. `write_prediction_csv(...)`
3. `validate_prediction_csv(path)`

### 7.3 `tools/preflight_split_check.py`
Purpose:
1. Verify split disjointness and annotation file presence.
2. Quick confidence check before running modality scripts.

Run this before major validation or teammate QA:
```bash
uv run tools/preflight_split_check.py
```

---

## 8) Modality Pipeline Reference

## 8.1 Raw Face
Main training entrypoint:
1. `raw_face/raw_face_main.py`

Key behavior:
1. Loads defaults + manifest.
2. Asserts split disjointness + annotation existence.
3. Trains and saves model checkpoints under `raw_face/model/`.

Prediction entrypoint used in multimodal context:
1. `Multimodal - Transcript + Raw Face + Speech/rawface_model_predictions.py`

Outputs:
1. Legacy `.npy`: `predictions/rawface_legacy_npy`
2. Canonical CSV: `predictions/rawface`

## 8.2 Landmarks
Training entrypoint:
1. `landmarks/landmarks_main.py`

Preprocessing entrypoint:
1. `landmarks/landmarks_preprocessing.py`

Standalone prediction-only entrypoint:
1. `landmarks/landmarks_generate_predictions.py`

Important note:
1. Use `landmarks_generate_predictions.py` when generating predictions from an already trained model without rerunning full training.

## 8.3 Fullbody
Main training entrypoint:
1. `fullbody/fullbody_main.py`

Preprocessing entrypoint:
1. `fullbody/fullbody_preprocessing.py`

Current status note:
1. Fullbody path hardening is an active infrastructure task area.

## 8.4 Speech
Key scripts under `speech/src/`:
1. preprocessing: `preprocessing_seq_1.py`
2. training: `build_model_rnn_seq_2.py`
3. evaluation + prediction export: `evaluate_model_seq_3.py`
4. feature extraction helper: `feat_analysis2.py`
5. legacy conversion utility: `convert_predictions2npy_4.py`

Config access:
1. `speech/src/speech_config.py`
2. Sources values from `configs/defaults.yaml` (`speech` section)

Outputs:
1. `speech/model_output`
2. `speech/last_latent_dim`
3. canonical CSV in `predictions/speech`

## 8.5 Transcript
Prediction entrypoint:
1. `transcript/model_predictions.py`

Outputs:
1. legacy `.npy`: `predictions/transcript_legacy_npy`
2. canonical CSV: `predictions/transcript`

## 8.6 Multimodal Fusion
Core fusion script:
1. `Multimodal - Transcript + Raw Face + Speech/late_fusion.py`

Cross-modality checker utility:
1. `Multimodal - Transcript + Raw Face + Speech/test.py`

---

## 9) Team Member Execution Map

## 9.1 Member 1 (Lead Engineering)
Primary plan file:
1. `plans/weekly_team_split/member_1_lead_engineering_plan.md`

Owns:
1. Fullbody path hardening
2. Rawface behavior alignment
3. Multimodal checker path fixes

Must produce:
1. `log/member1_expected_behavior.md`
2. `log/member1_change_summary.md`
3. `log/member1_final_handoff.md`

## 9.2 Member 2 (Validation)
Primary plan file:
1. `plans/weekly_team_split/member_2_validation_plan.md`

Owns:
1. Lightweight path determinism validation across CWDs
2. Validation-only scripts in `tools/member2_validation/`

Must produce:
1. `log/member2_validation_matrix.md`
2. `log/member2_validation_report.md`
3. `log/member2_release_readiness.md`
4. `log/member2_mismatches.md` (if needed)

## 9.3 Member 3 (Documentation)
Primary plan file:
1. `plans/weekly_team_split/member_3_documentation_plan.md`

Owns:
1. README updates
2. Canonical runbook creation

Must produce:
1. `docs/runbook_infra_paths.md`
2. `log/member3_doc_gap_list.md`
3. `log/member3_command_verification.md`
4. `log/member3_docs_handoff.md`

## 9.4 Member 4 (QA/Integration)
Primary plan file:
1. `plans/weekly_team_split/member_4_qa_integration_plan.md`

Owns:
1. strict GO/NO-GO integration status
2. final QA risk register and readiness report

Must produce:
1. `log/member4_qa_checklist.md`
2. `log/member4_risk_register.md`
3. `log/member4_go_no_go_draft.md`
4. `log/member4_final_readiness.md`

---

## 10) Branching, PR, and Coordination Protocol

### 10.1 Branch Naming
Suggested pattern:
1. `member1/<short-topic>`
2. `member2/<short-topic>`
3. `member3/<short-topic>`
4. `member4/<short-topic>`

### 10.2 Commit Scope Rule
1. One logical unit per commit.
2. Do not bundle unrelated modality changes together.

### 10.3 Cross-Member Dependencies
1. Member 2 and Member 4 should run validation after Member 1 implementation updates.
2. Member 3 should align docs with latest merged behavior, not assumptions.

### 10.4 Conflict Prevention
1. Check `git status --short` before editing.
2. Avoid editing files assigned to another member unless coordinated.
3. If overlap is unavoidable, agree on ownership first in team chat/log.

---

## 11) Standard Operational Commands

### 11.1 Health Checks
```bash
uv run tools/preflight_split_check.py
uv run python -m py_compile shared_utils/config_loader.py shared_utils/pred_schema.py
```

### 11.2 Multi-File Compile Checks
```bash
uv run python -m py_compile \
fullbody/fullbody_main.py \
transcript/model_predictions.py \
speech/src/evaluate_model_seq_3.py \
"Multimodal - Transcript + Raw Face + Speech/rawface_model_predictions.py" \
"Multimodal - Transcript + Raw Face + Speech/test.py" \
landmarks/landmarks_generate_predictions.py
```

### 11.3 Landmarks Prediction-Only Example
```bash
uv run python landmarks/landmarks_generate_predictions.py --help
uv run python landmarks/landmarks_generate_predictions.py --mode validation_predictions
```

---

## 12) Troubleshooting Playbook

## 12.1 "File not found" on outputs
1. Confirm defaults path values in `configs/defaults.yaml`.
2. Confirm path resolution through `shared_utils/config_loader.py`.
3. Confirm script is using defaults key instead of local hardcoded path.

## 12.2 "Works in root, fails in subdirectory"
1. Check if script uses raw `Path("relative")` without repo-root resolution.
2. Validate with Member 2 CWD matrix method.
3. Fix by routing path through `resolve_repo_path(...)`.

## 12.3 Split inconsistency
1. Run `uv run tools/preflight_split_check.py`.
2. Inspect `configs/splits/current_repo.yaml`.
3. Ensure story filters use manifest values.

## 12.4 Canonical prediction CSV issues
1. Validate schema with `shared_utils/pred_schema.py` (`REQUIRED_COLUMNS`).
2. Ensure writer uses `write_prediction_csv(...)`.
3. Confirm `split` and `manifest_id` are populated.

## 12.5 Missing model artifacts
1. For prediction-only scripts, confirm checkpoint file exists.
2. Use script-specific flags where available (`--model` in landmarks prediction script).
3. Log as `BLOCKED` for QA/validation if dependency is absent.

---

## 13) Quality Gates Before Declaring "Done"

### 13.1 Engineering Gate
1. Relevant scripts compile.
2. Preflight split check passes.
3. Path behavior deterministic across CWD for changed scripts.

### 13.2 Documentation Gate
1. Commands in docs have verification entries.
2. README statements align with runbook/path source-of-truth.

### 13.3 QA Gate
1. Risk register has owner and due date for every open item.
2. GO/NO-GO recommendation includes evidence references.

---

## 14) New Contributor Quickstart (First 60 Minutes)
1. Read this playbook fully.
2. Read your member plan in `plans/weekly_team_split/`.
3. Run:
```bash
uv run tools/preflight_split_check.py
```
4. Run a compile check for your touched files.
5. Create your log artifact file in `log/` before making edits.
6. Keep all work tied to your assigned deliverables.

---

## 15) Change Logging Template (Copy/Paste)
Use this in member handoff docs:

```md
## Change Summary
1. Files changed:
2. Why changed:
3. Commands run:
4. Results:
5. Residual risks:
6. Next owner/actions:
```

---

## 16) Final Notes
1. Prioritize deterministic behavior over convenience shortcuts.
2. Keep config and split logic centralized.
3. Keep validation evidence reproducible.
4. Keep ownership boundaries clear to avoid accidental regressions.

If any conflict appears between scripts, defaults, and docs:
1. check defaults and manifest first,
2. then shared_utils behavior,
3. then update docs to reflect actual validated behavior.
