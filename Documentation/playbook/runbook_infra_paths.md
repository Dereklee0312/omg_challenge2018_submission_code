# Infrastructure Paths Runbook

## Purpose
This runbook defines current path/output conventions for active workflows.

## Source of Truth
1. Config: `configs/defaults.yaml`
2. Split manifest: `configs/splits/current_repo.yaml`
3. Path resolver: `shared_utils/config_loader.py`

## Canonical Prediction Outputs
Base: `predictions/`
1. `predictions/rawface`
2. `predictions/landmarks`
3. `predictions/fullbody`
4. `predictions/speech`
5. `predictions/transcript`

## Legacy Outputs (Retained)
1. Rawface legacy `.npy`: `predictions/rawface_legacy_npy`
2. Transcript legacy `.npy`: `predictions/transcript_legacy_npy`
3. Speech model output CSVs: `speech/model_output`
4. Speech latent vectors: `speech/last_latent_dim`

## Standard Validation Commands
From repo root:

```bash
uv run tools/preflight_split_check.py
uv run python -m py_compile shared_utils/config_loader.py shared_utils/pred_schema.py
```

## Run From Subdirectories
If running outside repo root, keep commands equivalent and adjust relative paths only.
Expected behavior: resolved output targets remain the same absolute directories.

## Team Execution Docs
1. Team onboarding: `docs/team_codebase_playbook.md`
2. Weekly roles: `plans/weekly_team_split/`
3. QA logs and historical notes: `log/`
