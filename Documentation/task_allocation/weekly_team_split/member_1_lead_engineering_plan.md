# Updated Member 1 Plan (Lead Engineering) - Next Week

## Summary
This updated plan is now decision-complete and explicitly aligned to the remaining audit goals:
1. `#1` Fullbody CWD/path hardening
2. `#3` Multimodal checker path correction
3. `#4` Rawface behavior alignment with **Train+Val story coverage**

## Role
Own code-level fixes to the remaining infrastructure audit items and provide implementation handoff evidence for validation/docs/QA tracks.

## Weekly Goal
Deliver merged fixes for:
1. `fullbody/fullbody_main.py` path hardening (`#1`)
2. `Multimodal - Transcript + Raw Face + Speech/rawface_model_predictions.py` behavior alignment (`#4`)
3. `Multimodal - Transcript + Raw Face + Speech/test.py` checker update (`#3`)

## Scope
### In Scope
1. Code updates in the three target files
2. Small config/path constant updates only if needed for deterministic output behavior
3. Validation evidence and technical handoff notes

### Out of Scope
1. Heavy model retraining
2. Broad refactors outside the three target files
3. Presentation/document-only work

## Implementation Policy (Locked)
1. Rawface story coverage policy: **Train+Val stories**
   - `stories = sorted(set(manifest["stories_train"] + manifest["stories_val"]))`
2. Keep canonical CSV split tagging via `split_for_story(...)`.
3. Preserve deterministic repo-root output behavior.

## Day-by-Day Plan

## Day 1 (Mon): Baseline and Expected Behavior Spec
1. Reproduce current behavior from repo root and at least one subdirectory.
2. Capture current output paths and failure points for all three scripts.
3. Write `log/member1_expected_behavior.md` with explicit expected behavior:
   1. Fullbody path resolution is repo-root deterministic
   2. Rawface generates Train+Val story outputs
   3. Checker reads current rawface/transcript output locations

### Deliverable
`log/member1_expected_behavior.md`

## Day 2 (Tue): Fullbody Path Hardening (`#1`)
Update `fullbody/fullbody_main.py` so all relevant paths are repo-root resolved:
1. `_resolve_img_template(...)`
   1. `split_override` path
   2. `dataset_root` fallback path
2. `_resolve_label_template(...)`
   1. `split_override` path
   2. `dataset_root` fallback path
3. `main()`
   1. `checkpoint_dir`
   2. `log_dir`

### Acceptance Criteria
1. Same absolute paths used when run from root vs subdirectory.
2. CLI/config precedence behavior remains unchanged.
3. No model/training semantic change.

## Day 3 (Wed): Rawface Behavior Alignment (`#4`)
Update `Multimodal - Transcript + Raw Face + Speech/rawface_model_predictions.py`:
1. Keep deterministic model/data/output path resolution.
2. Change story coverage from val-only to Train+Val union:
   1. `stories = sorted(set(manifest["stories_train"] + manifest["stories_val"]))`
3. Keep subject coverage consistent with existing project convention.
4. Keep canonical CSV write + `split_for_story(...)` tagging intact.

### Acceptance Criteria
1. Rawface no longer val-only unless file truly missing.
2. Canonical CSV split metadata remains correct.
3. Legacy `.npy` outputs still go to configured repo-root path.

## Day 4 (Thu): Multimodal Checker Fix (`#3`)
Update `Multimodal - Transcript + Raw Face + Speech/test.py`:
1. Replace stale directories with current output directories used by generators.
2. Align stories/subjects with manifest-driven expectations (or explicitly document fallback behavior).
3. Add clear missing-file handling/reporting instead of hard failure where possible.

### Acceptance Criteria
1. Checker resolves correct folders for current pipeline outputs.
2. Checker runs without immediate path-related `FileNotFoundError` in normal state.
3. Output logs clearly identify missing pairs if data is incomplete.

## Day 5 (Fri): Verification + PR Prep
1. Run compile and split checks.
2. Run target smoke tests from at least two CWDs.
3. Produce `log/member1_change_summary.md`:
   1. changes
   2. rationale
   3. validation evidence
   4. residual risks

### Commands
```bash
uv run python -m py_compile fullbody/fullbody_main.py \
"Multimodal - Transcript + Raw Face + Speech/rawface_model_predictions.py" \
"Multimodal - Transcript + Raw Face + Speech/test.py"
uv run tools/preflight_split_check.py
```

## Day 6 (Sat): Address Review Feedback
1. Incorporate findings from Member 2/3/4.
2. Fix edge cases and update summary docs.

## Day 7 (Sun): Final Handoff
1. Publish `log/member1_final_handoff.md` with:
   1. final file changes
   2. validation summary
   3. known limitations
   4. recommended next actions

## Important Interfaces/Contracts
1. No schema change to canonical prediction CSV (`shared_utils/pred_schema.py` contract remains).
2. No new public CLI contracts required beyond current script interfaces.
3. Config source of truth remains `configs/defaults.yaml`.

## Test Cases and Scenarios
1. Fullbody run from root vs subdirectory uses identical resolved paths for templates/checkpoints/logs.
2. Rawface script emits outputs for Train+Val stories and split tags are correct.
3. Multimodal checker reads updated output directories and reports shape checks.
4. Compile and split preflight checks pass.

## Assumptions and Defaults
1. Assumption: manifest defines authoritative train/val story sets.
2. Default: rawface story coverage is Train+Val union.
3. Default: behavior changes are limited to path determinism and intended coverage restoration.
