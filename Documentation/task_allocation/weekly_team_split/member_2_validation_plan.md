# Member 2 Plan (Validation and Reproducibility) - Week Plan

## Role
Own path-invariance validation and reproducibility evidence for infrastructure scripts.

## Weekly Goal
Deliver objective, reproducible evidence that path/config behavior is deterministic across current working directories.

## Hard Guardrails
1. Do not edit existing pipeline scripts.
2. Do not modify these files:
   1. `fullbody/fullbody_main.py`
   2. `transcript/model_predictions.py`
   3. `speech/src/evaluate_model_seq_3.py`
   4. `Multimodal - Transcript + Raw Face + Speech/rawface_model_predictions.py`
   5. `Multimodal - Transcript + Raw Face + Speech/test.py`
3. Validation depth is lightweight only:
   1. no heavy training
   2. no full inference runs that depend on large model/data artifacts

## Exact Files Member 2 Must Create
1. `tools/member2_validation/resolve_expected_paths.py`
   1. read defaults and print expected absolute output paths from source-of-truth config
2. `tools/member2_validation/run_lightweight_checks.sh`
   1. execute the defined lightweight checks from required CWDs
   2. write a single log file
3. `tools/member2_validation/evaluate_results.py`
   1. parse log and generate final pass/fail report
4. `log/member2_validation_matrix.md`
5. `log/member2_validation_report.md`
6. `log/member2_mismatches.md` (only if mismatches exist)
7. `log/member2_release_readiness.md`

## Source of Truth for Expected Paths
Expected output paths must be derived from:
1. `configs/defaults.yaml` (via shared loader)
2. resolved absolute paths from project path utilities
Not from guessed/hardcoded strings.

## Required Validation Contexts (CWD Matrix)
Run checks from exactly these locations:
1. repo root (`/Users/dereklee0312/Uni/omg_challenge2018_submission_code`)
2. `speech/src`
3. `raw_face`

## Required Lightweight Checks
1. Split preflight check
2. Python compile checks for changed infrastructure targets
3. Expected-path resolution printout from `resolve_expected_paths.py`

## Required Commands (Canonical Set)
```bash
uv run tools/preflight_split_check.py
uv run python -m py_compile fullbody/fullbody_main.py transcript/model_predictions.py speech/src/evaluate_model_seq_3.py "Multimodal - Transcript + Raw Face + Speech/rawface_model_predictions.py" "Multimodal - Transcript + Raw Face + Speech/test.py"
uv run python tools/member2_validation/resolve_expected_paths.py
```

When executed from subdirectories, adjust relative paths but keep command meaning identical.

## Objective Pass/Fail Rules
1. `PASS`
   1. command exit code is 0
   2. resolved expected paths are identical across all CWD contexts
2. `FAIL`
   1. nonzero exit code
   2. path mismatch across CWD contexts
   3. required output directory/path key missing from expected path set
3. `BLOCKED`
   1. only for optional checks requiring unavailable external artifacts
   2. must include exact reason and missing dependency

## Day-by-Day Plan

## Day 1 (Mon): Setup and Matrix Definition
1. Create `tools/member2_validation/resolve_expected_paths.py`.
2. Create `log/member2_validation_matrix.md` with columns:
   1. check_id
   2. command
   3. cwd
   4. expected_paths_snapshot
   5. actual_result
   6. status

## Day 2 (Tue): Root-Level Execution
1. Create `tools/member2_validation/run_lightweight_checks.sh`.
2. Run all canonical checks from repo root.
3. Append output to `log/member2_lightweight_checks.log`.

## Day 3 (Wed): Cross-Directory Execution
1. Run the same check set from `speech/src`.
2. Run the same check set from `raw_face`.
3. Append outputs to the same log.

## Day 4 (Thu): Result Parsing and Mismatch Isolation
1. Create `tools/member2_validation/evaluate_results.py`.
2. Parse `log/member2_lightweight_checks.log`.
3. Generate:
   1. `log/member2_validation_report.md`
   2. `log/member2_mismatches.md` (if needed)

## Day 5 (Fri): Retest on Latest Branch State
1. Re-run lightweight checks after Member 1 updates.
2. Regenerate report and mismatch files.
3. Compare with Day 4 and highlight deltas.

## Day 6 (Sat): Reproducibility Confirmation
1. Repeat one full validation cycle.
2. Confirm same status outcomes with same commands.
3. Document any nondeterminism.

## Day 7 (Sun): Final Readiness Handoff
1. Produce `log/member2_release_readiness.md`:
   1. total checks
   2. pass/fail/blocked counts
   3. top unresolved risks
   4. recommended next actions

## Definition of Done
1. All three validation scripts are created under `tools/member2_validation/`.
2. Validation matrix and reports are complete and reproducible.
3. Every status is evidence-backed by logged commands/output.
