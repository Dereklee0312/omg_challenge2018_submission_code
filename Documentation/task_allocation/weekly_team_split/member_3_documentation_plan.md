# Member 3 Plan (Documentation and Runbooks) - Next Week

## Summary
This updated plan is decision-complete and focused on documentation consistency for infrastructure paths, commands, and outputs.

Primary objective:
1. Make docs accurate and runnable with `uv`
2. Centralize path/output truth in one runbook
3. Ensure README content stays aligned to that runbook

## Role
Own documentation updates and command/runbook clarity for infrastructure and outputs.

## Weekly Goal
Deliver:
1. Updated `README.md`
2. Updated `speech/README.md`
3. Updated `landmarks/README.md`
4. New canonical runbook: `docs/runbook_infra_paths.md`
5. Verification and handoff logs in `log/`

## Hard Guardrails
1. Do not edit model/training/inference source code.
2. Do not change config values in `configs/defaults.yaml`.
3. Do not introduce alternate path definitions that conflict with the runbook.

## Exact Files to Edit
1. `README.md`
2. `speech/README.md`
3. `landmarks/README.md`

## Exact New File to Create
1. `docs/runbook_infra_paths.md`

## Required Log Deliverables
1. `log/member3_doc_gap_list.md`
2. `log/member3_command_verification.md`
3. `log/member3_docs_handoff.md`

## Canonical Documentation Policy
1. `docs/runbook_infra_paths.md` is the single source of truth for:
   1. output directory map
   2. canonical vs legacy output explanation
   3. `uv` execution commands
   4. root vs subdirectory invocation examples
2. If README content conflicts with runbook, update README to match runbook.

## Required Content in `docs/runbook_infra_paths.md`
1. **Path Source of Truth**
   1. Explain that `configs/defaults.yaml` is authoritative for configured paths.
2. **Outputs Map**
   1. Canonical CSV outputs: `predictions/<modality>`
   2. Legacy outputs (where retained), with exact directories per modality.
3. **Execution Guide (`uv`)**
   1. Run from repo root examples
   2. Run from subdirectory examples (with adjusted relative paths)
4. **Validation/Smoke Commands**
   1. `uv run tools/preflight_split_check.py`
   2. `uv run python -m py_compile ...` for key infra files
5. **Known Constraints**
   1. Lightweight validation boundaries
   2. data/model dependency caveats

## Day-by-Day Plan

## Day 1 (Mon): Audit and Gap Capture
1. Review current docs:
   1. `README.md`
   2. `speech/README.md`
   3. `landmarks/README.md`
2. Record stale commands/path statements in `log/member3_doc_gap_list.md`.

## Day 2 (Tue): Build Canonical Runbook
1. Create `docs/runbook_infra_paths.md` with required sections.
2. Draft output map using current defaults/path conventions.

## Day 3 (Wed): Update Root README
1. Update `README.md` to:
   1. reference runbook for path/output details
   2. keep quick-start concise and accurate
2. Remove or correct outdated execution guidance.

## Day 4 (Thu): Update Modality READMEs
1. Update `speech/README.md`:
   1. confirm defaults-based config flow
   2. align command examples with runbook
2. Update `landmarks/README.md`:
   1. document standalone prediction script usage
   2. align outputs and command examples with runbook

## Day 5 (Fri): Command Verification Pass
1. Run every command added/changed in docs once.
2. Record each in `log/member3_command_verification.md`:
   1. command
   2. cwd
   3. result (`pass/fail`)
   4. note (if dependency-limited)

## Day 6 (Sat): Consistency and Review Integration
1. Compare all README path/output mentions with runbook.
2. Resolve mismatches.
3. Incorporate feedback from Member 1 and Member 2 findings.

## Day 7 (Sun): Final Documentation Handoff
1. Produce `log/member3_docs_handoff.md`:
   1. files changed
   2. major clarifications made
   3. unresolved doc risks or limitations

## Acceptance Criteria
1. Runbook exists and is complete: `docs/runbook_infra_paths.md`.
2. All three README files are aligned with runbook.
3. Every added/updated command has verification evidence.
4. No docs claim paths/outputs that conflict with current config conventions.

## Definition of Done
1. Documentation is consistent, reproducible, and teammate-usable.
2. Output locations and command usage are unambiguous.
3. Handoff logs are complete and review-ready.
