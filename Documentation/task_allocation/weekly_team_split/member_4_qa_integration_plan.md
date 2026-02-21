# Member 4 Plan (QA and Integration Checklist) - Next Week

## Summary
This updated plan is decision-complete and uses a strict release gate.  
Member 4 owns final integration quality status based on reproducible evidence from Members 1-3 plus independent lightweight QA checks.

## Role
Own final QA checklist, integration risk triage, and strict go/no-go recommendation.

## Weekly Goal
Produce a clear, evidence-backed release decision:
1. `GO` only if all strict gate conditions pass
2. `NO-GO` if any blocker/path/docs consistency condition fails

## Hard Guardrails
1. Do not edit core pipeline code.
2. Do not modify training/inference scripts directly.
3. Limit changes to:
   1. QA artifacts under `log/`
   2. optional QA helper scripts under `tools/member4_qa/`
4. No heavy training/e2e runtime jobs.

## Mandatory Inputs (Must Be Consumed)
1. `log/member1_change_summary.md`
2. `log/member1_final_handoff.md` (if available by final review)
3. `log/member2_validation_report.md`
4. `log/member2_mismatches.md` (if present)
5. `docs/runbook_infra_paths.md`
6. `log/member3_command_verification.md`
7. `log/member3_docs_handoff.md` (if available by final review)

## Scope
### In Scope
1. Build and execute infra-focused QA checklist
2. Run lightweight smoke + compile checks
3. Cross-validate Member 1-3 deliverables for consistency
4. Publish strict go/no-go decision

### Out of Scope
1. Heavy model training runs
2. Core algorithm/code feature changes
3. Product feature development

## Strict Go/No-Go Gate (Locked)
Mark `NO-GO` if any of the following is true:
1. Any blocker-level issue remains open.
2. Any path determinism check fails across required CWD contexts.
3. Any canonical documented command fails in verification evidence.
4. Any README content conflicts with `docs/runbook_infra_paths.md`.
5. Any required artifact from Members 1-3 is missing without owner/date commitment.

Mark `GO` only when all gate checks pass or are explicitly closed with verified evidence.

## Required QA Checklist Sections
`log/member4_qa_checklist.md` must include:
1. Source-of-truth config checks
2. Path determinism checks
3. Compile/smoke checks
4. Documentation consistency checks (README vs runbook)
5. Risk register linkage (owner + due date)

## Canonical Lightweight Commands
```bash
uv run tools/preflight_split_check.py
uv run python -m py_compile shared_utils/config_loader.py shared_utils/pred_schema.py fullbody/fullbody_main.py transcript/model_predictions.py speech/src/evaluate_model_seq_3.py "Multimodal - Transcript + Raw Face + Speech/rawface_model_predictions.py" "Multimodal - Transcript + Raw Face + Speech/test.py" landmarks/landmarks_generate_predictions.py
```

## Day-by-Day Plan

## Day 1 (Mon): Checklist + Evidence Framework
1. Create `log/member4_qa_checklist.md` with required sections.
2. Create `log/member4_risk_register.md` template with columns:
   1. issue_id
   2. severity (`blocker/high/medium/low`)
   3. evidence
   4. owner
   5. due_date
   6. status

## Day 2 (Tue): Baseline QA Run (Repo Root)
1. Run canonical lightweight commands from repo root.
2. Record outputs in checklist.
3. Mark initial pass/fail.

## Day 3 (Wed): Cross-Directory QA
1. Re-run equivalent checks from at least two additional CWDs:
   1. `speech/src`
   2. `raw_face`
2. Confirm path determinism parity.
3. Log mismatches and suspected owners.

## Day 4 (Thu): Integration Review with Member Artifacts
1. Ingest required Member 1-3 artifacts.
2. Verify internal consistency:
   1. code changes vs validation reports
   2. docs/runbook vs verification logs
3. Update `log/member4_risk_register.md` with blocker/high-risk items.

## Day 5 (Fri): Final QA Pass on Latest Branch State
1. Re-run critical checks after latest updates.
2. Confirm no regressions introduced.
3. Update checklist and risk statuses.

## Day 6 (Sat): Go/No-Go Draft
1. Create `log/member4_go_no_go_draft.md`.
2. For each strict gate condition, record:
   1. status (`pass/fail`)
   2. evidence link
   3. blocker owner if failed

## Day 7 (Sun): Final QA Handoff
1. Publish `log/member4_final_readiness.md` containing:
   1. summary scorecard
   2. strict gate outcome (`GO` or `NO-GO`)
   3. blocker list with owners/dates
   4. approved areas
   5. prioritized next-step actions

## Required Final Deliverables
1. `log/member4_qa_checklist.md`
2. `log/member4_risk_register.md`
3. `log/member4_go_no_go_draft.md`
4. `log/member4_final_readiness.md`

## Acceptance Criteria
1. QA checklist is complete and reproducible.
2. All strict gate conditions are explicitly evaluated with evidence.
3. Final readiness report includes clear ownership and due dates for unresolved issues.

## Definition of Done
1. Final status is explicit (`GO`/`NO-GO`) and evidence-backed.
2. Team can make next-week prioritization directly from Member 4 artifacts.
3. No decision ambiguity remains for release readiness.
