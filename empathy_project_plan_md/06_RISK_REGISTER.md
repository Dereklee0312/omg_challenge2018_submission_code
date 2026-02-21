# Risk register (reconciled execution phase)

Keep this file current while implementing integration and evaluation.

---

## R1 — Split-definition drift across modules
- Likelihood: High
- Impact: Very high
- Signals:
1. One script assumes story-1 val while another assumes story-2 val.
2. Inconsistent train/val file roots per modality.
- Mitigation:
1. Immutable split manifests.
2. Preflight overlap checks.
3. Require `manifest_id` in all metrics rows.

## R2 — Data leakage (invalid results)
- Likelihood: Medium
- Impact: Very high
- Signals:
1. Unusually high CCC with low reproducibility.
2. Validation data influencing normalization/scaling.
- Mitigation:
1. Train-only fit for normalization/scaling.
2. File-level overlap assertions.
3. Leakage unit tests.

## R3 — Temporal misalignment (fusion failure)
- Likelihood: High
- Impact: High
- Signals:
1. Negative CCC for subsets of subjects.
2. Visible phase shifts in plotted curves.
- Mitigation:
1. Alignment auditor.
2. Shared frame-level prediction contract.
3. Explicit per-stream alignment documentation.

## R4 — Stale path conventions and hardcoded paths
- Likelihood: High
- Impact: High
- Signals:
1. Scripts referencing nonexistent legacy roots.
2. Silent loading of wrong files.
- Mitigation:
1. Centralized config injection.
2. Path existence assertions before training/eval.
3. Remove hardcoded dataset paths.

## R5 — Fusion modality artifact mismatch
- Likelihood: High
- Impact: High
- Signals:
1. Mixed `.npy`/`.csv` assumptions.
2. Length mismatches during fusion assembly.
- Mitigation:
1. Standardized per-stream schema.
2. Conversion wrappers with validation checks.
3. Fail-fast on schema mismatch.

## R6 — Compute bottlenecks
- Likelihood: Medium
- Impact: High
- Signals:
1. Slow iteration cycles and missed integration deadlines.
- Mitigation:
1. Cache reusable artifacts.
2. Start with subset debug runs.
3. Use GPU where available.

## R7 — Overfitting to one split
- Likelihood: Medium
- Impact: Medium
- Signals:
1. Gains not replicated in alternative manifests.
- Mitigation:
1. At least one alternative split evaluation.
2. Clearly separated reporting per manifest.

## R8 — Documentation/code divergence
- Likelihood: High
- Impact: Medium
- Signals:
1. README/plan states differ from script behavior.
- Mitigation:
1. Reconciliation checklist ownership.
2. Update docs in same PR as behavior changes.
