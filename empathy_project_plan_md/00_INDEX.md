# Empathy Detection Project (COMP6015 Part 2) — Reconciled Planning Baseline

This folder contains the **reconciled** solo plan to finish the empathy detection project over a 14-week semester at **~30 hours/week**.

## What you have right now (Current repo verified, 2026-02-17)
A functional baseline with implemented modality pipelines and an active 3-stream fusion:
- **Raw Face** (3D CNN + subject ID)
- **Speech/Audio** (BiGRU sequence model)
- **Transcript/Text** (LSTM + attention + post-hoc normalization)
- **Landmarks** pipeline + training scripts implemented
- **Full-body** pipeline + training scripts implemented
- A **3-modality late fusion** script currently wired for raw face + transcript + speech

## Reality check on splits (Current repo verified, 2026-02-17)
- Training annotations available for stories: **2, 4, 5, 8**
- Validation annotations available for stories: **1**
- Historical report references to “story #2 validation” are kept as historical context only.

## What remains
- Integrate landmarks and full-body outputs into a robust **5-stream fusion** path.
- Standardize evaluation and prediction artifact formats across streams.
- Add leakage/alignment guardrails and reproducible run manifests.

## How to use these docs
1. Read **01_WHAT_WAS_DONE.md** for historical baseline vs current verified state.
2. Read **02_GAP_ANALYSIS_AND_GOALS.md** for reconciled must-haves.
3. Use **03_TECHNICAL_BACKLOG.md** as the prioritized execution queue.
4. Execute **04_14_WEEK_PLAN_SOLO_30H_PER_WEEK.md** as a remaining-work schedule.
5. Follow **05_EXPERIMENT_PROTOCOLS.md** for split discipline and reproducible metrics.
6. Keep **06_RISK_REGISTER.md** open while integrating and evaluating.
7. Use **07_DELIVERABLES_AND_DEFINITION_OF_DONE.md** as finish-line criteria.
8. Track status in **08_RECONCILIATION_AUDIT_CHECKLIST.md**.

## Assumptions
- Long trainings should run on Linux + GPU whenever possible.
- OMG-Empathy dataset paths are available under the repo `data/` layout.

## Quick pointers
- If you do one thing first: lock split manifests and remove split ambiguity.
- If you do one thing second: standardize prediction exports so fusion is stream-agnostic.
- If you do one thing third: complete 5-stream integration and ablation reporting.
