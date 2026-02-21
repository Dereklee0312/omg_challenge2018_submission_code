# Technical backlog (reconciled, prioritized, with status)

Scope assumes solo execution at ~30h/week. This backlog is now aligned to current repository reality.

---

## Legend
- Priority: `P0` must, `P1` should, `P2` nice
- Status: `done`, `partial`, `missing`
- DoD: definition of done

---

# Workstream 0 — Split integrity and reproducibility (P0)

## 0.1 Declare immutable split manifests
- Priority: P0
- Status: missing
- DoD:
1. `split_manifest_current_repo` (`train: 2,4,5,8`, `val: 1`).
2. Optional `split_manifest_paper_compare` if story-2 comparison is rebuilt.
3. Every metric/report row references manifest ID.

## 0.2 Add leakage guardrails
- Priority: P0
- Status: missing
- DoD:
1. Assert train and validation file sets are disjoint.
2. Assert normalization fit is training-only.
3. Add a test that fails on overlap.

## 0.3 Run artifact contract
- Priority: P0
- Status: missing
- DoD:
1. Every run writes config snapshot, git hash, metrics JSON, and artifact paths.
2. Run folder naming is deterministic and timestamped.

---

# Workstream 1 — Prediction contract and fusion wiring (P0)

## 1.1 Standardize prediction schema
- Priority: P0
- Status: missing
- DoD:
1. All streams export a common frame-level schema (CSV contract).
2. Fusion loader has no stream-specific branches for path/schema quirks.

## 1.2 Integrate landmarks and full-body into fusion
- Priority: P0
- Status: missing
- DoD:
1. Fusion consumes 5 streams (raw face, landmarks, full-body, speech, transcript).
2. Per-stream availability checks and clear errors.
3. Produces per-subject CCC table and aggregate CCC.

## 1.3 Fusion weighting/postprocessing strategies
- Priority: P0
- Status: partial
- Current: fixed 3-stream weights + filtering exist in legacy script.
- DoD:
1. Unified support for fixed weights and CCC-proportional weights.
2. Documented train-only fit for subject scaling.
3. Optional coarse grid search as robustness check.

---

# Workstream 2 — Alignment and data-path reliability (P0)

## 2.1 Alignment auditor tool
- Priority: P0
- Status: missing
- DoD:
1. Reports video frame count, annotation length, audio-feature mapping, transcript coverage.
2. Flags mismatch thresholds and likely causes.

## 2.2 Remove hardcoded paths from scripts
- Priority: P0
- Status: partial
- DoD:
1. No modality script depends on stale/legacy hardcoded directories.
2. Paths are supplied via shared config.

---

# Workstream 3 — Modality baselines (status refresh)

## 3.1 Raw face stream
- Priority: P1
- Status: partial
- Notes: implementation exists; split/path logic needs cleanup and manifest alignment.

## 3.2 Landmarks stream
- Priority: P1
- Status: partial
- Notes: extraction and model training exist; export format and fusion integration pending.

## 3.3 Full-body stream
- Priority: P1
- Status: partial
- Notes: extraction and training harness exist; integration and standardized output pending.

## 3.4 Speech stream
- Priority: P1
- Status: partial
- Notes: preprocessing/training/eval exist; config path issues and output normalization need harmonization.

## 3.5 Transcript stream
- Priority: P1
- Status: partial
- Notes: preprocessing/training/prediction exist; output contract harmonization pending.

---

# Workstream 4 — Evaluation and reporting (P0)

## 4.1 Unified evaluation driver
- Priority: P0
- Status: missing
- DoD:
1. Single CLI for per-stream + fusion evaluation.
2. Emits metrics JSON, summary CSV, and plots.
3. Requires explicit split manifest selection.

## 4.2 Required ablations
- Priority: P0
- Status: missing
- DoD:
1. Stream-only results for all 5 streams.
2. 3-stream vs 5-stream fusion.
3. Postprocessing on/off and weighting ablations.

## 4.3 Cross-split robustness
- Priority: P1
- Status: missing
- DoD:
1. At least one alternative split evaluation beyond default current-repo split.
2. No mixed-manifest reporting.

---

# Workstream 5 — Packaging and demo (P1)

## 5.1 Unified inference CLI (`predict_video.py`)
- Priority: P1
- Status: missing
- DoD:
1. Runs preprocessing/inference/fusion end-to-end.
2. Emits output CSV and valence curve plot.

## 5.2 Environment reproducibility
- Priority: P1
- Status: partial
- Notes: requirements and lock artifacts exist; end-to-end one-command setup still needs verification.

---

# Stretch (after P0/P1)

## S1 Learned fusion
- Status: missing

## S2 Advanced multimodal fusion architectures
- Status: missing
