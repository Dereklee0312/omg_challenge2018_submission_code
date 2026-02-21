# Gap analysis & goals (reconciled to current codebase)

This file converts historical intent + current repository reality into concrete end goals.

---

## A) Target architecture (Target state)
Desired final system remains the 5-stream late-fusion design:
1. Raw Face
2. Face Landmarks
3. Full Body
4. Audio
5. Language

Expected fusion/postprocessing behavior:
- Per-stream low-pass filtering.
- Per-subject scaling fit on training only.
- Weighted averaging with documented weight policy.

---

## B) Reconciled gaps vs current baseline

### B1) Integration gaps (Current repo verified)
- Landmarks stream code exists.
- Full-body stream code exists.
- Active fusion is still 3-stream only.
- Missing work is **integration + standardization**, not stream code from zero.

### B2) Evaluation and split discipline gaps (Current repo verified)
- Repo default split is currently `train: 2,4,5,8` and `val: 1`.
- Historical references to `val: 2` remain in docs and cause confusion.
- No single standardized evaluation driver exists yet.

### B3) Engineering reliability gaps (Current repo verified)
- Hardcoded paths and inconsistent data roots remain in scripts.
- Prediction formats differ by stream (`.npy` and `.csv` conventions differ).
- No unified run manifest contract (config + commit + metrics + artifacts).

---

## C) Success criteria (reconciled)

### C1) Minimum viable success (must-have)
1. Reproducible end-to-end pipeline on a declared split manifest.
2. 5-stream integrated fusion path using outputs from all modality pipelines.
3. Standardized prediction artifact contract across streams.
4. Reporting-ready outputs: per-subject CCC, fusion summaries, plots.

### C2) Strong success (should-have)
5. Clear comparison between:
- Track A: current-repo split (`train 2,4,5,8 / val 1`).
- Track B: paper-comparison split (explicitly rebuilt if needed).
6. Ablations for stream contribution and postprocessing choices.

### C3) Stretch goals (nice-to-have)
7. Learned fusion variants (stacking/gating/attention).
8. Unified inference demo CLI for unseen videos.

---

## D) Stream-specific goals

### D1) Landmarks (Target state)
- Keep existing extraction/model as baseline implementation.
- Add robust prediction export into standardized fusion-ready format.
- Validate alignment and CCC under declared split manifest.

### D2) Full-body (Target state)
- Keep existing 128x128 and sequence-based baseline.
- Add prediction export and integration into fusion.
- Validate split-consistent CCC and error cases.

---

## E) Global fusion & evaluation goals

### E1) Fusion goals (Target state)
Build a single fusion module that:
- Ingests all stream predictions via a common schema.
- Applies optional per-stream filtering.
- Applies optional per-subject scaling (fit on train only).
- Computes CCC per subject/story and aggregate summaries.

### E2) Evaluation goals (Target state)
Provide one standardized evaluation driver that:
- Declares split manifest explicitly.
- Emits metrics JSON, summary CSV, and plots.
- Supports both current-repo and paper-comparison tracks without mixing.
