# What has been done (historical baseline + current repo verification)

This file separates **historical report claims** from **current repository-verified state**.

---

## 1) Historical baseline summary (Historical claim)
Goal: predict continuous valence (−1 to +1) as proxy for empathy-related affect using multimodal signals from OMG-Empathy videos.

Historical baseline approach:
- Raw Face
- Speech/Audio
- Transcript/Text
- Late fusion over post-processed predictions

Historical report context also references a 5-stream paper target (raw face, landmarks, full-body, audio, text), but the historically delivered integrated baseline was 3-stream.

---

## 2) Historical evaluation conventions (Historical claim)
Historically documented conventions included:
- Validation on story #2 across subjects 1–10.
- CCC as the primary metric.

These statements are retained for report continuity, not as current repo defaults.

---

## 3) Current repository state verification (Current repo verified, 2026-02-17)

### 3.1 Data split reality in this repository
- Training annotations present for stories: **2, 4, 5, 8**.
- Validation annotations present for stories: **1**.
- Any claim that this repo currently validates on story #2 by default is inaccurate.

### 3.2 Implemented modality code (present in repo)
- Raw face pipeline and training scripts are present.
- Speech preprocessing/training/evaluation scripts are present.
- Transcript preprocessing/training/prediction scripts are present.
- Landmarks extraction/training/prediction scripts are present.
- Full-body extraction/training scripts are present.

### 3.3 Integration status
- Late fusion currently wired as **3-stream** (raw face + transcript + speech).
- Landmarks/full-body are implemented as standalone streams but are **not integrated in the active fusion path**.

---

## 4) Reconciled baseline takeaway
You are not starting from scratch. The codebase already contains all modality pipelines, but project wrap-up still requires integration and standardization work:
1. Split hygiene and leakage guardrails.
2. Standardized per-stream prediction outputs.
3. A reproducible 5-stream fusion/evaluation pipeline.
4. Reporting artifacts generated from declared split manifests.

---

## 5) Labeling convention used in this plan folder
- **Historical claim**: statement from prior report context.
- **Current repo verified**: directly checked against this repository.
- **Target state**: work still to be completed.
