# 14-week solo plan (remaining-work execution, ~30h/week)

Total budget remains **14 x 30h = 420h**, but this schedule is now based on current codebase state.

Design principle: **stabilize split/evaluation truth first, then integrate 5-stream fusion, then report**.

---

## Weeks 1-2 — Stabilization sprint
Objectives:
- Lock split manifests and remove split ambiguity.
- Add leakage and path sanity checks.
- Define unified prediction artifact schema.

Deliverables:
- Split manifests documented and enforced.
- Preflight checks for overlap/path/alignment basics.
- Prediction schema spec and sample files.

---

## Weeks 3-4 — Fusion input unification
Objectives:
- Make all 5 modalities export fusion-ready predictions in the same schema.
- Remove modality-specific loader hacks in fusion input stage.

Deliverables:
- Standardized exports for raw face, landmarks, full-body, speech, transcript.
- Loader validation report for all subjects/stories in default split.

---

## Weeks 5-6 — 5-stream fusion integration
Objectives:
- Integrate landmarks and full-body into active fusion pipeline.
- Implement configurable filtering/scaling/weight strategies.

Deliverables:
- 5-stream fusion run on default split (`train 2,4,5,8 / val 1`).
- Per-subject CCC table + aggregate CCC.

---

## Weeks 7-8 — Evaluation hardening and ablations
Objectives:
- Build unified evaluation driver.
- Produce required ablation suite and postprocessing comparisons.

Deliverables:
- Stream-only and fusion ablation tables.
- Weighted strategy comparison (equal/fixed/CCC-proportional).

---

## Weeks 9-10 — Robustness and split comparison
Objectives:
- Run at least one alternative split or paper-comparison track.
- Keep manifests isolated and non-mixed in reporting.

Deliverables:
- Cross-split comparison table with explicit manifest IDs.
- Notes on failure modes and modality disagreement.

---

## Weeks 11-12 — Packaging and inference workflow
Objectives:
- Implement unified inference CLI.
- Ensure graceful handling of missing modalities/files.

Deliverables:
- `predict_video.py` equivalent CLI behavior (final script name may vary but interface documented).
- Output CSV + valence plot artifacts.

---

## Weeks 13-14 — Final report assets and submission hardening
Objectives:
- Produce final tables/figures directly from run artifacts.
- Perform final reproducibility pass and package deliverables.

Deliverables:
- Submission-ready report assets and reproducibility checklist.
- Final metrics with explicit split manifests and artifact paths.
