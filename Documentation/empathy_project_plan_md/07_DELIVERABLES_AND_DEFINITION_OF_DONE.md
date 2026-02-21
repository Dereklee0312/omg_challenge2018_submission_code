# Deliverables & Definition of Done (reconciled)

Final checklist for “project wrapped successfully” based on current repo state.

---

## Deliverable A — Reproducible codebase
DoD:
- One repository with:
1. preprocessing scripts for all streams
2. training scripts for all models
3. prediction export scripts
4. fusion/evaluation scripts
- No unresolved hardcoded legacy paths in active workflow.
- Environment dependencies and lock artifacts documented.

---

## Deliverable B1 — Current baseline reproducibility (3-stream)
DoD:
- Reproduce active 3-stream path (raw face + audio + transcript) on declared manifest.
- Output per-subject CCC table and aggregate CCC.
- Artifacts include manifest ID and settings.

## Deliverable B2 — Target baseline integration (5-stream)
DoD:
- Integrate landmarks and full-body into active fusion path.
- Run 5-stream fusion with documented postprocessing and weighting.
- Produce per-subject CCC table, aggregate CCC, and representative plots.

---

## Deliverable C — Experimental evidence (tables + ablations)
DoD:
- Ablation table:
1. each stream alone
2. 3-stream fusion
3. 5-stream fusion
- Postprocessing ablations:
1. low-pass on/off
2. per-subject scaling on/off
- Weighting ablations:
1. equal
2. fixed/manual
3. CCC-proportional
- At least one additional split/manifest comparison.

---

## Deliverable D — Demo / inference pipeline
DoD:
- Unified inference CLI that runs end-to-end for a given video.
- Produces output CSV and valence plot.
- Handles partial modality availability with explicit errors/warnings.

---

## Deliverable E — Report-ready narrative and artifacts
DoD:
- Clear method description per modality and fusion stage.
- All tables/figures generated from stored run artifacts.
- Every result includes:
1. manifest ID
2. modality set
3. postprocessing settings
4. weighting policy
5. artifact path reference

---

## Optional stretch deliverables
- Learned fusion module with validated gain.
- Containerized or one-command reproducible setup.
- External-dataset evaluation if feasible.
