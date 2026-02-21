# Reconciliation Audit Checklist (live tracker)

Date initialized: 2026-02-17

Use this file to track closure of mismatches between planning docs and codebase behavior.

| Item | Status | Evidence file(s) | Owner | Next action |
|---|---|---|---|---|
| Split manifests explicitly defined and enforced | missing | `empathy_project_plan_md/05_EXPERIMENT_PROTOCOLS.md` | solo | Add manifest files + enforce in eval entrypoints |
| Leakage overlap assertions added | missing | `empathy_project_plan_md/03_TECHNICAL_BACKLOG.md` | solo | Implement overlap test and preflight checks |
| Unified prediction schema across 5 streams | missing | `empathy_project_plan_md/05_EXPERIMENT_PROTOCOLS.md` | solo | Implement converters/exporters for all streams |
| Landmarks stream integrated into active fusion | missing | `Multimodal - Transcript + Raw Face + Speech/late_fusion.py` | solo | Extend fusion loader and weighting to include landmarks |
| Full-body stream integrated into active fusion | missing | `Multimodal - Transcript + Raw Face + Speech/late_fusion.py` | solo | Extend fusion loader and weighting to include full-body |
| Fusion script uses current data roots and split manifests | missing | `Multimodal - Transcript + Raw Face + Speech/late_fusion.py` | solo | Remove legacy roots/hardcoded paths |
| Unified evaluation driver implemented | missing | `empathy_project_plan_md/03_TECHNICAL_BACKLOG.md` | solo | Build single CLI for stream + fusion evaluation |
| Inference CLI implemented | missing | `empathy_project_plan_md/07_DELIVERABLES_AND_DEFINITION_OF_DONE.md` | solo | Implement end-to-end `predict_video`-style CLI |
| Run artifact contract enforced (`config/git_hash/metrics/logs`) | missing | `empathy_project_plan_md/05_EXPERIMENT_PROTOCOLS.md` | solo | Add run folder writer utilities |
| Hardcoded legacy paths removed from active scripts | partial | modality scripts under `raw_face/`, `speech/`, `transcript/`, fusion script | solo | Centralize config and add path validation |
| Documentation split ambiguity resolved in plan docs | done | `empathy_project_plan_md/00_INDEX.md`, `empathy_project_plan_md/05_EXPERIMENT_PROTOCOLS.md` | solo | Keep synchronized with code changes |
| Historical vs current vs target labeling applied | done | `empathy_project_plan_md/01_WHAT_WAS_DONE.md` | solo | Continue using labels in future updates |

## Status legend
- `done`: completed and verified against code behavior.
- `partial`: partly complete; gaps remain.
- `missing`: not implemented yet.

## Update rule
When status changes, update both this file and the corresponding backlog/protocol entries in the same change.
