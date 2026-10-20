# Task Context: Damage Detection Improvement

Session ID: 2026-10-13-damage-classifier-improvement
Created: 2026-10-13T08:35:19Z
Status: in_progress (blocked at Phase 2 label-decision gate)

## Current Request
Implement full 8-phase Damage Detection Improvement Implementation Plan. Improve damage classifier from baseline (acc 86.7%, balanced acc 79.8%, damaged recall 91.6%, test=120 samples, 8 FN) to targets: acc >=90%, balanced acc >=85%, damaged recall >=95%, FN <5, no backend inference regression. User approved proposed approach (delegate to TaskManager, batched CoderAgent execution) and said "implement plan".

## Context Files (Standards to Follow)
- NONE — `.opencode/context/` does not exist, no paths.json found. ContextScout failed (free tier limit). Fallback: follow existing repo conventions (read all key source files, see Reference Files).

## Reference Files (Source Material)
- src/train_classifier.py — current training loop (AdamW, weighted sampler, best-val-f1 checkpoint)
- src/model.py — EggGradingCNN (custom 2-conv CNN), checkpoint has class_names/img_size/seed/best_val_metrics
- src/utils.py — get_transforms (flip only) + get_labels
- src/dataset.py — EggDataset (cv2 BGR→RGB → transform)
- src/evaluate_classifier.py — test eval, damage_recall from confusion_matrix row 0, no probs/threshold sweep
- src/create_damage_split.py — stratified split, CLASS_LABELS {Damaged:0, Not Damaged:1}, seed 42
- src/verify_damage_dataset.py — dataset verification
- backend/app/ml/damage_inference.py — DamageClassifier, hardcoded inline EggGradingCNN class, review threshold → Needs Review, BGR2RGB conversion
- backend/app/config.py — DAMAGE_REVIEW_THRESHOLD: float = 0.75
- backend/app/services/prediction_service.py — stores damage_status/confidence/model_version/review_reason
- backend/app/services/grading_service.py — Needs Review routing
- tests/test_backend_grading.py, tests/test_detector_artifacts.py — existing test patterns
- reports/damage_test_report.json — BASELINE (87/8; 8/17 confusion, mps)
- data/damage/manifests/{train,val,test}.csv — 555/119/120 rows

## External Docs Fetched
- None needed. Local env: torchvision 0.18.1, ResNet18_Weights enum OK. Pretrained weight download requires network — verify before Phase 3; fallback if offline: train resnet18 without pretrained weights or use random init.

## Components
1. Phase 1: baseline verify + eval (scripts only, save report)
2. Phase 2: error audit CSV dump (filename/true/pred/conf/error_type), label inspection tooling; label corrections require USER decision
3. Phase 3: model factory (resnet18 + custom fallback), arch metadata in checkpoint, arch-aware backend loader, two-stage training
4. Phase 4: moderate augmentations, sampler vs class-weighted CE comparison, early stopping, scheduler, seed/hyperparam metadata
5. Phase 5: validation prob collection, threshold sweep 0.50-0.95, Needs Review policy, write threshold to metadata/config
6. Phase 6: experiment matrix + comparison report
7. Phase 7: copy winner to models/egg_grader_resnet18_v1.pth, backend regression tests (7 crop cases)
8. Phase 8: acceptance gate checklist

## Constraints
- Label 0 = Damaged, label 1 = Not Damaged (INVERTED from intuition; preserve everywhere)
- class_names checkpoint order: ("Damaged", "Not Damaged")
- Test set untouched until threshold finalized (leak prevention)
- Moderate aug only — no aggressive transforms that hide cracks
- Model selection: balanced accuracy + damaged recall, NEVER plain accuracy
- Reject model if acc>=90 but damaged recall regressed
- No regression in backend inference: damage_status, damage_confidence, damage_model_version, review_reason, Needs Review behavior
- Eval/eval reports under reports/
- Training runs on mps (device auto-select: cuda > mps > cpu in eval; train uses cuda-if-available — keep behavior)
- Python venv: ./.venv/bin/python; scripts run from repo root
- CAVEMAN MODE for all subagent delegation prompts; token-economy.md MANDATORY (but file doesn't exist — instruct terse output anyway)
- Approval gate: no writes/exec without approval; Phase 2 label edits = irreversible → user decides
- STOP on test fail/build error, report → propose → approve → fix

## Progress Log

| Date | Update |
|---|---|
| 2026-10-18 | Subtask 01 done: baseline eval re-verified, acc 0.8667, damaged recall 0.9158, confusion [[87,8],[8,17]], `reports/damage_baseline_report.json`. |
| 2026-10-18 | Subtask 02 done: `src/dump_damage_errors.py` + `tests/test_dump_damage_errors.py` (unittest, 3 tests pass; pytest not installed in repo). Tool reproduces baseline exactly: TP 87, TN 17, FN 8, FP 8. |
| 2026-10-18 | Subtask 03 artifacts generated: `reports/damage_error_audit.csv` (120 rows) + `reports/damage_error_audit_summary.md`. BLOCKED: waiting user label-triage decision (16 error images). No manifests modified. |
| 2026-10-18 | Commits: `20bfefa` (test cache), `8ce7ff5` (audit tooling + baseline audit). |
| 2026-10-18 | Tests: `python -m unittest discover -s tests` → 11/11 pass. |
| 2026-10-18 | Subtask 03 CLOSED. User verified all 16 FN/FP labels correct. No manifest edits; test set remains locked. Errors treated as genuine model errors → proceed to Phase 3 (model factory). |
| 2026-10-18 | Subtask 04 done: `src/model_factory.py` (build_model custom/resnet18, checkpoint_metadata) + `tests/test_model_factory.py` (6 tests). Pretrained-download fallback warns + random init. Full suite 17/17. |

## Exit Criteria
- [x] Baseline report reproduced/saved
- [x] Error audit CSV generated; FN/FP inspected (user-informed label fixes) — labels verified correct, no edits
- [ ] Model factory + resnet18 training path works; checkpoint carries arch metadata
- [ ] Backend loads arch from checkpoint metadata (custom + resnet18 both)
- [ ] Threshold sweep executed on val; selected threshold written to config/metadata
- [ ] Experiment comparison report in reports/
- [ ] Winner copied to models/egg_grader_resnet18_v1.pth (or best candidate name)
- [ ] Backend regression tests pass (7 crop cases + existing tests green)
- [ ] Acceptance checklist: acc>=90, bal>=85, recall>=95, FN<5, no leakage, backend OK
