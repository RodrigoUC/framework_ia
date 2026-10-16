# Lab03 regression benchmarking and Nexo IA

## Objective and authorization

Implement the user-approved functional Lab03 plan on `lab_03`, starting at
`2f770806ff75b1baecb85642650cda9a1f575535` (verified origin/lab_2).
Deliver all required regression estimators, reproducible intra/inter-family
benchmarking, CLI/export and Streamlit workflows, preserving the established
structure and visual design. Rename visible branding to **Nexo IA**.
Local implementation and Conventional Commits are authorized. No push, PR,
merge, remote execution, final academic report or submission ZIP.

## Contracts and acceptance

- Keep `Supervisado -> Regresion`, `Progresion`, and RLS/RLM compatibility.
  Add Pythonic training, prediction, evaluation and experiment APIs. Logistic RL
  is not part of this lab and remains unchanged.
- Use dedicated regression/experiment results; no classification metrics,
  stratification or classification mutual-information selection.
- Reject invalid numerical targets; simple regression uses exactly one numeric
  predictor. Preprocessing learns only from the training partition/fold.
- Default seed 42, 25% reserved test, shuffled outer KFold(5), minimum mean RMSE
  selection; report MAE/MSE/R2 and fold deviation. Declaration order breaks ties.
- Fourteen default candidates: RLS/RLM; Lasso and Ridge alpha=.1/1; LassoCV and
  RidgeCV alpha grid=(.01,.1,1,10); RBF SVR C=1/10; decision tree depth=None/5;
  random forest 100 trees depth=None/5. CV estimators use explicit seeded
  inner KFold(5), and RidgeCV uses neg_mean_squared_error.
- Literal LassoCV/RidgeCV receive transformed train data for their internal
  alpha folds. Document that limitation; only outer validation is benchmark
  evidence. Do not claim internally fold-local preprocessing.
- External validation defaults to holdout preserving membership. Explicit CV
  leaves external validation unused/reserved. Test never selects models.
  Candidate failures remain visible; invalid data/splits abort before fitting.
- Reference dataset: load_diabetes(as_frame=True, scaled=False), target target,
  example simple predictor bmi supplied by the preset, not hardcoded in models.
  Continue supporting arbitrary CSV files.
- Navigation/comparison/results never train. Explicit actions save snapshots;
  changed data/context invalidate them, while failed same-context runs preserve
  the last successful result. Preserve current styles/native controls.
- Shared UI/CLI UTF-8 CSV/JSON exports include predictions, metrics, parameters,
  split indices, dependency versions and dataset provenance/hash.
- APP_NAME='Nexo IA' controls visible branding. Keep existing atlas CSS/private
  helper contracts, package/import names, and historical Lab02 evidence.

## Verification and delivery

- TDD: **off**, inherited from odd/tasks/lab2-workflow.md, Constraints and
  verification. Ordinary runner: python -m pytest -q.
- Isolated Python 3.12.14 environment ready at .venv; pip check clean.
  Pre-change baseline: 142 passed, 59 subtests passed in 175.29s (exit 0).
  Use .venv/Scripts/python.exe -m pytest -q once available.
- Focused unit/integration tests per task; complete existing suite at closure,
  Ruff for new/changed Python, AppTest and real-browser desktop/mobile,
  light/dark validation. Record missing or failed checks honestly.
- RDD: **on**, global; verified outside sandbox. Follow native candidate
  consent and transitions; never fabricate approval or toggle mode.
- Initial assessment: explicit inventory exclusion resolved cache discovery;
  committed range has 0 pending paths/lines, so no assessable candidate yet.
  Native assessment/preflight pending the first work-unit commit; no approval.
- Last reviewed boundary: 2f770806ff75b1baecb85642650cda9a1f575535.
- Delivery strategy: single-pr (approved plan), one requested local lab_03
  branch with work-unit commits. No actual PR is authorized.
- Forecast: 2,000-3,500 authored changed lines, generated files excluded.
  About 400 lines per work unit is advisory, not a hard acceptance gate:
  do not omit tests/docs, minify, delete whitespace or split artificially.
- Running authored count: 0. No implementation commits yet.
- Mirror: odd/lab03-regresion-benchmarking/tasks; full document plus locator.
  Parent owns reconciliation, task checkoffs, commits and native review.

## Tasks

- [ ] T01 — Implement linear regression core and result contracts, with focused
  tests and API documentation. Route: delegated (multiple non-trivial files;
  preparation and mapping). Check valid/invalid targets, train-only transforms,
  partitions, predictions, metrics, aliases and original framework tests.
- [ ] T02 — Add remaining required estimators and typed/configuration catalog,
  with estimator and parameter/CV tests. Route: delegated (multi-file behavior).
  Check all nine model entries, fixed seeds and literal CV-estimator behavior.
- [ ] T03 — Implement benchmark selection, shared serialization and CLI.
  Route: delegated (domain/orchestration/export/tests). Check identical splits,
  no test leakage, folds/holdout, ties, partial failures and CLI/UI export parity.
- [ ] T04 — Integrate regression UI, saved results/comparison and Nexo IA branding.
  Route: delegated (multi-file UI/state/tests/docs). Check explicit training,
  invalidation/recovery, branding, preserved styling and browser behavior.
- [ ] T05 — Add reproducible Diabetes preparation, usage docs and Lab03 CI;
  run end-to-end reproduction and full compatibility checks. Route: delegated
  (scripts/docs/CI/tests). Record actual environment versions and outcome;
  preserve Lab02 lock/evidence and never fabricate empirical findings.

Each task's work-unit commit contains its behavior, tests and documentation;
append commit identity, exact check outcomes, RDD assessment/outcome, authored
line count and rollback boundary here before considering the task closed.

## Progress and next step

- 2026-10-16: reconciled plan and Nexo IA amendment with local refs and memory.
  Created lab_03 from the verified Lab02 tip. Only pre-existing untracked
  .codegraph cache was present; no user source changes found.
- 2026-10-16: isolated dependencies and full Lab02 baseline passed before
  source changes. T01 delegated writer released. Dependency versions:
  sklearn 1.9.1, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, Streamlit 1.66.0,
  pytest 9.1.1, Ruff 0.17.1. Baseline thread limits: OMP/OPENBLAS/NUMBA=2.
- T01 implementation verified: RLS/RLM and Pythonic aliases, frozen result DTO,
  strict partitions/targets, train-only preprocessing, atomic successful state.
  Final full suite: `.venv/Scripts/python.exe -m pytest -q` => 151 passed,
  82 subtests passed in 146.64s (exit 0), thread limits OMP/OPENBLAS/NUMBA=2.
  Ruff core/result/new tests, format and git diff --check pass. Legacy
  test_framework.py has pre-existing RUF007 at line 165 (verified against
  HEAD); Ruff passes when ignoring that unchanged violation.
  Runtime boundary: exercised fit/predict/evaluate through tests; no UI in T01.
  Rollback: regresion.py, resultados.py, test_regresion.py, test_framework.py,
  docs/lab03/README.md only. Roughly 639 authored implementation/test/doc lines;
  cohesive validation and evidence require exceeding advisory size, not splitting
  tests away from behavior. Commit and native assessment pending.
- Next: commit verified T01, perform native assessment, then T02.

