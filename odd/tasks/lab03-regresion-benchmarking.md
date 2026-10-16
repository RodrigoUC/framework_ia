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
- Initial cache inventory resolved through explicit exclusion. T01 native
  reliability review approved without findings and exact acknowledgement applied.
- Last reviewed boundary: de6cffbe53b05abfe336a64043f9d904150d4054.
- Delivery strategy: single-pr (approved plan), one requested local lab_03
  branch with work-unit commits. No actual PR is authorized.
- Updated forecast: 4,500-5,500 authored changed lines, generated files excluded;
  complete UI/state/browser evidence exceeds initial estimate, scope unchanged.
  About 400 lines per work unit is advisory, not a hard acceptance gate:
  do not omit tests/docs, minify, delete whitespace or split artificially.
- Running authored count: 2,824 (T01: 755; T02: 610; T03: 1,362; T03a: 97). T01 includes 116 tracking
  lines and 639 behavior/test/doc lines; T02 includes tracking plus 571
  behavior/test/doc lines. Each commit remains a local single-PR slice.
- Mirror: odd/lab03-regresion-benchmarking/tasks; full document plus locator.
  Parent owns reconciliation, task checkoffs, commits and native review.

## Tasks

- [x] T01 — Implement linear regression core and result contracts, with focused
  tests and API documentation. Route: delegated (multiple non-trivial files;
  preparation and mapping). Check valid/invalid targets, train-only transforms,
  partitions, predictions, metrics, aliases and original framework tests.
- [x] T02 — Add remaining required estimators and typed/configuration catalog,
  with estimator and parameter/CV tests. Route: delegated (multi-file behavior).
  Check all nine model entries, fixed seeds and literal CV-estimator behavior.
- [x] T03 — Implement benchmark selection, shared serialization and CLI.
  Route: delegated (domain/orchestration/export/tests). Check identical splits,
  no test leakage, folds/holdout, ties, partial failures and CLI/UI export parity.
- [x] T03a — Verify and correct active-result atomicity when only the
  validation-best winner fails final test evaluation (advisory R3-001).
  Route: delegated (core and regression tests). Preserve the previous complete
  active snapshot while returning the partial attempt; never choose by test.
  This stays within the approved failed-run recovery contract. Separate work
  unit; the approved T03 receipt remains closed and is not reopened.
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
  tests away from behavior. Work-unit commit:
  `7bb1966b0f3895a1280124ce032c804f84e6d51e`.
  Native assessment: medium, review_due=true, slice_budget_reached; 6 files,
  755 lines. Cache-only .codegraph/.gitignore explicitly excluded using native
  canonical inventory. User explicitly granted candidate review (Si autorizo).
  Native compiled reliability reviewer inspected all six immutable patches;
  no findings. Approved and exact acknowledgement applied. Reviewer did not
  independently execute tests; functional evidence is the observed suite above.
  Candidate: sha256:b676a5370d24f6c532d2d63589e3340403e1fbdb8c65cba314ac5413de82474d.
  Approved lineage: review-16a6f56aeb1358f6.
- T02 implementation verified: nine native estimators, fourteen independent
  default candidate configurations, typed UI catalog and Pythonic wrappers.
  Literal CV estimators retain native KFold objects; result parameters encode
  splitter settings and metadata records selected alpha and inner-preprocessing
  limitation. No benchmark/UI/export implementation included yet.
  Final full suite: `.venv/Scripts/python.exe -m pytest -q` => 158 passed,
  152 subtests passed in 118.11s; Ruff, format and git diff --check pass.
  Runtime boundary: native estimator fit/predict plus native-CV equivalence,
  deterministic results and holdout independence exercised through tests.
  Rollback: utils/configuracion_regresion.py, utils/__init__.py,
  modelos/supervisado/algoritmos_regresion.py, modelos/supervisado/regresion.py,
  tests/test_regresion_algoritmos.py, docs/lab03/README.md; preserve T01.
  Approximately 571 authored behavior/test/doc lines form one coherent catalog
  unit; tests and documentation remain alongside behavior. Work-unit commit:
  `94d6768b621e50322f7381303abf5f8108e634fa` (595 additions, 15 deletions).
  Native assessment: medium, slice_budget_reached, 7 files / 610 lines;
  User granted separate candidate consent. Native reliability review inspected
  seven immutable patches, approved without findings; acknowledgement applied.
  Review did not independently execute tests; suite evidence recorded above.
  Candidate: sha256:380d63275619122c868dbec52e73d4952d399bda2595e9040f5682827bb631b4.
  Approved lineage: review-bdf7cdc359e9055d.
- T03 implementation verified: shared outer folds, isolated RLS predictor,
  validation-only selection, reserved external validation under explicit CV,
  visible partial failures and preserved last success on all-failed retries.
  Shared individual/experiment CSV+JSON export and CLI implemented; stored fit
  versions take precedence over export-time values, source bytes retain SHA-256.
  Final full suite: `.venv/Scripts/python.exe -m pytest -q` => 184 passed,
  152 subtests passed in 123.50s (thread limits OMP/OPENBLAS/NUMBA=2).
  Focused: 42 passed, 93 subtests in 12.23s. Ruff check/format, CLI help and
  git diff --check pass. Initial lint findings corrected before final checks.
  Runtime: actual CLI on local synthetic CSV, default14 candidates / 9 winners /
  0 errors, exit0; smoke artifacts outside repository in temporary storage.
  Rollback: new experiment/export/CLI modules and two test files, plus only T03
  additions to regresion.py, resultados.py and docs/lab03/README.md; keep T01/T02.
  Approximately 1,326 behavior/test/doc lines are cohesive orchestration plus
  no-leakage/export/CLI checks; advisory size exceeded without removing coverage.
  Commit: `de6cffbe53b05abfe336a64043f9d904150d4054` (1,352 additions / 10
  deletions including tracking). Native risk medium, slice_budget_reached,
  9 files / 1,362 lines. User granted review; approved and acknowledged with
  non-blocking advisory R3-001 (partial final-evaluation failure can mix active
  experiment/model runs). No correction transition opened; separate T03a will
  verify the advisory against the existing atomic-state contract. Reviewer did
  not rerun tests. UI and preset reproduction remain pending.
  Candidate: sha256:11e912dca6043e84bcb16849594b9ffba70a086a3894bed06282366e7034f082.
  Approved lineage: review-58c88d65923322dc.
- T03a: deterministic fresh/prior-success reproductions both failed before the
  fix. Root cause: experiment published when any family succeeded, while other
  active artifacts depended on the validation-best result being available.
  Fixed by a shared publication condition for all active artifacts; partial
  DTO remains returned and no substitute is selected from test performance.
  Final checks: focused 44 passed / 93 subtests; full 186 passed / 152 subtests;
  Ruff check/format and git diff --check pass. Three-file rollback: regresion.py,
  test_regresion_experimentos.py and relevant README paragraph (51+/13-).
  Original approved receipt untouched. Separate commit
  `51661e3252eea494c9f279c434831867a042285a` (79+/18- including tracking).
  Native assessment: medium, review_due=false, under_budget. This 97-line
  slice remains pending until the next due work unit; reviewed boundary stays
  de6cffb (no approval fabricated).
- T04 implemented: isolated regression configuration/state/native parameters,
  nine model views, experiment/partial-error views, read-only comparison/results,
  numeric diagnostics/shared downloads, source/preparation/split invalidation,
  APP_NAME Nexo IA and current design docs (private atlas contracts preserved).
  Visual ordering corrected after screenshot inspection: metrics, chart, table,
  downloads; explicit renderer-order assertion added.
  Final suite: 207 passed / 152 subtests in 298.00s. Focused UI 21 passed;
  earlier UI/themes/Lab02 focused 84 passed / 22 subtests. Ruff newfiles/tests,
  F821 entrypoint/visualization and git diff --check pass. Twelve C408 warnings
  in shared visualization were verified unchanged from HEAD, not unrelated-fixed.
  Real Chromium: uploaded synthetic80-row CSV, explicit RLM train60/test20,
  saved navigation/comparison, default14 benchmark / 9 families, individual and
  experiment downloads (ZIP integrity verified), console0 errors/0 warnings.
  Desktop1440x1000/mobile390x844 light/dark inspected; no horizontal overflow,
  popup/sidebar scrolling and Tab/ShiftTab/ArrowDown/Enter/Escape verified.
  Final light desktop screenshot: temporary lab03-browser-qa/.playwright-cli/
  page-2026-10-16T18-45-02-366Z.png, inspected by parent. Processes retained
  for T05: Streamlit26932/browser43892, isolated CLI session lab03qa.
  Limits: no-data state verified by AppTest, not browser (example auto-loads).
  Theme switch can leave rendered chart colors stale until ordinary rerender;
  rerender resolves it; inherited origin unproven. Local premium audit stopped
  without result because it scans .venv frontend bundles; not claimed passed.
  Hosted12ui not run (remote upload outside authorized scope); local maintained
  design/craft checks used. Rollback: five new UI modules/new UI tests and T04
  edits in entrypoint, visualization, current design/config/docs only; keep domain.
  Commit/native assessment pending, including the under-budget T03a slice.
- Next: commit/assess T04, then T05 offline Diabetes preset/reproduction/CI.

