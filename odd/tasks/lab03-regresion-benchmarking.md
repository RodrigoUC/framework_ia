# Lab03 regression benchmarking and Nexo IA

## Objective and authorization

Implement the user-approved functional Lab03 plan on `lab_03`, starting at
`2f770806ff75b1baecb85642650cda9a1f575535` (verified origin/lab_2).
Deliver all required regression estimators, reproducible intra/inter-family
benchmarking, CLI/export and Streamlit workflows, preserving the established
structure and visual design. Rename visible branding to **Nexo IA**.
User additionally authorized guided stages, actionable missing prerequisites and
disabled dependent functionality (2026-10-16).
User further authorized removing duplicate comparison/results views and reducing
textual noise across classification/regression: model execution and diagnostics
belong to model pages; Results is comparative only.
User clarified that configuration is general/shared, while each model retains
its own Train button. The proposed global training action was a misunderstanding
and is explicitly withdrawn; do not implement it.
Initial implementation authorized local Conventional Commits without an academic
report or remote delivery. The 2026-10-21 Concrete extension below supersedes
that initial scope: report, branch publication and one PR are authorized; merge
and submission ZIP remain excluded.

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
- Last reviewed boundary: 7c4fe426c4cf86d8c356ccad96c3dac421828e45.
- Delivery strategy: single-pr (approved plan), one requested local lab_03
  branch with work-unit commits. No actual PR is authorized.
- Updated forecast: 5,500-7,000 authored changed lines, generated files excluded;
  includes the user-requested guided workflow and prerequisite gating extension.
  About 400 lines per work unit is advisory, not a hard acceptance gate:
  do not omit tests/docs, minify, delete whitespace or split artificially.
- Running authored count: 5,723 (T01: 755; T02: 610; T03: 1,362; T03a: 97;
  T04: 1,310; T05: 671; T06: 918). T01 includes 116 tracking
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
- [x] T04 — Integrate regression UI, saved results/comparison and Nexo IA branding.
  Route: delegated (multi-file UI/state/tests/docs). Check explicit training,
  invalidation/recovery, branding, preserved styling and browser behavior.
- [x] T05 — Add reproducible Diabetes preparation, usage docs and Lab03 CI;
  run end-to-end reproduction and full compatibility checks. Route: delegated
  (scripts/docs/CI/tests). Record actual environment versions and outcome;
  preserve Lab02 lock/evidence and never fabricate empirical findings.

- [x] T06 — Guide the app through Source, Configuration, Execution and Results.
  Route: delegated (CodeGraph mapping and multiple non-trivial UI/test files).
  Derive readiness from actual prerequisites, not visited/completed flags.
  Explain missing requirements next to disabled dependent navigation/actions;
  retain correction routes and guard direct entry. Keep global preparation and
  manual partitions optional, EDA/clustering independent, and train-only safety.
  Check both labs, invalid/valid targets/features/splits, recovery, partial evidence,
  no implicit fitting, AppTest, full suite and real browser desktop/mobile.

- [x] T07 - Consolidate shared configuration and comparative Results.
  Route: delegated; mapping required 4+ files and implementation spans multiple
  non-trivial UI/test files. Remove duplicate Compare navigation, migrate old
  route IDs, keep execution/individual diagnostics on model pages, and present
  only comparative evidence in Results. Preserve individual test vs benchmark
  validation distinctions, partial errors, exports, RF criterion snapshots,
  state invalidation and prerequisite gates. Compact guidance and disclose
  secondary details rather than repeat explanatory blocks.
  Clarified scope: one Configuration page and canonical common state for task,
  target, predictors, split/seed and train-only preparation; task-specific controls
  remain conditional (stratification or simple predictor). Explicit task selection
  avoids guessing from numeric dtype; source presets may explicitly select task.
  Enable only the selected task's models when valid. Retain each model's own
  hyperparameters and Train button. Data keeps load/inspection/exclusion/restore;
  global transformations remain advanced exploration-only, not supervised prep.
  Centralize optional external supervised split controls in Configuration without
  conflicting percentages/seeds. Migrate legacy routes/presets/reset contracts.
  Latest accepted refinement: disabled navigation reasons belong in button hover
  help, not permanent sidebar paragraphs. Keep correction details in Configuration
  and operation errors near their action; preserve accessible control names.
  Preparation refinement: explicit null-row filtering on selected retained
  columns before supervised configuration; include target only if user selects it.
  Preserve original/default API behavior, record policy/removed counts, reject
  empty prepared result before publishing. Prepared data feeds configuration;
  cleanup invalidates stale evidence/partitions. Global learned imputation/scaling
  stays blocked; message offers reapplying cleanup without global transforms,
  not mandatory destruction of safe column/row cleanup. Test target-null behavior,
  no implicit fits, safe cleanup-to-regression, recovery and transform guard.
  Forecast: 1,500-1,950 authored lines; cohesive behavior/tests/docs, advisory only.
  Checks: focused AppTest/read-only route and isolation assertions, full pytest,
  changed-file Ruff, diff check, bounded desktop/mobile/keyboard browser QA,
  scoped design detector, canonical configuration/preset migration, task gating,
  shared-state invalidation and no fit on configuration/navigation. TDD remains off; existing .venv runner unchanged.
  Rollback boundary: T07 UI/test/docs hunks plus EDA preparation filter/metadata;
  preserve classifier/regressor domain behavior and earlier reference dataset.

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
  Commit: `5032478452540922d4b1c27a09db86c0e1a66e42` (1,273+/37-).
  Native assessment includes under-budget T03a: medium, slice_budget_reached,
  15 files / 1,399 net-range authored lines. User granted review; native
  reliability reviewer inspected all 15 immutable patches with no findings.
  Approved and exact acknowledgement applied; authority burned. Reviewer did
  not independently run tests; functional evidence remains the checks above.
  Candidate:
  sha256:866297ba5faec810b76402415d960fc791df1b312ce9a9d04ba7f5c55a0c06a2.
  Approved lineage: review-7573f86162a3afb3.
- T06 mapping verified: global preparation defaults to imputation and would
  block both supervised labs; do not force it. Manual split is optional in domain.
  Reuse native widgets/design and existing validators; no model fitting in readiness.
- T05 implemented: offline raw Diabetes provider/materializer with provenance,
  preflight conflicts/idempotence; explicit source Apply and one-shot post-sync
  preset; preserved CSV paths/manual edits; scoped ignores, usage/evidence docs,
  CI lab_03 plus offline smoke. No change to historical Lab02 lock/evidence.
  Full suite: 217 passed / 152 subtests in 156.69s; focused 98 passed in 50.41s.
  New-file Ruff/format, app F821, pip check, CLI helps and diff check pass.
  Full app Ruff retains 15 pre-existing code/message-equivalent findings.
  Premium audit interrupted without result while scanning virtualenv; GitHub
  Actions not run (local definition and equivalent commands checked).
  Actual offline CLI twice: 14 candidates / nine families / zero failures,
  train331/test111, validation folds67/66/66/66/66. Lasso alpha=.1 selected by
  validation RMSE56.055613 +/-3.573092; test RMSE53.281974. All39 files equivalent:
  29 byte-identical and ten JSON identical except argumentos.salida. Evidence
  temporary lab03-reference-20261016-130250/verificacion.json; versions/metrics
  documented in docs/lab03/VERIFICACION.md, not a final academic report.
  Real browser: source choice alone retained prior CSV; Apply loaded442rows,
  target/all10/bmi/test25/seed42; manual seed persisted through navigation.
  Actual14-candidate benchmark matched CLI and ZIP/JSON metadata/integrity passed.
  Screenshot page-2026-10-16T19-15-16-082Z.png inspected by parent. Four console
  errors came from antivirus-injected Kaspersky DNS requests, no app errors;
  browser network silence not claimed. No dataset upload/download required.
  Rollback: new provider/prep script/reference tests/data README/verification doc,
  T05 entrypoint source-preset hunks and CI/ignore/usage/design/README edits only.
  Commit: b38c24f (643 additions / 28 deletions). Native high risk due to
  CI shell source; review_due=true/high_risk, 12 files/671 lines.
  User granted native review; risk, resilience, readability and reliability
  inspected all12 immutable patches. Approved and acknowledged; authority burned.
  Advisory R4-001: interrupted direct file writes can leave truncated output
  blocking retries. Non-blocking; no correction opened, receipt remains closed.
  Record for separate scoped follow-up; no tests independently run by reviewers.
  Candidate: sha256:6e12153d0b292731c86f9a8dac450054b6f973ca5b8cd050a4e84cceb61acbc2.
  Approved lineage: review-821edbc8c7b50192.
- T06 implemented: shared pure readiness adapters reuse domain data/split checks
  without fitting; Source -> Configuration -> Execution -> Results guide with
  current context/next action, disabled dependent navigation and execution with
  visible missing requirements, direct-route guards and reachable correction.
  Configuration readiness and saved evidence are separate. Successful explicit
  fits rerun rendering once to enable results immediately, never refit on navigation.
  Global preprocessing and manual partitions remain optional; independent
  exploration/clustering are not forced through supervised stages. Partial
  regression candidates remain possible. No domain/source/preset behavior changed.
  Full final suite: 232 passed / 152 subtests in 254.16s (exit0), threads2.
  Focused flow/VC3/themes:34 passed /22subtests; VC4 updated navigation:3passed.
  Earlier full iteration231passed had one old test trying a now-disabled empty
  results link; updated to assert disabled then test the direct-route guard.
  New/changed UI modules and tests Ruff pass; entrypointF821 passes and full
  entrypoint Ruff retains exactly15 inherited code/message-equivalent findings.
  Diff check passes. Real browser blocked->corrected->enabled: remove predictors
  disables models/benchmark/Continue/results with specific reason; restore all10
  enables execution only; explicit Ridge train331/test111 unlocks results.
  Observed RMSE53.3182,R2.4859. Stage4->configuration keyboard Enter works.
  Desktop/light and mobile390/dark inspected; width390=scrollWidth390.
  Diabetes classification correctly blocks invalid stratification, regression
  remains available. No global preparation/manual split performed or required.
  Browser console12 accumulated antivirus Kaspersky DNS errors; no app errors.
  Parent inspected blocked screenshot20-07-13-780Z and mobile20-08-44-194Z in
  temporary lab03-browser-qa/.playwright-cli (2026-10-16). Local audit limitations
  and previously observed theme-rerender limitation remain as recorded in T04/T05.
  Rollback: new ui/flujo.py,ui/guia_flujo.py,tests/test_flujo_ui.py and T06 UI
  integration/tests/docs hunks only; preserve T01-T05 domain/reference/evidence.
  Commit: c24f0f36eb6a87c5c4cb33f245af9b83646e0f30 (873 additions /45 deletions). Native medium,
  review_due=true/slice_budget_reached,14files/918lines. User granted review;
  reliability reviewer inspected all14 immutable patches without findings.
  Approved and exact acknowledgement applied; authority burned. Reviewer did
  not independently execute tests. Candidate:
  sha256:af7fb85562a12f7223310e274973ac53a353d83b850a8d6ca884602ed1ff7100.
  Approved lineage: review-92d5583bc001d816. T05 advisory R4-001 remains recorded
  separately, not silently claimed fixed by this UX change.
- Authorized Lab03 and guided-workflow implementation is complete locally.
  No push, PR or merge performed. Non-blocking T05 R4-001 and the documented
  audit/theme limitations remain explicit follow-ups, not claimed fixed.


- T07 started after user request: CodeGraph mapping confirms Compare and Results
  dispatch the same renderers, with Results adding experiment evidence. Those
  renderers also embed individual diagnostics; shared guidance repeats context.
  Prior completed T01-T06 remain valid. Current working tree was clean at start.
  T07 uses existing single-pr strategy and local branch, no remote operations.
  Implementation and checks pending; native RDD mode verified on/global.

- T07 first increment implemented (uncommitted): unique comparative Results,
  legacy route aliases, model-only diagnostics and reduced guidance. Full tests
  233 passed /152 subtests in133.37s; focused109/22; changed Ruff/F821/diff and
  scoped detector[] pass;15 inherited entrypoint lint findings unchanged.
  Browser attempt not accepted: old server appeared to render prior imported
  modules, then QA paused for user clarification. Restart only owned server for
  final QA. No claim that current T07 browser checks passed.
- User clarified configuration is shared, training remains per model. T07 expanded
  before completion to canonical common configuration, explicit task choice and
  per-task availability. Delegated CodeGraph mapping confirms duplicated controls
  and separate clasif_/reg_ common state. Domain contracts remain unchanged;
  no numeric-dtype task guessing, no automatic fit. Existing single-pr strategy
  retained; larger cohesive scope includes migration/tests, no size-only splitting.
  Next: delegated configuration changes, all final checks, local work-unit commit
  and native assessment against4d1973b. Earlier receipts remain closed.

- T07 shared configuration and visual correction verified: full238passed plus
  152subtests in235.16s; changed Ruff/F821/diff pass. Scoped design detector[]
  earlier in this same increment. Fresh owned server8502 browser confirmed one
  configuration, task keyboard switch, per-model Ridge/RF training, comparative
  Results only, blocked empty predictors and recovery, desktop/mobile no clipping.
  Parent inspected final mobile22-14-31-139Z after removing duplicate route titles.
  No repeated benchmark/partial-errors/exports/external-split browser run; tests
  cover them. Existing8501 includes unowned server; not stopped or modified.
- User now requested disabled navigation reasons on hover instead of permanent
  text. Accept this presentation change; core writer owns narrow nav refinement.
  User also reported null cleanup blocked by train-only guard: separate mapping
  verified current UI offers global imputation but no null-row deletion; config
  already consumes dataset_preparado. Safe deterministic cleanup design pending,
  guard not weakened. Final commit/checkoff remains pending these refinements.

- Null-cleanup mapping reproduced: current UI only imputes globally, using all
  columns including future target; it has no drop-null-row action. Deterministic
  column exclusion/dedup already passes readiness and configuration already reads
  dataset_preparado. Add explicit selected-column row filtering, not a bypass.
  Mapper released as bounded writer after tooltip source settled; preserve T07.
- Hover refinement focused25passed in25.29s; local Streamlit button help/disabled
  API and AppTest proto.help verified. No permanent sidebar blocker paragraphs;
  enabled buttons have no stale help. Full suite deferred until cleanup increment.
  Process deviation: parent initially released tooltip writer before tracker
  amendment; writer applied authorized batch before hold arrived. Tracker then
  synchronized before continuing checks; no unauthorized scope or rollback needed.

- T07 final implementation and functional verification complete, local commit and
  native assessment pending. One common sup_* Configuration, explicit task and
  conditional controls, train-only preparation with optional fingerprint-bound
  external split. Each model retains own hyperparameters/Train/diagnostic;
  Results comparative-only with distinct test/validation evidence and exports.
  Legacy routes migrate, inactive family direct actions guarded, context/presets
  invalidate safely. Duplicate headings/copy removed; disabled nav help appears
  only on hover, while correction details stay in Configuration.
- Safe cleanup: EDA.preparar_dataset accepts explicit columnas_sin_nulos, preserves
  default API and original input, rejects empty result before UI publication,
  records removed-row counts/policy. Preparation widget choices persist across
  navigation; recovery reapplies safe cleanup without global statistical transforms.
  Model snapshots/export retain preparacion_previa. No change to historical
  classification missing-label handling; preparation itself never infers target.
- Final checks: full252passed/152subtests in334.25s, exit0 with thread limits2.
  Focused cleanup12passed, broader121passed in197.34s; prior tooltip25passed.
  Changed modules/newtests Ruff and entryF821 pass; inherited entry15/EDA5 lint
  code/message-equivalent to HEAD. Scoped detector[] and git diff --check pass.
  Initial obsolete recovery-text expectation corrected; behavior still blocked
  global learned transforms. No failed final tests; CI remote execution not run.
- Browser final evidence on owned http://127.0.0.1:8502: configuration/task gates,
  explicit Ridge and RF, comparison-only Results, desktop1440/light and390/dark,
  keyboard task change; final title correction confirmed desktop/mobile without
  retraining. Cleanup localCSV40rows with predictor+target nulls, explicitly select
  x/target ->38prepared, Ridge28train/10test. ZIP CRC and preparacion_previa counts,
  source/effective identity verified. Hover native help visible for disabled Results,
  permanent sidebar messages absent. Fresh console0errors/0warnings.
  Parent inspected images22-25-30-569Z tooltip and22-25-13-737Z cleanup in temporary
  lab03-browser-qa/.playwright-cli. Browser did not repeat full benchmark/partial
  errors/external-split/advanced-impute recovery after latest changes; tests cover
  these. Unowned8501 server untouched; updated app uses8502.
- Hosted12ui not run (external upload not authorized); earlier broad premium audit
  unavailable, scoped detector used instead. Existing theme-rerender limitation and
  separate T05 interrupted-write advisory remain, not silently claimed fixed.
  Final rollback includes configuracion_supervisada.py/test_configuracion_supervisada.py,
  test_preparacion_supervisada.py, EDA filter/metadata and T07 existing UI/tests/docs
  hunks only. No push, PR, merge or model-domain rewrite.

- T07 work-unit commit: 7c4fe426c4cf86d8c356ccad96c3dac421828e45,
  feat(ui): unify supervised configuration and simplify results.
  23 files,1271additions/590deletions=1861 authored lines (including tracker).
  Running feature count:7604 including prior passive closure20lines.
  Native assessment base4d1973b: medium,review_due=true/slice_budget_reached.
  Preflight and consent relay completed; REVIEW NOT STARTED/NOT APPROVED.
  Await explicit candidate consent for review-a2837d1160699986;
  target sha256:d8c5ed5dbbbf8c0469d3c7d431092f11f36b58d058c593072eac16520787ae53.
  Implementation task checked off on observed checks and work-unit commit;
  review closure was subsequently approved as recorded below. No push/PR/merge
  performed at this point.

- User granted T07 review and requested commit/push of all changes. Native
  reliability review inspected all23 immutable patches, approved without findings;
  exact acknowledgement applied and authority burned. Reviewer did not independently
  execute commands/tests; functional evidence remains252tests/152subtests above.
  Lineage review-a2837d1160699986; committed candidate7c4fe42 unchanged.
  All T07 implementation and review work complete locally. Closure notes now
  committed separately; push requested to configured origin lab_03, pending
  explicit authorization of credential/session under repository remote policy.
  No credential inspection or remote probe performed. No PR/merge requested.


## Concrete academic extension (2026-10-21)

Branch: `codex/lab03-entrega`, based on `lab_03` at `59abc33`.
Delivery: work-unit Conventional Commits and one PR into `lab_03`, using the
connected university account. No merge, ZIP or platform submission. The report
uses the Lab02 format and the four existing authors; complete standalone LaTeX
is the document deliverable. Dataset and generated evidence remain local.

- [x] T08 — Prepare official Concrete XLS, deduplicate exact rows only, preserve
  source and canonical hashes, expose a local idempotent CLI and provenance.
  Dataset input/output is separate from generic supervised models and UI.
- [x] T09 — Validate all 39 regression exports against data, provenance, common
  splits, the fourteen-candidate catalog, validation selection and predictions.
  Reject partial or inconsistent evidence without training or writing files.
- [ ] T10 — Generate an autonomous academic LaTeX report from validated evidence;
  add sourced foundations, taxonomy, real experimental analysis and bibliography.
- [ ] T11 — Reproduce Concrete twice, capture the environment, compile the report,
  run focused/full checks and publish one draft PR with exact verification status.

Baseline: 44 regression/algorithm/experiment/CLI tests and 93 subtests passed
in 29.34 s. Runtime currently resolves Python 3.11.8; historical versions above
are not assumed for this extension. Native RDD tools are absent from the current
tool inventory. Independent review and CI are recorded separately and are not
claimed as native RDD approval. The PR remains draft while that review is missing.

T08 verification: original UCI XLS 1030 rows, zero missing, 25 exact duplicate
rows removed; canonical CSV 1005 rows. Original SHA-256
`710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4`;
CSV `1eee7ee2b111a352cbadd30e49cc6a4f28e09f6ccd09636926008cefe74493a9`.
`python -m pytest -q -p no:cacheprovider tests/test_concrete_lab03.py`: 21 passed
in 5.85 s; real preparation and derivation reconstruction passed. Changed-file
Ruff and diff check passed; pip check reports no broken requirements. Rollback:
Concrete adapter/preparation/tests/docs, xlrd requirement and generated-path
ignore entries. Original classifier/regressor training behavior is unchanged.
Native RDD unavailable; independent parent readback found no unresolved issue.

T08 commit: `daa8dcc`; 529 additions/2 deletions, 531 authored lines.
T09 verification: 41 focused evidence tests passed in 19.62 s, including malformed
JSON/CSV, altered hashes, splits, catalog, selections, snapshot metrics and
recalculated predictions. Real Concrete exports loaded successfully with all
42 evidence/data hashes. Ruff and format passed. Independent read-only review
found no functional/scientific issue. Native RDD remains unavailable. Rollback:
remove evidencia_lab03.py and its isolated tests; existing export schema remains
unchanged. No model is trained during evidence validation.
