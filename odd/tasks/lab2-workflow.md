# Complete the classification workflow and clarify clustering

## Objective and authorization

Implement the plan authorized by the user on 2026-10-09. Repair confirmed
classification, comparison, and results defects; simplify navigation and class
configuration without duplicating model logic. Keep the existing Atlas visual
identity and Spanish application copy. Local development and work-unit commits
are authorized; push, PR creation, merge, data publication, and remote execution
are not authorized.

## Problem and scope

The current application mixes legacy classification and experiment routes,
groups K-means/K-medoids behind a selector, and exposes projection techniques as
standalone destinations. The user reports execution and results errors that
still require reproduction. EDA already owns exploratory chart generation;
DataFrame remains responsible for tabular operations. The utils package exists
but does not yet provide consistent typed class configuration.

## Constraints and verification

- TDD: off; no project-level setting exists. Prior task documents explicitly
  use ordinary functional verification. Source: odd/tasks/streamlit-theme-contrast.md
  and odd/tasks/issues-5-7-visual-contrast.md.
- Test runner: python -m pytest -q, as documented in docs/lab02/README.md and CI.
- Current environment: Python 3.14.7. A compatible isolated environment is now
  available at `/tmp/framework-start-lab2-venv` (created with system-site
  packages and project dev requirements installed); it resolves Streamlit 1.65.0,
  pytest 9.1.1, xgboost 3.4.1, and ruff 0.16.10. The user's global Streamlit
  1.63.0 was not modified. Pip cache is isolated at
  `/tmp/framework-start-lab2-pip-cache`.
- Preserve train-only preprocessing, shared reproducible partitions, selection
  on validation, and final evaluation on test. No fabricated academic findings.
- Synthetic fixtures prove software behavior, not professor-dataset outcomes.
- The user explicitly authorized local Streamlit and browser QA on 2026-10-09.
  The server is running only on `127.0.0.1:8502`, using the isolated environment.
  No network exposure beyond loopback, remote publication, or main integration
  is authorized.
- Keep changes Pythonic, typed where useful, and independent of Streamlit in
  domain/configuration modules. Separate model screens must reuse shared logic.
- Engram mirror: pending. Runtime session registration is unavailable; developer
  instructions prohibit agent-attributed memory mutations until restored.

## Delivery and review

- Route: delegated direct for implementation; multi-file logic and preparation
  trigger delegation. One writer at a time; fresh verification when required.
- Forecast: 500–800 authored changed lines, excluding generated artifacts.
  This is an advisory estimate, not a code-size acceptance criterion.
- Delivery: local work-unit commits directly on lab_2, explicitly requested by
  the user. Do not integrate into main. PR/chain planning is deferred because no
  PR or integration is authorized; no additional delivery choice is needed.
- Work-unit commits stay local on lab_2. No PR is created implicitly.
- RDD: on (global), observed with gentle-ai review mode status. Candidate consent
  remains separate. Last reviewed boundary: `44b4989` (T5 approved/acknowledged;
  authority burned). T3 and T4 were reviewed together; their nonblocking
  coverage warning was resolved in T5.
- Authored running count: 1,908 through T5 (T2 538, T3 256, T4 445, T5 669
  additions/deletions including task-document changes). Work-unit commits:
  T2 `d3fe83a`, T3 `d0e1fba`, T4 `780a868`, T5 `44b4989`.
  T6 documentation changes remain uncommitted for the parent-owned work-unit
  commit and any applicable candidate assessment.

## Checklist

- [ ] T1 — Establish a compatible test environment and reproduce the reported
  failures; add regression coverage for confirmed defects.
  Route: delegated; execution and investigation cross the preparation boundary.
  Checks: focused classification/UI tests; record exact exceptions and baseline.
  Environment evidence: `/tmp/framework-start-lab2-venv/bin/python` is Python
  3.14.7; requirements-dev.txt installed successfully after the sandbox's PyPI
  DNS failure was retried with approved network access. No global packages were
  changed. Focused baseline command (thread limits set, pytest cache disabled):
  `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
  /tmp/framework-start-lab2-venv/bin/python -m pytest -p no:cacheprovider -q
  tests/test_lab2_classification.py tests/test_lab2_ui.py tests/test_lab2_cli.py
  tests/test_vc4_classification_panels.py` — 39 passed, 22 subtests passed in
  17.67s. Full baseline with the same environment/thread limits and `-q` — 76
  passed, 52 subtests passed in 40.72s. No exception/traceback surfaced in
  these existing suites, so user-reported training/benchmark/results failures
  are not yet classified or confirmed; no regression test/source change made.
- [x] T2 — Consolidate typed configuration in utils and shared view helpers.
  Route: delegated; multiple non-trivial files.
  Outcome: added a Streamlit-independent typed classification registry in
  `framework_ia/utils/configuracion_clasificacion.py`. The registry now owns
  classifier labels, parameter-widget metadata, selection metric labels, and
  experiment candidate defaults. The parameter view renders its controls from
  that metadata, and the experiment domain and existing `configuraciones_lab2`
  API consume the shared fresh-copy factory. Naive Bayes remains outside the
  five-model LAB02 registry for legacy compatibility. Cluster's algorithm
  defaults remain on its public model methods: they are direct runtime
  parameters, not duplicated UI/experiment configuration.
  Route evidence: delegated direct writer; T2 changes six Python files and this
  task record. No navigation, clustering behavior, or preprocessing changes.
  Checks: focused command — `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 /tmp/framework-start-lab2-venv/bin/python
  -m pytest -p no:cacheprovider -q tests/test_lab2_classification.py
  tests/test_lab2_ui.py tests/test_lab2_cli.py
  tests/test_vc4_classification_panels.py` — 40 passed, 22 subtests passed in
  15.46s. Full command with the same environment and `-q` — 77 passed, 52
  subtests passed in 35.90s. `ruff check` on all six touched Python files passed;
  `ruff format` was applied before the final full-suite run. The UI regression
  test caught and the writer corrected an invalid `format_func=None` call for
  selectboxes; focused and full suites pass after correction.
  Runtime harness: N/A; no Streamlit server was launched. Rollback boundary:
  remove the new shared classification config module and restore the previous
  local experiment catalog and parameter-widget branching; no navigation or
  clustering methods depend on this work. Commit `d3fe83a`; 538 authored lines;
  last reviewed boundary is `d3fe83a`.
- [x] T3 — Restructure clustering: contextual EDA/ACP, distinct K-means and
  K-medoids actions, t-SNE/UMAP visualization within K-means, no standalone
  dimensional-reduction destination.
  Route: delegated direct; changed the navigation and shared rendering adapters
  in `framework_ia/ui/streamlit_app.py`, added clustering navigation/state tests,
  and updated `DESIGN.md` plus `.superdesign/design-system.md`. EDA charts remain
  intact; ACP is explicitly run within EDA; K-Means and K-Medoids have separate
  destinations; t-SNE/UMAP remain explicit projections inside K-Means.
  Checks: focused UI/navigation/classification panel suites — 24 passed in
  11.27s; full suite — 79 passed, 52 subtests passed in 30.69s. `git diff
  --check` passed. `ruff check` was run and reports the same 18 pre-existing
  issues present on HEAD across the app module; no new test-code issues were
  reported.
  Runtime harness: N/A; no Streamlit server/browser was authorized. Rollback:
  revert the navigation and shared rendering adapter changes, the two design
  map updates, and their navigation tests. Commit `d0e1fba`; 256 authored lines,
  79 verified
  lines + 52 full-suite subtests and parent spot-check of 22 focused cases.
  Native assessment for `d3fe83a..d0e1fba` was medium / under_budget; reviewed
  boundary remains `d3fe83a`, pending slice count 256. No approval is inferred.
- [x] T4 — Give every available classifier its own action/view and add the
  source-target balance chart before training.
  Route: delegated direct; shared classifier renderer and EDA-owned chart.
  Outcome: KNN, decision tree, Random Forest, XGBoost, AdaBoost, and Naive Bayes
  each have a separate sidebar route and call the shared renderer. Parameters
  and results are keyed per classifier; Naive Bayes uses the existing public NR
  model entry point without entering the five-model experiment catalog. The
  selected source target is summarized with class counts/proportions, a separate
  missing-value count, rare-class warnings, and an EDA-owned bar figure before
  model execution. Numeric labels use an explicit categorical chart axis;
  collision-safe class data supports target names `Cantidad` and `Proporción`;
  unobserved categorical values are excluded from stats and rarity warnings.
  No chart or route trains a model implicitly. Removed the
  legacy LAB02/individual-model wording from visible navigation and corrected
  `_render_eda`'s ownership docstring.
  Checks: AppTest covers all six separate routes and model execution, absence of
  algorithm selector, per-model settings/results preservation, numeric classes,
  missing target values, rare classes, spaced numeric labels, target names that
  collide with metric labels, unobserved categories, missing-denominator
  proportions, and no target mutation. Focused command (four UI/classification
  suites) — 45 passed, 22 subtests passed in 18.39s. Full command with thread
  limits and pytest cache disabled — `OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
  /tmp/framework-start-lab2-venv/bin/python -m pytest -p no:cacheprovider -q` —
  82 passed, 52 subtests passed in 32.70s. `git diff --check` passes. Ruff checks
  passed on touched modules except `eda.py` and `streamlit_app.py`, which retain
  pre-existing lint findings; no new findings were reported in the changed
  lines. Runtime harness: N/A; no server launched.
  Rollback: revert the T4 navigation/configuration/rendering, EDA balance factory,
  state scoping, and corresponding tests; no domain training API was changed.
  Commit: `780a868`; reviewed with T3 and acknowledged, authority burned.
- [x] T5 — Unify comparison and results UX and repair confirmed execution/state
  defects. Remove LAB02 and Modelo individual from user-facing copy.
  Route: delegated; experiment rendering, state, and navigation.
  Outcome: a separate legacy comparison-path mismatch was confirmed: its
  renderer looked only in `resultado_modelos_clasificacion`, while T4's actual
  trainer stores each individual result as `resultado_lab2_modelo_<code>`. This
  mismatch is distinct from the original user-reported crashes, which remain
  unconfirmed.
  Removed the obsolete duplicate comparison route/renderer and made one
  read-only Results route render individual-model results and validation-based
  experiment outcomes in named sections. Compatible legacy RF/NR results are
  projected into that destination without training. Results include a compact
  side-by-side table of each trained algorithm's test metrics and effective
  parameters, explicitly labeled exploratory and not for selection by test;
  expanded details remain available per model. They show dataset, target, and
  features for individual runs, and configuration/metric plus validation-vs-test
  provenance for experiments. The route compares current data/configuration
  signatures before rendering, so stale results disappear even when the user
  navigates directly to results; it renders no training widgets or computation.
  If fewer than two data columns remain, it clears stale results and shows an
  actionable unavailable state. Comparison setup now has preset, standard-only, and
  advanced JSON variant modes with an explicit candidate review and run step.
  Candidate failures, no-winner states, and test-evaluation errors remain
  visible/actionable. T4's omitted RF/NB comparison assertions were restored.
  Checks: focused command (five required suites) — 54 passed, 22 subtests passed
  in 20.92s; full suite — 86 passed, 52 subtests passed in 35.83s. `ruff format`
  applied to changed modules/tests except `streamlit_app.py`, whose original
  formatting was retained to avoid reformatting unrelated code; import-only Ruff
  normalization was applied there. `git diff --check` passed. Full Ruff check remains non-clean with 16
  pre-existing app findings after removing unreachable legacy code; no unrelated
  lint cleanup was attempted. Runtime harness: N/A;
  no Streamlit server launched. Rollback: revert the T5 results/configuration
  state helper, shared results renderer, route removal/migration, and associated
  tests; no classifier training/domain API changed. Side-by-side
  metrics/parameters and validation-vs-test semantics are covered; candidate
  navigation is asserted to call neither training nor experimentation. T5
  authored count: 669 including this document update; commit `44b4989`.
  Native review approved and acknowledged with no findings; authority burned.
  Last reviewed boundary: `44b4989`.
- [ ] T6 — Run full regression and bounded visual/accessibility verification;
  reconcile README and canonical UI maps with observed behavior.
  Route: delegated verification; parent structural readback and spot check.
  Observed: independent full regression with thread limits and cache disabled
  after the bounded source correction — 86 passed, 52 subtests passed in 34.31s.
  A bounded AppTest smoke
  using 60 synthetic rows with noncontiguous indices executed real K-Means,
  t-SNE and UMAP; both projections returned finite 60×2 coordinates aligned to
  the cluster labels, color traces matched cluster counts, navigation reused
  results, and parameter changes hid stale projections. The changed-UI
  Impeccable detector returned `[]` for the five requested source modules.
  Read-only Ruff exposed a new `F821` at `streamlit_app.py:1384`: an obsolete
  `comparacion` branch called undefined `_render_comparacion`. Normal route
  selection migrated that view to Results, so navigation tests did not exercise
  the branch. The parent confirmed RED with `ruff --select F821` and removed
  exactly that dead branch. Independent GREEN recheck of
  `ruff check framework_ia/ui/streamlit_app.py --select F821` passed. Full Ruff
  on the five changed modules remains non-clean with 21 other findings already
  present before this bounded correction; no unrelated lint cleanup was done.
  The parent owns the source fix; this verification worker made no source edits.
  `DESIGN.md`, `.superdesign/design-system.md`, `README.md` and
  `docs/lab02/README.md` now describe the observed navigation, six classifier
  views, configuration variants and unified read-only results. Headless checks
  do not verify rendered contrast, light/dark appearance, mobile viewport or
  keyboard behavior. Parent focused verification passed 26 tests in 13.53s;
  structural readback and documentation reconciliation completed. The bounded
  cleanup/docs unit was committed as `08102d9` (174 authored lines); native
  assessment was medium / under_budget against `44b4989`, not a review approval.
  The user subsequently authorized server/browser QA. The local server health
  endpoint returned `ok`; owned foreground session `73714`, PID `106489`.
  CUA had no connected browser; the user explicitly authorized isolated
  Playwright/Chromium QA. The approved escalated launch rendered HTTP 200 with
  no page errors. Captures at 1440px light/dark, 1024px light and 390px light
  showed no horizontal page overflow. App text/background contrast measured
  14.35:1 light and 16.94:1 dark (not an exhaustive control/chart audit).
  K-Means, KNN, comparison and results routes rendered; Tab produced visible
  focus. Evidence: `/tmp/lab2_qa_*.png`; parent inspected KNN desktop and the
  collapsed-sidebar mobile capture. At 390px the initially expanded sidebar
  obscures the main content until collapsed. Full keyboard-only navigation,
  screen-reader behavior and exhaustive chart/control contrast remain pending.
  T6 remains partial; no source change was made during this QA.
- [ ] T7 — Produce real laboratory evidence and report material after verifying
  the required professor datasets, target columns, provenance, and report template.
  Route: delegated when prerequisites are provided.
  Status: prerequisite pending; only data/ejemplo_analisis.csv is present.
  Checks: five required model families, standard/variant validation comparison,
  final test metrics, reproducibility exports, evidence-based report.
- [x] T8 — Separate EDA and ACP from the Clustering navigation group.
  Route: delegated direct; user correction to the T3 navigation composition.
  TDD: off, per existing project ODD record; runner is the isolated Python
  environment documented above. Keep EDA and ACP as independent destinations
  outside Clustering; Clustering contains K-Means, K-Medoids, and HAC only.
  Preserve ACP's explicit execution and current rendering/domain API, EDA's
  existing charts, and the K-Means-only t-SNE/UMAP projection composition.
  Update canonical design maps and targeted navigation regression coverage.
  Checks: clustering navigation has no EDA/ACP entries; EDA and ACP remain
  independently reachable; existing clustering/projection behavior is retained;
  focused UI tests, full pytest suite, `ruff check --select F821` on affected
  Python modules, and `git diff --check`.
  Outcome: EDA and ACP now have separate navigation buttons under Exploration;
  the Clustering expander contains K-Means, K-Medoids, and HAC only. EDA no
  longer renders clustering controls, and ACP retains its independent explicit
  action and unchanged overlay tabs. Domain methods and K-Means projections are
  unchanged. Focused suites — 31 passed in 15.56s; full suite — 86 passed,
  52 subtests passed in 40.95s. `ruff check --select F821` and `git diff --check`
  passed. Runtime/browser inspection is handed to the parent; no server launch
  or overlay redesign occurred. Parent to record exact T8 authored count,
  commit identity, and review boundary after spot check.
- [x] T9 — Keep accessible descriptions compatible across installed Streamlit
  versions. TDD: off; ordinary functional checks. Route: bounded delegated
  writer, preparation spans Streamlit APIs, app rendering, and regression tests.
  Root cause: global Streamlit 1.63's actual `DeltaGenerator.dataframe` and
  `plotly_chart` signatures omit `alt`; isolated Streamlit 1.65 adds
  `alt: str | None = None` to both. T8's AppTest only exercised 1.65, so the
  unsupported arguments went undetected there. All eight table/chart
  descriptions are now rendered as visible Streamlit captions and never passed
  as unsupported kwargs. A real global 1.63 AppTest executed experiment and
  results routes with synthetic data and confirmed tables/charts and captions
  render. Isolated 1.65 AppTest asserts candidate, comparison, figure, confusion,
  and prediction descriptions remain visible. T8 changes are preserved.
  Checks: focused T9 suites — 26 passed in 13.29s; full suite — 86 passed, 52
  subtests passed in 34.48s; `ruff check --select F821` on `lab2.py` and
  `streamlit_app.py`, `ruff format --check` on changed implementation/tests,
  and `git diff --check` passed. Global 1.63 was exercised with in-process
  AppTest (not a server). No commit, remote work, or main merge.
- [x] T10 — Move shared classifier configuration and reproducibility controls
  into one dedicated classification-setup destination. TDD: off per this task
  document; use the isolated pytest environment above. Render the target,
  features, partition, preprocessing, and target-balance panel only on setup;
  all classifier and experiment routes consume that saved configuration.
  Preserve model-specific parameter controls and existing signature/state
  invalidation. Keep dataset health counts on Data and EDA only, not classifier
  screens or ACP. Before setup, classifier destinations must either use the
  viable shared defaults or show an explicit actionable setup path. Update
  accessible navigation copy and AppTest coverage for shared configuration,
  multiple models, stale-result invalidation, and absence of repeated dataset
  metrics/configuration. Check F821, format, focused and full suites, and diff.
  Outcome: one dedicated Configuration section/button owns target, predictors,
  balance chart, split/seed/global-partition choice, and preprocessing controls.
  Classifier and experiment destinations consume the same persisted settings;
  individual screens retain per-model parameters. Defaults remain usable before
  visiting setup; an underspecified dataset shows an actionable setup path.
  Dataset health counts are rendered only on Data and EDA. Documentation now
  describes the shared flow. Tests assert common target/features and identical
  train/test indices for multiple model families and experiments, invalidation
  across model/experiment results when shared settings change, absence of
  repeated configuration/health metrics, and the insufficient-dataset path.
  Focused suites — 52 passed, 22 subtests passed in 20.32s. Full command with
  thread limits and pytest cache disabled — `OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
  /tmp/framework-start-lab2-venv/bin/python -m pytest -p no:cacheprovider -q` —
  89 passed, 52 subtests passed in 35.41s. `ruff check --select F821` and
  `ruff format --check` passed; `git diff --check` passed. No server launched.
  Rollback: revert the new configuration route, renderer dispatch, header health
  scoping, shared-flow tests, and README clarification; T8/T9 edits remain
  independently identifiable in the worktree. Commit and review remain pending
  parent ownership; no remote or main-branch changes.
- [ ] T11 — Correct ACP biplot variable-label collisions and clipping.
  Route: bounded delegated direct correction, explicitly authorized after T10
  browser evidence. TDD is off per this task document; use the isolated pytest
  environment. Keep `VisualizadorNoSupervisado.sobreposicion_acp` as the sole
  biplot renderer and preserve exact observation coordinates, loading scores,
  arrow endpoints, and correlation-circle geometry. Improve only label layout
  and chart margins/responsive space; do not distort axes to force labels into
  view or add a duplicated renderer. Regression fixtures must prove source
  coordinates/loadings remain unchanged and labels avoid overlaps/clipping for
  multiple feature counts and left/right crowded cases. Verify both approved
  captures (1024 dark, 1440 light) after implementation; do not expand into
  mobile/configuration scope. Checks: focused graph tests, full pytest, F821 and
  formatter checks on affected files, plus `git diff --check`.
  First-pass browser evidence was partial: four labels no longer clipped, but
  `scaleanchor` expanded the visible ranges to approximately ±5.22/±3.63 and
  diluted the data; at 564px, twenty long labels had only 10px spacing and
  overlapped. Evidence: `/tmp/lab2_t11_acp_overlay_1440_light.png`,
  `/tmp/lab2_t11_acp_overlay_1024_dark.png`, and
  `/tmp/lab2_t11_crowded_overlay_564_dark.png`. One bounded correction is
  authorized. The correction keeps the observation/loading values and arrows
  unchanged, fixes the axis ranges while constraining the plot domain, places
  annotations in separated left/right paper-coordinate bands, and increases
  figure height proportionally to the busiest label column. Focused graph tests
  pass on 4/12/20 crowded features; exact ranges, endpoints, coordinates, paper
  placement, spacing, and correlation-circle geometry are asserted. The focused
  result is 2 passed. Full isolated thread-limited suite — 91 passed, 52
  subtests passed in 35.44s. `ruff check --select F821` passed for the two
  affected Python files; affected-range `ruff format --check` and `git diff
  --check` passed. Parent recapture of light/dark/crowded cases remains pending;
  source is frozen after this single correction batch pending visual
  confirmation. Browser was not launched by this worker.
- [x] T12 — Remove duplicate Data/preparation dataset-health summaries without
  removing distinct quality indicators or changing EDA's summary. TDD is off;
  use the isolated Python test environment above. Establish one canonical
  Data/preparation summary covering rows, columns, nulls, duplicates, missing
  percentage, numeric/categorical counts, and outliers; remove repeated header
  and expander presentations there. Keep EDA's distinct summary intact. Add
  AppTest assertions for one visible value per health metric and verify
  preparation changes update the canonical summary. Checks: focused AppTest,
  full pytest, F821, affected formatter check, and `git diff --check`.
  Outcome: removed the duplicate header and expander summaries from
  Data/preparation. Its single metric summary retains row/column/null/duplicate
  counts, missing percentage, numeric/categorical counts, and outliers; the EDA
  expander summary remains intact. AppTest confirms each health metric appears
  once, duplicate removal updates row/duplicate values, and EDA still shows its
  distinct summary. Focused UI/layout tests — 28 passed in 12.83s. Full command
  with thread limits and pytest cache disabled — `OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
  /tmp/framework-start-lab2-venv/bin/python -m pytest -p no:cacheprovider -q` —
  91 passed, 52 subtests passed in 35.12s. `ruff check --select F821`,
  `ruff format --check tests/test_lab2_ui.py`, and `git diff --check` passed.
  Whole-file formatter check for `streamlit_app.py` still requests broad
  pre-existing formatting changes; HEAD has the same baseline failure, so no
  unrelated reformat was applied. No server or browser change.
  Rollback: restore the header's Data/EDA metric block and the Data page's
  duplicate expander call, remove the outlier metric addition and its regression
  test. No dataset or EDA domain behavior changed.

## Progress and next step

T1 remains open pending the concrete user error scenario/traceback. T2–T5 are
committed on `lab_2`; all due native reviews were approved and acknowledged,
with T5 the latest reviewed boundary. T6's headless regression and projection
smoke passed, and documentation was reconciled. The parent removed the dead
`comparacion` branch, the independent `F821` recheck passed, and the full suite
passed again on the corrected bytes. The parent committed the cleanup/docs as
`08102d9` and completed the structural readback and focused check. Authored
commit count through that unit is 2,082 (538 + 256 + 445 + 669 + 174); pending
review slice is 174 lines after `44b4989`. Authorized Playwright browser QA
rendered the principal routes without page errors or horizontal overflow;
bounded contrast/focus checks passed, with the mobile expanded-sidebar issue
and full accessibility coverage explicitly pending.
Next: address the mobile first-view issue and complete keyboard-only coverage.
T8 corrects T3's former combined EDA/ACP-in-Clustering UI grouping per the
user's clarification; ACP is now an independent Exploration destination.
T9 verified the version-dependent `alt` API mismatch and uses visible
descriptions for both table and Plotly output; the 1.63 and 1.65 runtime checks
passed. T8/T9 changes are preserved. T10 implementation and verification are
complete. Independent verification passed 89 tests and 52 subtests in 34.79s,
F821 and diff-check. Browser KNN and Random Forest training with unchanged
shared settings succeeded, with both outputs in Results and no `alt` TypeError
or dataset health tiles on model screens. Comparison setup rendered; browser
benchmark execution was not checked. The owned localhost server was restarted
to load current imports (session `73126`).
ACP overlay visual verification FAILED: 16 points and four loading arrows
render, but variable labels overlap and clip at 1024px dark mode. Evidence:
`/tmp/lab2_t10_acp_overlay_1440_light.png` and
`/tmp/lab2_t10_acp_overlay_1024_dark.png`; parent inspected the latter.
The user's overlay request was verification-only; a bounded label-space and
collision-layout correction awaits authorization. T8-T10 form one coherent
navigation/shared-configuration compatibility work unit; local commit and
native candidate assessment completed after parent readback. Work-unit commit:
`a154181` (507 authored lines), cumulative 2,589. Parent spot check: 29 tests
passed in 13.40s. Native assessed pending range against `44b4989` as medium,
649 lines; user granted this candidate. Reliability review reported no
findings, and exact acknowledgement burned authority for
`review-3f64b13d32748440`. Latest reviewed boundary: `a154181`.
T12 summary consolidation and regression checks are complete but uncommitted;
browser verification confirmed one Data summary and retained EDA summary.
T11 final permitted confirmation passed the real four-variable example at
1440px light and 1024px dark: 16 observations, four unchanged arrow endpoints,
separated labels, no clipping, actual axis ranges [-1.9, 1.9]. The 20-variable
long-label stress case remains visually partial: variable labels do not overlap
each other, but one left label intersects the CP2 axis title. Evidence:
`/tmp/lab2_t11_crowded_overlay_564_dark.png`; parent inspected this capture.
No further polishing loop was run. Full implementation tests passed 91 tests
and 52 subtests; independent final graph/UI checks passed 4 tests and diff-check.
The unavailable prior server was confirmed absent, then restarted only on
localhost:8502 (owned session `85269`). Commit/review remain pending while
T11's crowded-axis-title defect is disclosed and unresolved.
Untracked `.aws` was explicitly excluded and left untouched. No main or remote
action. Engram mirror remains pending authoritative runtime registration.
T7 requires the professor's datasets,
provenance and report template. Do not invent academic results. Update the
pending Engram mirror only after host session registration.

## T13 — Compare saved executions and retain Random Forest criteria

User-authorized follow-up: replace the configuration benchmark destination with
a read-only comparison of successful executions already stored in Results.
Add an explicit Random Forest criterion choice (`gini` or `entropy`) and retain
the latest successful result independently for each criterion. Retraining one
criterion replaces only its own saved result; failed training preserves the
last successful result and exposes the attempt failure. Other models retain
their latest successful result. Parameter-widget edits alone do not invalidate
or retrain saved executions; shared dataset/target/partition context changes
still invalidate incomparable snapshots. Only trained models appear, with
saved parameters and partition provenance. Comparison/navigation must never
call training or experimentation. Do not implement proposed `general %`, `y %`,
or `n %` metrics until metric semantics are clarified.
Route: delegated direct; existing shared renderer/state/UI/tests span multiple
non-trivial files. TDD off; ordinary functional checks. Checks: focused AppTest
regressions, full pytest suite with configured thread limits, F821, affected
format check, and `git diff --check`. Commit and review are parent-owned.
Outcome: the comparison route now reads the same individual-result store as the
Results route and has no training or experiment action. Untrained algorithms
remain absent. Random Forest exposes `gini`/`entropy` independently, retains one
successful snapshot per criterion, and failed attempts retain the previous
success with an error. Editing per-model controls no longer clears saved
execution; the active shared classification context still removes all model
snapshots. Applying a fresh global partition now clears all individual result
keys (the previous cleanup only removed the exact base key, not suffixed keys).
Checks: focused command — 54 passed, 22 subtests passed in 17.64s; full
thread-limited suite — 87 passed, 52 subtests passed in 31.91s. F821 and
affected Ruff format checks passed; `git diff --check` passed. No runtime server
or browser was started. Do not implement `general %`, `y %`, or `n %` metrics
without the user's semantic clarification. Rollback: restore the comparison
benchmark route/renderer, per-model invalidation behavior, and RF result store;
restore the prior tests and task-document entry. All changes remain uncommitted
for parent review; no commit identity is claimed.
Test count context: T13 replaced tests that exercised the retired configuration-
benchmark UI with read-only result-route, saved-execution, RF-criterion, and
invalidation coverage. The suite changed from the prior 91 tests to 87; no
applicable behavior coverage was intentionally dropped. T14 adds two metric
regressions, bringing the current suite to 89 tests.

## T14 — Show accuracy and actual-class recall percentages

User clarification: `general %` means overall accuracy; `y %` and `n %` mean
recall for actual `y` and actual `n` respectively (correct predictions within
each actual class). Expose general accuracy and per-actual-class recall as
percentages in saved individual and comparison results. When actual class labels
are y/n, identify the matching recalls; otherwise use each actual class's
literal label, including numeric `1`/`0` without inferring y/n semantics. A class
with no evaluated actual cases must be shown as unavailable, not zero. Derive
metrics from the saved execution without retraining. Preserve the read-only
comparison and RF criterion snapshots. Route: delegated direct, multiple
render/test files; TDD off. Checks: asymmetric confusion-matrix semantics,
binary and multiclass labels, absent evaluated class, percentage rendering,
no training/experiment during comparison, focused/full tests, F821,
changed-range format check, and `git diff --check`. Parent owns commit/review.
Outcome: the saved-result summary now shows overall accuracy as a percentage and
recall for each actual class as a percentage, derived from the saved actual and
predicted labels. Case-insensitive binary string labels `y`/`n` receive Y/N
names; all other labels (including numeric `1`/`0`) retain their literal class
identity. A class absent from evaluated truth is labeled “Sin casos evaluados”.
The individual detail view and read-only comparison expose the same metrics;
no training or experimentation is performed to derive them. T13's unrelated
formatting expansion in `streamlit_app.py` was removed while preserving T12's
single Data summary, T13's comparison routing and partition invalidation, and
T14's percentage metrics. Checks: focused command — 56 passed, 22 subtests
passed; full thread-limited suite — 89 passed, 52 subtests passed. F821,
changed Python-file formatting checks, and `git diff --check` passed.
`streamlit_app.py` remains at its pre-existing nonformatted baseline; it was not
whole-file formatted. No server/browser or commit/review was run.

Independent verification evidence: the parent verifier passed the final 89 tests
and 52 subtests and confirmed the RF snapshots in browser captures
`/tmp/lab2_t14_rf_compare.png` and `/tmp/lab2_t14_rf_results.png`. Documentation
reconciliation updates README, the LAB02 guide, DESIGN, and the Superdesign UI
map to distinguish read-only saved-result comparison from the offline
`Clasificacion.experimentar(...)` API.
