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
- Do not launch a Streamlit server without explicit permission.
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
  keyboard behavior; no server/browser launch was authorized. T6 remains
  partial only for those browser-dependent checks. Parent focused tests and
  structural readback are separate completion checks, pending at this record.
- [ ] T7 — Produce real laboratory evidence and report material after verifying
  the required professor datasets, target columns, provenance, and report template.
  Route: delegated when prerequisites are provided.
  Status: prerequisite pending; only data/ejemplo_analisis.csv is present.
  Checks: five required model families, standard/variant validation comparison,
  final test metrics, reproducibility exports, evidence-based report.

## Progress and next step

T1 remains open pending the concrete user error scenario/traceback. T2–T5 are
committed on `lab_2`; all due native reviews were approved and acknowledged,
with T5 the latest reviewed boundary. T6's headless regression and projection
smoke passed, and documentation was reconciled. The parent removed the dead
`comparacion` branch, the independent `F821` recheck passed, and the full suite
passed again on the corrected bytes. Visual/accessibility browser checks remain
pending authorization. Next: parent reads back the T6 docs, finishes its
focused check, and owns the commit/review decision; obtain permission
before any server/browser visual pass. T7 requires the professor's datasets,
provenance and report template. Do not invent academic results. Update the
pending Engram mirror only after host session registration.
