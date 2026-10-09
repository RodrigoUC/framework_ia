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
  remains separate. Last reviewed boundary: `d3fe83a` (T2 reliability review
  acknowledged; authority burned).
- Authored running count: 538 (T2 commit `d3fe83a`). Commit/slice evidence:
  T2 `d3fe83a`; T3 remains uncommitted pending parent spot check and review.

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
  map updates, and their navigation tests. Parent to record T3 authored lines,
  commit identity, and next RDD boundary after spot check/review.
- [ ] T4 — Give every available classifier its own action/view and add the
  source-target balance chart before training.
  Route: delegated; shared model UI and EDA visualization.
  Checks: all classifier routes, target counts/proportions/missing values,
  rare classes, no target leakage, no implicit training.
- [ ] T5 — Unify comparison and results UX and repair confirmed execution/state
  defects. Remove LAB02 and Modelo individual from user-facing copy.
  Route: delegated; experiment rendering, state, and navigation.
  Checks: visible per-candidate errors, validation/test distinction, provenance,
  stale-state invalidation, navigation preservation, read-only results.
- [ ] T6 — Run full regression and bounded visual/accessibility verification;
  reconcile README and canonical UI maps with observed behavior.
  Route: delegated verification; parent structural readback and spot check.
  Checks: python -m pytest -q, changed-UI detector, light/dark and viewport checks
  when a browser/server is authorized and available. Record unavailable checks.
- [ ] T7 — Produce real laboratory evidence and report material after verifying
  the required professor datasets, target columns, provenance, and report template.
  Route: delegated when prerequisites are provided.
  Status: prerequisite pending; only data/ejemplo_analisis.csv is present.
  Checks: five required model families, standard/variant validation comparison,
  final test metrics, reproducibility exports, evidence-based report.

## Progress and next step

Read-only audit and baseline completed. The selected Streamlit AppTest/focused
tests and full pytest suite pass, but no per-model failure has been reproduced
from the existing tests; T1 therefore remains open pending the concrete user
error scenario/traceback. Next: use the authorized AppTest or minimal command
for each reported scenario and record exact behavior before proposing a source
fix. Preserve each observed check and commit identity here; update the pending
Engram mirror only after host session registration.
