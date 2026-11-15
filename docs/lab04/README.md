# Lab 04

## Coordinated agent workflow

From the repository root, use the project environment to run the coordinator:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_agents data/diabetes_lab03.csv --agent coordinated --target target --provider rules --language en
.\.venv\Scripts\python.exe -m scripts.run_agents data/breast_cancer.csv --agent coordinated --target target --provider rules --language en
```

Both runs first show `Assigned EDA`. The coordinator then assigns one specialist
and reports its completion. Expect `regression` for Diabetes and `classification`
for Breast Cancer. With the reference data and default settings, the recommended
models are Lasso and logistic regression (`RL`), respectively.

Use `--task clustering` to request clustering. If you pass `--target`, the
coordinator excludes that column from the clustering inputs. In the default
`auto` mode, omitting `--target` runs only EDA.

### Predictor screening and run trace

For supervised requests, the coordinator excludes predictors flagged by EDA
as constant, identifier-like or potential target leakage. It also excludes exact
copies of the target. If no predictors remain, the command exits with code `2`
and reports that the input needs review.

Pass `--trace` with a new JSON file path to record decisions, lifecycle events,
task durations and the final result. The parent directory must exist. A failed
run still writes the trace with status `failed`, or `needs_review` when no
predictors remain. Traces contain column names and summary metrics.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_agents data/diabetes_lab03.csv --agent coordinated --target target --provider rules --language en --trace "$env:TEMP\lab04-diabetes-trace.json"
```

## Reproduce the experiments

Run the reference suite from the repository root:

```powershell
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
python -m scripts.run_lab04 --output output/lab04/reference
```

The suite evaluates Diabetes regression, Breast Cancer classification and a
synthetic leakage case derived from Diabetes. That case runs with other eligible
predictors and with only a copy of the target. An additional clustering run
excludes the target from its inputs.

The direct baseline calls EDA and the predetermined specialist in sequence.
Both routes use the same screening policy, model catalog, seed 42, 25% test split
and five validation folds for these datasets. The comparison checks the selected
features, train/test split hash, winner, reported metrics and candidate failures.
Screening uses the existing EDA profile over the supplied dataset before training.

`summary.csv` records duration, dispatched events, completed tasks, errors and
quality metrics per run. `runs.json` includes the corresponding decisions, traces,
dataset hashes and environment metadata. A direct run dispatches zero bus events.
The runner alternates route order across repetitions and times processing after
CSV loading. Times vary with hardware and execution order. The output directory
must be new. Use `--scenario` to select one or more cases and `--repeats` to change
the default of three paired repetitions. A mismatch or unexpected failure exits
with code `1` after writing evidence.

## Academic report

The Spanish academic report is in `docs/lab04/main.tex`. The current draft
compiles without external files or BibTeX.

### Build the report

In Overleaf, upload `main.tex` and compile with `pdfLaTeX`.

To build locally with TeX Live or MiKTeX, run:

```powershell
cd docs/lab04
pdflatex main.tex
pdflatex main.tex
```
