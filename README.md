# Nexo IA

Nexo IA is the EIF420O final project for exploratory data analysis, data preparation, clustering, classification and regression. The Streamlit dashboard delegates analysis to a local multi-agent system. The rules mode works without an API key.

## Run the dashboard

From the extracted project directory, use Python 3.11. On Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-project-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS or Linux, use `python3.11 -m venv .venv` and `.venv/bin/python`. The lock records the Windows/Python 3.11 reference environment. Use `requirements.txt` when installing on a different supported platform.

## Demonstrate the system

1. Open **Data source**, select a local CSV and click **Apply source**. Set the dashboard language in the sidebar.
2. Open **Data & Preparation** or **Exploratory data analysis** to review the data. Applying a preparation sends an explicit request to the EDA agent.
3. For classification, load `breast_cancer.csv`. In **Configuration**, select Classification and set the target to `target`. Run **Comparative Model Evaluation** from the Classification section.
4. For regression, load `diabetes_lab03.csv`. Select Regression and set the target to `target`. Run the regression benchmark.
5. In **Multi-agent system**, choose Clustering, enter an agent name, register it and select it under **Agents selected for this session**. Open **Coordinated analysis**, select Clustering and set the target to `target` so that it is excluded from the inputs. Run the analysis.
6. Open **Experiment history**. Verify the run, assignee, completion event and result. The local SQLite workspace is saved under `output/project/`. Export JSON traces, CSV results and fitted model artifacts where available. Registered agents and history persist across app restarts. Agent selections apply to the current session.

## Reproduce the reference evidence

```powershell
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:NUMBA_NUM_THREADS = '2'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.verify_project --output output/project/reference
```

The verifier drives the Streamlit entry point against an isolated SQLite workspace and checks registration, dispatch, completion, target exclusion and reopening the stored result. It records reference package versions and input SHA-256 hashes. Use a new directory for each run. Durations and run identifiers vary. The delivery ZIP includes the evaluated traces and comparison tables under `evidence/`.

| Included dataset | Source and use |
| --- | --- |
| `data/breast_cancer.csv` | scikit-learn Breast Cancer Wisconsin dataset, 569 rows, classification target `target`. |
| `data/diabetes_lab03.csv` | scikit-learn Diabetes dataset, 442 rows, continuous response `target`. It is used for regression and for clustering with the response excluded. |
| `data/ejemplo_analisis.csv` | Small CSV example included with the course framework. |

## Structure

| Path | Responsibility |
| --- | --- |
| `app.py`, `ai_framework/ui/` | Streamlit entry point, views, controllers, session state and translations. |
| `ai_framework/agents/` | Agent definitions, registry, factory, coordinator, event bus and shared analysis state. |
| `ai_framework/data/`, `ai_framework/models/` | CSV loading, EDA, preprocessing and the model implementations from the labs. |
| `ai_framework/persistence/` | Local storage for registered agents, experiment history and trained models. |
| `scripts/`, `tests/` | Reproducible execution and verification. |
| `docs/project/` | Spanish technical report and diagram source. |

Build the report from `docs/project/` with two passes of `pdflatex main.tex`, or upload the source files in that directory to Overleaf. The compiled report is included at `report/project.pdf` in the delivery ZIP.

The prior labs remain reproducible through their English commands and compatibility aliases. See the [Lab 04 guide](docs/lab04/README.md) for that lab's original protocol.
