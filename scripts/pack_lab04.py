"""Package the Lab04 source, report and experiment evidence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
PROJECT_FILES = (
    "README.md", "app.py", "pyproject.toml", "requirements.txt", "scripts/__init__.py",
    "scripts/run_agents.py", "scripts/ejecutar_agentes.py", "scripts/run_lab04.py",
    "scripts/pack_lab04.py", "tests/test_coordinator_agent.py", ".streamlit/config.toml",
    ".streamlit/secrets.toml.example", "data/diabetes_lab03.csv", "data/breast_cancer.csv",
    "docs/lab04/README.md", "docs/lab04/main.tex",
    "ai_framework/ui/views/coordinated_analysis_view.py",
    "tests/test_ui_coordinated_analysis.py", "tests/streamlit_app_entry.py",
)


def build_package(destination: Path, pdf: Path, evidence: Path, llm_evidence: Path | None) -> None:
    trace = evidence / "runs.json"
    results = json.loads(trace.read_text(encoding="utf-8"))
    if not results["runs"] or not all(run["matches_expectation"] for run in results["runs"]):
        raise ValueError("The experimental evidence must contain successful reference checks.")
    with pdf.open("rb") as report:
        if report.read(5) != b"%PDF-":
            raise ValueError("The report must be a compiled PDF.")

    tracked_python = subprocess.check_output(
        ["git", "ls-files", "--", "ai_framework"], cwd=ROOT, text=True,
    ).splitlines()
    project_files = set(PROJECT_FILES) | {
        name for name in tracked_python if name.startswith("ai_framework/") and name.endswith(".py")
    }
    files = {name: ROOT / name for name in project_files}
    files.update({"report/lab04.pdf": pdf, "evidence/runs.json": trace, "evidence/summary.csv": evidence / "summary.csv"})
    if llm_evidence is not None:
        files["evidence/groq_smoke_test.md"] = llm_evidence
    for path in files.values():
        if not path.is_file():
            raise ValueError(f"Missing delivery file: {path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as package:
        for name, path in sorted(files.items()):
            package.write(path, name)
    with ZipFile(destination) as package:
        if set(package.namelist()) != files.keys() or package.testzip() is not None:
            raise ValueError("Delivery ZIP failed verification.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", required=True, nargs="+", help="Student IDs in submission order.")
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--llm-evidence", type=Path, help="Observed Groq smoke-test record in English.")
    parser.add_argument("--output", required=True, type=Path, help="Destination directory.")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[0-9]+", identifier) for identifier in args.ids):
        parser.error("Student IDs must contain digits only.")
    destination = args.output / ("_".join(args.ids) + ".zip")
    try:
        build_package(destination, args.pdf, args.evidence, args.llm_evidence)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Could not package Lab04: {error}\n")
    print(f"Verified delivery ZIP: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
