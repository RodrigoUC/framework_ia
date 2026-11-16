"""Build the EIF420O delivery ZIP from project sources and verified evidence."""

import argparse
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
NAME = "504610652_402680547_118910482_119180324.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads((args.evidence / "summary.json").read_text(encoding="utf-8"))
    if not summary["checks"]["history_reopened"] or not summary["checks"]["downloadable_model"]:
        parser.error("The project verification is incomplete.")
    if args.pdf.read_bytes()[:5] != b"%PDF-":
        parser.error("Provide a compiled PDF.")
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
         "--", "ai_framework", "data", "docs", "scripts", "tests", "app.py", "README.md",
         "COMO_EJECUTAR.md", "pyproject.toml", "requirements.txt", "requirements-project-lock.txt",
         ".streamlit/config.toml", ".streamlit/secrets.toml.example"],
        cwd=ROOT,
    ).decode().split("\0")
    files = {name: ROOT / name for name in paths if name}
    files["report/project.pdf"] = args.pdf
    files.update({f"evidence/{path.name}": path for path in args.evidence.iterdir()
                  if path.is_file() and path.suffix in {".csv", ".json"}})
    for name in ("preparation", "eda", "classification", "regression", "clustering"):
        trace = json.loads(files[f"evidence/{name}.json"].read_text(encoding="utf-8"))
        if trace["status"] != "completed":
            parser.error(f"Incomplete reference run: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / NAME
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
    with ZipFile(destination) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != files.keys():
            raise RuntimeError("Delivery ZIP verification failed.")
    print(destination)


if __name__ == "__main__":
    main()
