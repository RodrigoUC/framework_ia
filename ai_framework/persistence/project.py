"""SQLite storage for agent definitions and experiment evidence."""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sqlite3
from uuid import uuid4

from ai_framework.agents.registry import AgentDefinition, AgentRegistry


class ProjectStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY, name_key TEXT NOT NULL UNIQUE, definition TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source TEXT NOT NULL,
                    operation TEXT NOT NULL, status TEXT NOT NULL, trace TEXT NOT NULL, csv TEXT
                );
                CREATE TABLE IF NOT EXISTS models (
                    run_id TEXT PRIMARY KEY, artifact BLOB NOT NULL
                );
            """)

    @contextmanager
    def connection(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.row_factory = sqlite3.Row
            yield db

    def registry(self) -> AgentRegistry:
        registry = AgentRegistry.with_defaults()
        with self.connection() as db:
            definitions = db.execute("SELECT definition FROM agents ORDER BY rowid").fetchall()
        for row in definitions:
            values = json.loads(row["definition"])
            values["capabilities"] = frozenset(values["capabilities"])
            registry.register(AgentDefinition(**values))
        return registry

    def register(self, name: str, specialty: str, capabilities: frozenset[str]) -> AgentDefinition:
        definition = AgentDefinition(uuid4().hex, name.strip(), specialty, capabilities)
        self.registry().register(definition)
        payload = {
            "agent_id": definition.agent_id, "name": definition.name, "specialty": specialty,
            "capabilities": sorted(capabilities),
        }
        try:
            with self.connection() as db:
                db.execute("INSERT INTO agents VALUES (?, ?, ?)", (
                    definition.agent_id, definition.name.casefold(), json.dumps(payload, ensure_ascii=False),
                ))
        except sqlite3.IntegrityError as error:
            raise ValueError("The agent name is already registered.") from error
        return definition

    def record(self, trace: dict, source: str, operation: str, csv: str | None = None, model=None) -> dict:
        evidence = {**trace, "run_id": uuid4().hex}
        with self.connection() as db:
            db.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)", (
                evidence["run_id"], datetime.now(timezone.utc).isoformat(), source,
                operation, evidence["status"], json.dumps(evidence, ensure_ascii=False, default=str), csv,
            ))
            if model is not None:
                db.execute("INSERT INTO models VALUES (?, ?)", (
                    evidence["run_id"], pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL),
                ))
        return evidence

    def list_runs(self) -> list[dict]:
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT id, created_at, source, operation, status FROM runs ORDER BY created_at DESC"
            )]

    def get_model(self, run_id: str) -> bytes | None:
        with self.connection() as db:
            row = db.execute("SELECT artifact FROM models WHERE run_id = ?", (run_id,)).fetchone()
        return row["artifact"] if row is not None else None

    def get_run(self, run_id: str) -> dict | None:
        with self.connection() as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row is not None else None
