"""
Reads the benchmarks in data_src/ (see data_src/README.md). Uses only the
standard library, so tools outside the backend can load it by path:

    <benchmark>/<benchmark>_tabular.json   KramaBench questions (id, query, answer)
    <benchmark>/benchmark.json             DataAgentBench questions (id, question, answer)
    <benchmark>/metadata.csv               descriptions: KramaBench (table_name, description),
                                           DataAgentBench (database, table_name, column_name, description)
    <benchmark>/databases.json             DataAgentBench: each database's type and file (and Postgres connection)
    <benchmark>/dataset/                   the tables (KramaBench CSVs) or database files (DataAgentBench)
"""

import csv
import json
from pathlib import Path

KRAMABENCH = ["archeology", "astronomy", "biomedical", "environment", "legal", "wildfire"]
DAB = ["bookreview", "crmarenapro", "deps_dev_v1", "github_repos", "googlelocal", "imdb",
       "music_brainz_20k", "pancancer_atlas", "patents", "stockindex", "stockmarket"]


def load_questions(root: Path, benchmark: str, ids: list[str] | None = None) -> list[dict]:
    """
    Returns a benchmark's questions as {question_id, question, ground_truth, dataset},
    or only those in `ids` (in that order). Raises ValueError for an unknown benchmark or id.
    """
    if benchmark in KRAMABENCH:
        raw = json.loads((root / benchmark / f"{benchmark}_tabular.json").read_text())
        questions = [{"question_id": q["id"], "question": q["query"], "ground_truth": q["answer"],
                      "dataset": benchmark} for q in raw]
    elif benchmark in DAB:
        raw = json.loads((root / benchmark / "benchmark.json").read_text())
        questions = [{"question_id": q["id"], "question": q["question"], "ground_truth": q["answer"],
                      "dataset": benchmark} for q in raw]
    else:
        raise ValueError(f"Unknown benchmark {benchmark!r}: one of {', '.join(KRAMABENCH + DAB)}.")
    if not ids:
        return questions
    by_id = {q["question_id"]: q for q in questions}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise ValueError(f"Unknown question id(s) in {benchmark}: {', '.join(unknown)}")
    return [by_id[i] for i in ids]


def databases(root: Path, benchmark: str) -> dict[str, dict]:
    """Returns a DataAgentBench benchmark's {database: {"type", "file", ...Postgres connection}}."""
    return json.loads((root / benchmark / "databases.json").read_text())


def dataset_name(root: Path, benchmark: str, database: str) -> str:
    """
    Returns the dataset a database is registered as: a KramaBench benchmark's
    name, or a DataAgentBench database's lowercased name, prefixed with its
    benchmark when another benchmark has a database of the same name.
    """
    if benchmark in KRAMABENCH:
        return benchmark
    name = database.lower()
    shared = sum(name in {d.lower() for d in databases(root, b)} for b in DAB)
    return f"{benchmark}_{name}" if shared > 1 else name


def sources(root: Path, postgres_host: str) -> dict[str, dict]:
    """
    Returns {dataset: {"benchmark", "database", "type", "config"}} for every
    benchmark database: each KramaBench benchmark as a CSV folder, each
    DataAgentBench database as its file or its Postgres database on `postgres_host`.
    """
    out = {}
    for benchmark in KRAMABENCH:
        out[benchmark] = {"benchmark": benchmark, "database": benchmark, "type": "csv_dir",
                          "config": {"path": str(root / benchmark / "dataset"), "file_pattern": "*.csv"}}
    for benchmark in DAB:
        for database, db in databases(root, benchmark).items():
            if db["type"] == "postgres":
                cfg = {"host": postgres_host, "port": db["port"], "dbname": db["dbname"],
                       "user": db["user"], "password": db["password"]}
            else:
                cfg = {"path": str(root / benchmark / "dataset" / db["file"])}
            out[dataset_name(root, benchmark, database)] = {
                "benchmark": benchmark, "database": database, "type": db["type"], "config": cfg}
    return out


def metadata_rows(root: Path, benchmark: str) -> list[dict]:
    """Returns a benchmark's metadata.csv rows (KramaBench rows get the benchmark as their database)."""
    path = root / benchmark / "metadata.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.setdefault("database", benchmark)
        row.setdefault("column_name", "")
    return rows


def stray_files(root: Path) -> list[Path]:
    """Returns the metadata.csv files inside a KramaBench dataset/ folder, which would be read as tables."""
    return [root / b / "dataset" / "metadata.csv" for b in KRAMABENCH
            if (root / b / "dataset" / "metadata.csv").exists()]
