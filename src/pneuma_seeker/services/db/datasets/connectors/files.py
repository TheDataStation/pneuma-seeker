"""
Connectors over files on the server: a folder of Parquet tables, a folder of
CSV/TSV files, a DuckDB database file and a SQLite database file.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import duckdb
from pydantic import BaseModel, field_validator

from pneuma_seeker.services.db.datasets.connectors.base import (
    DatasetConnector,
    DatasetUnavailable,
    digest,
    quote_ident,
    source_alias,
    sql_literal,
)


def _stat_parts(files: list[Path]) -> list[str]:
    parts = []
    for file in files:
        stat = file.stat()
        parts.append(f"{file.as_posix()}:{stat.st_size}:{stat.st_mtime_ns}")
    return parts


def _relative_pattern(value: str) -> str:
    if value.startswith("/") or ".." in Path(value).parts:
        raise ValueError("file_pattern must be relative to the folder and can't contain '..'")
    return value


class _PathConfig(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def _required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path is required")
        return value.strip()


class _FolderConfig(_PathConfig):
    file_pattern: str = "*.parquet"

    @field_validator("file_pattern")
    @classmethod
    def _pattern(cls, value: str) -> str:
        return _relative_pattern(value)


class ParquetDirConnector(DatasetConnector):
    """
    Represents a folder with one subfolder per table holding its Parquet file(s);
    a table with several files is read as their union. Uploads are stored this way.
    """

    type = "parquet_dir"
    label = "Parquet folder"
    ConfigModel = _FolderConfig

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.root = Path(self.cfg.path)

    def tables(self) -> dict[str, list[Path]]:
        """Returns, per table folder, its Parquet files."""
        if not self.root.is_dir():
            raise DatasetUnavailable(f"Folder not found: {self.root}")
        found = {}
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            files = sorted(folder.glob(self.cfg.file_pattern))
            if files:
                found[folder.name] = files
        return found

    def relations(self, conn: duckdb.DuckDBPyConnection, alias: str) -> dict[str, str]:
        return {
            name: f"read_parquet({sql_literal((self.root / name / self.cfg.file_pattern).as_posix())}, "
            "union_by_name=true, binary_as_string=true)"
            for name in self.tables()
        }

    def allowed_paths(self) -> tuple[list[str], list[str]]:
        return [self.root.resolve().as_posix() + "/"], []

    def fingerprint(self) -> str:
        return "parquet:" + digest(_stat_parts([f for files in self.tables().values() for f in files]))


class _CSVConfig(_PathConfig):
    file_pattern: str = "*.csv"

    @field_validator("file_pattern")
    @classmethod
    def _pattern(cls, value: str) -> str:
        return _relative_pattern(value)


# As uploads read CSVs; a serial read also handles quoted newlines alongside null padding.
CSV_OPTIONS = "header=true, ignore_errors=true, strict_mode=false, null_padding=true, sample_size=100000, parallel=false"
# What a read with a known layout (auto_detect=false) keeps of CSV_OPTIONS.
_KNOWN_LAYOUT_OPTIONS = "ignore_errors=true, strict_mode=false, null_padding=true, parallel=false"


def csv_layout_root() -> Path:
    """Returns the folder of the stored CSV layouts (see CSVDirConnector.build_layouts)."""
    from pneuma_seeker.shared.config import Config  # deferred: connectors load without the app config

    return Path(Config().RETRIEVAL_INDEX_ROOT) / "csv_layouts"


class CSVDirConnector(DatasetConnector):
    """
    Represents a folder of CSV or TSV files, one table per file named by its stem.

    Reading a CSV first detects its layout (delimiter, quoting, header, column types),
    which for thousands of files makes attaching slow. build_layouts stores each
    file's detected layout once, keyed by its size and modification time, after
    checking that reading with it gives the same columns and types; files with a
    stored layout are then read with it (auto_detect=false), the others as before.
    """

    type = "csv_dir"
    label = "CSV folder"
    ConfigModel = _CSVConfig

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._known: dict[str, list[str]] = {}  # relation -> its columns, from stored layouts

    def _layout_path(self) -> Path:
        folder = Path(self.cfg.path).resolve().as_posix() + "|" + self.cfg.file_pattern
        return csv_layout_root() / f"{hashlib.sha256(folder.encode()).hexdigest()[:24]}.json"

    def _load_layouts(self) -> dict[str, dict]:
        try:
            return json.loads(self._layout_path().read_text())
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _stat(file: Path) -> str:
        stat = file.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    @staticmethod
    def _detect_options(file: Path) -> str:
        """Returns the options detecting a file's layout: CSV_OPTIONS, plus a tab delimiter for TSV."""
        return CSV_OPTIONS + (", delim='\\t'" if file.suffix.lower() == ".tsv" else "")

    def _detected_relation(self, file: Path) -> str:
        return f"read_csv({sql_literal(file.as_posix())}, {self._detect_options(file)})"

    def build_layouts(self, progress: Callable[[int, int, str], None] | None = None) -> int:
        """
        Stores the layout of each file that has none stored or changed since; returns
        how many it stored. `progress(done, total, file)` is called before each file.
        """
        layouts, files = self._load_layouts(), self.files()
        current = {f.name for f in files}
        stored = {name: v for name, v in layouts.items() if name in current}
        added = 0
        conn = duckdb.connect()
        try:
            for done, file in enumerate(files):
                if progress:
                    progress(done, len(files), file.name)
                if stored.get(file.name, {}).get("stat") == self._stat(file):
                    continue
                stored.pop(file.name, None)
                layout = self._sniff(conn, file)
                if layout is not None:
                    stored[file.name] = layout
                    added += 1
        finally:
            conn.close()
        path = self._layout_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(".json.tmp")
        staging.write_text(json.dumps(stored))
        os.replace(staging, path)
        return added

    def _sniff(self, conn: duckdb.DuckDBPyConnection, file: Path) -> dict | None:
        """Returns the file's layout, or None if reading with it would not match detection."""
        stat = self._stat(file)
        try:
            prompt = conn.execute(
                f"SELECT Prompt FROM sniff_csv({sql_literal(file.as_posix())}, {self._detect_options(file)})"
            ).fetchone()[0]
            body = prompt.strip().removeprefix("FROM ").rstrip(";").rstrip()
            if not body.startswith("read_csv(") or not body.endswith(")"):
                return None
            relation = f"{body[:-1]}, {_KNOWN_LAYOUT_OPTIONS})"
            detected = conn.execute(f"DESCRIBE SELECT * FROM {self._detected_relation(file)}").fetchall()
            known = conn.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        except duckdb.Error:
            return None
        if [(r[0], r[1]) for r in known] != [(r[0], r[1]) for r in detected]:
            return None
        return {"stat": stat, "relation": relation, "columns": [r[0] for r in known]}

    def known_columns(self, relation: str) -> list[str] | None:
        return self._known.get(relation)

    def files(self) -> list[Path]:
        """Returns the folder's files matching the pattern."""
        root = Path(self.cfg.path)
        if not root.is_dir():
            raise DatasetUnavailable(f"Folder not found: {root}")
        return sorted(p for p in root.glob(self.cfg.file_pattern) if p.is_file())

    def relations(self, conn: duckdb.DuckDBPyConnection, alias: str) -> dict[str, str]:
        layouts, relations = self._load_layouts(), {}
        for file in self.files():
            layout = layouts.get(file.name)
            if layout is not None and layout.get("stat") == self._stat(file):
                relations[file.stem] = layout["relation"]
                self._known[layout["relation"]] = layout["columns"]
            else:
                relations[file.stem] = self._detected_relation(file)
        return relations

    def allowed_paths(self) -> tuple[list[str], list[str]]:
        return [Path(self.cfg.path).resolve().as_posix() + "/"], []

    def fingerprint(self) -> str:
        return "csv:" + digest([CSV_OPTIONS, *_stat_parts(self.files())])


class _DatabaseFileConnector(DatasetConnector):
    ConfigModel = _PathConfig
    attach_options: str = "READ_ONLY"
    extension: str | None = None

    def relations(self, conn: duckdb.DuckDBPyConnection, alias: str) -> dict[str, str]:
        path = Path(self.cfg.path)
        if not path.is_file():
            raise DatasetUnavailable(f"File not found: {path}")
        if self.extension:
            conn.execute(f"INSTALL {self.extension}; LOAD {self.extension}")
        source = source_alias(alias)
        conn.execute(f"ATTACH {sql_literal(path.as_posix())} AS {quote_ident(source)} ({self.attach_options})")
        rows = conn.execute(
            "SELECT schema_name, table_name FROM duckdb_tables() WHERE database_name = ? "
            "UNION ALL SELECT schema_name, view_name FROM duckdb_views() WHERE database_name = ? AND NOT internal "
            "ORDER BY 1, 2",
            [source, source],
        ).fetchall()
        return {table: f"{quote_ident(source)}.{quote_ident(schema)}.{quote_ident(table)}" for schema, table in rows}

    def allowed_paths(self) -> tuple[list[str], list[str]]:
        path = Path(self.cfg.path).resolve().as_posix()
        return [], [path, path + ".wal"]

    def fingerprint(self) -> str:
        path = Path(self.cfg.path)
        if not path.is_file():
            raise DatasetUnavailable(f"File not found: {path}")
        return f"{self.type}:" + digest(_stat_parts([path]))


class DuckDBFileConnector(_DatabaseFileConnector):
    """Represents a DuckDB database file, attached read-only."""

    type = "duckdb"
    label = "DuckDB file"


class SQLiteFileConnector(_DatabaseFileConnector):
    """Represents a SQLite database file, attached read-only."""

    type = "sqlite"
    label = "SQLite file"
    attach_options = "TYPE SQLITE, READ_ONLY"
    extension = "sqlite"
