"""
Connectors over files on the server: a folder of Parquet tables, a folder of
CSV/TSV files, a DuckDB database file and a SQLite database file.
"""

from pathlib import Path

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


class CSVDirConnector(DatasetConnector):
    """Represents a folder of CSV or TSV files, one table per file named by its stem."""

    type = "csv_dir"
    label = "CSV folder"
    ConfigModel = _CSVConfig

    def files(self) -> list[Path]:
        """Returns the folder's files matching the pattern."""
        root = Path(self.cfg.path)
        if not root.is_dir():
            raise DatasetUnavailable(f"Folder not found: {root}")
        return sorted(p for p in root.glob(self.cfg.file_pattern) if p.is_file())

    def relations(self, conn: duckdb.DuckDBPyConnection, alias: str) -> dict[str, str]:
        relations = {}
        for file in self.files():
            delim = ", delim='\\t'" if file.suffix.lower() == ".tsv" else ""
            relations[file.stem] = (
                f"read_csv({sql_literal(file.as_posix())}, header=true, ignore_errors=true, "
                f"null_padding=true, sample_size=100000{delim})"
            )
        return relations

    def allowed_paths(self) -> tuple[list[str], list[str]]:
        return [Path(self.cfg.path).resolve().as_posix() + "/"], []

    def fingerprint(self) -> str:
        return "csv:" + digest(_stat_parts(self.files()))


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
