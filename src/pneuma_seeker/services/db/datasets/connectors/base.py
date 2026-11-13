"""
A connector exposes one dataset's tables to DuckDB. Whatever the source, it is
attached under the dataset's alias as one view per table, named and with columns
cleaned like ingested tables, so agents and retrievers query every source as
<alias>."<table>".
"""

import hashlib
from abc import ABC, abstractmethod
from typing import ClassVar

import duckdb
from pydantic import BaseModel

from pneuma_seeker.shared.str_processor import clean_column_table_name


class DatasetUnavailable(Exception):
    """Raised when a dataset's source can't be read (missing files, unreachable server)."""


def quote_ident(name: str) -> str:
    """Quotes a SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def sql_literal(value: str) -> str:
    """Quotes a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def source_alias(alias: str) -> str:
    """Returns the alias under which a database source is attached behind the dataset's views."""
    return f"{alias}__source"


def dedupe(names: list[str]) -> list[str]:
    """Deduplicates names by appending _1, _2, etc. to repeats."""
    seen: dict[str, int] = {}
    result = []
    for name in names:
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            result.append(name)
    return result


def clean_names(names: list[str]) -> list[str]:
    """Cleans table or column names like ingestion does, keeping them distinct and non-empty."""
    return dedupe([clean_column_table_name(n) or "unnamed" for n in names])


def digest(parts: list[str]) -> str:
    """Returns a short stable hash of `parts`."""
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


class DatasetConnector(ABC):
    """
    Represents a dataset source. Subclasses set `type`, `ConfigModel` (the
    validated config) and `SECRET_FIELDS` (config fields never shown back), and
    implement `relations` and `fingerprint`.
    """

    type: ClassVar[str]
    label: ClassVar[str]
    ConfigModel: ClassVar[type[BaseModel]]
    SECRET_FIELDS: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: dict) -> None:
        self.cfg = self.ConfigModel(**config)

    @abstractmethod
    def relations(self, conn: duckdb.DuckDBPyConnection, alias: str) -> dict[str, str]:
        """
        Returns, per source table name, the SQL relation reading it. May attach the
        source under source_alias(alias) first. Raises DatasetUnavailable.
        """

    @abstractmethod
    def fingerprint(self) -> str:
        """Returns a value that changes when the dataset's data may have."""

    def attach(self, conn: duckdb.DuckDBPyConnection, alias: str, tables: set[str] | None = None) -> None:
        """
        Attaches the dataset under `alias` as one view per table, or only for
        `tables` (cleaned names; creating a view reads its source's schema, which
        adds up for thousands of tables). Raises DatasetUnavailable.
        """
        conn.execute(f"ATTACH ':memory:' AS {quote_ident(alias)}")
        try:
            relations = self.relations(conn, alias)
            for table, relation in zip(clean_names(list(relations)), relations.values()):
                if tables is not None and table not in tables:
                    continue
                conn.execute(
                    f"CREATE VIEW {quote_ident(alias)}.main.{quote_ident(table)} AS "
                    f"SELECT {self._select_list(conn, relation)} FROM {relation}"
                )
        except Exception as exc:
            self.detach(conn, alias)
            if isinstance(exc, DatasetUnavailable):
                raise
            raise DatasetUnavailable(f"Couldn't read dataset '{alias}': {self.scrub(str(exc))}") from exc

    def detach(self, conn: duckdb.DuckDBPyConnection, alias: str) -> None:
        """Detaches the dataset and its source, if attached."""
        attached = {row[0] for row in conn.execute("SELECT database_name FROM duckdb_databases()").fetchall()}
        for name in (alias, source_alias(alias)):
            if name in attached:
                conn.execute(f"DETACH DATABASE {quote_ident(name)}")

    def allowed_paths(self) -> tuple[list[str], list[str]]:
        """Returns the (directories, files) the dataset's views read; none for a remote source."""
        return [], []

    def lockdown(self, conn: duckdb.DuckDBPyConnection) -> None:
        """
        Restricts `conn` to the files the dataset's views read, so SQL run on it
        (e.g. read_csv('/etc/passwd')) can't reach other files, and locks the setting.
        """
        directories, files = self.allowed_paths()
        conn.execute(f"SET allowed_directories=[{', '.join(sql_literal(d) for d in directories)}]")
        conn.execute(f"SET allowed_paths=[{', '.join(sql_literal(f) for f in files)}]")
        conn.execute("SET enable_external_access=false")
        conn.execute("SET lock_configuration=true")

    def table_names(self) -> list[str]:
        """Returns the dataset's (cleaned) table names. Raises DatasetUnavailable."""
        conn = duckdb.connect()
        try:
            return sorted(clean_names(list(self.relations(conn, "dataset"))))
        except DatasetUnavailable:
            raise
        except Exception as exc:
            raise DatasetUnavailable(f"Couldn't read the dataset: {self.scrub(str(exc))}") from exc
        finally:
            conn.close()

    def health_check(self) -> tuple[bool, str | None, int]:
        """Returns (reachable, error, table count) without raising."""
        try:
            return True, None, len(self.table_names())
        except Exception as exc:
            return False, self.scrub(str(exc)), 0

    def redacted_config(self) -> dict:
        """Returns the config with secret fields masked."""
        config = self.cfg.model_dump(by_alias=True)
        for field in self.SECRET_FIELDS:
            if config.get(field):
                config[field] = "••••••••"
        return config

    def stored_config(self) -> dict:
        """Returns the config as stored in the registry."""
        return self.cfg.model_dump(by_alias=True)

    def scrub(self, text: str) -> str:
        """Removes secret values from `text` (e.g. an error echoing a connection string)."""
        for field in self.SECRET_FIELDS:
            secret = getattr(self.cfg, field, None)
            if secret:
                text = text.replace(str(secret), "••••••••")
        return text

    def _select_list(self, conn: duckdb.DuckDBPyConnection, relation: str) -> str:
        columns = [row[0] for row in conn.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]
        return ", ".join(
            f"{quote_ident(original)} AS {quote_ident(clean)}"
            for original, clean in zip(columns, clean_names(columns))
        )
