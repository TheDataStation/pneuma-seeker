"""
Which datasets exist: each has a name, a connector type and the connector's
config, stored in Postgres (dataset_sources). A local dataset is a Parquet
folder under the datasets root; uploads and the migration register it (with who
created it), and a folder copied in without a row is listed all the same.
Without Postgres, those local folders are the only datasets.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from pneuma_seeker.shared.config import Config

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
NAME_RULE = "lowercase letters, digits and underscores, starting with a letter, at most 63 characters"
LOCAL_TYPE = "parquet_dir"


@dataclass
class DatasetSource:
    name: str
    connector_type: str
    config: dict[str, Any]
    created_by: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


def check_name(name: str) -> None:
    """Raises ValueError unless `name` is a valid dataset name."""
    if not NAME_RE.match(name):
        raise ValueError(f"Invalid dataset name {name!r}: use {NAME_RULE}.")


def is_local_folder(path: Path) -> bool:
    """Checks whether `path` is a dataset folder: a table subfolder holding a Parquet file."""
    return path.is_dir() and any(path.glob("*/*.parquet"))


class DatasetRegistry:
    def __init__(self, config: Config, logger: Logger, root: Path, dsn: str | None = None) -> None:
        self.logger = logger
        self.root = Path(root)
        self.dsn = dsn or (
            f"postgresql://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}"
            f"@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
        )
        try:
            self._init_db()
            self.available = True
        except Exception as exc:
            self.logger.warning(
                f"[DatasetRegistry] Postgres unavailable, only local dataset folders are listed: {exc}"
            )
            self.available = False

    def list_sources(self) -> list[DatasetSource]:
        """Lists every dataset: registered ones and unregistered local folders."""
        sources = {s.name: s for s in self._select() if self._present(s)} if self.available else {}
        for name in self._local_folders():
            sources.setdefault(name, self._local_source(name))
        return sorted(sources.values(), key=lambda s: s.name)

    def get(self, name: str) -> DatasetSource | None:
        """Returns the dataset named `name`, or None."""
        if self.available:
            rows = [s for s in self._select(name) if self._present(s)]
            if rows:
                return rows[0]
        if NAME_RE.match(name) and is_local_folder(self.root / name):
            return self._local_source(name)
        return None

    def add(self, source: DatasetSource) -> DatasetSource:
        """Registers a dataset. Raises ValueError if the name is invalid or taken."""
        check_name(source.name)
        self._require_postgres()
        if self.get(source.name) is not None:
            raise ValueError(f"A dataset named {source.name!r} already exists.")
        self.remove(source.name)  # a leftover row of a deleted local folder
        return self._insert(source)

    def register_local(self, name: str, created_by: str | None = None) -> None:
        """Records a local dataset folder in the registry (a no-op without Postgres or if recorded)."""
        if self.available and is_local_folder(self.root / name):
            self._insert(self._local_source(name, created_by))

    def update_config(self, name: str, config: dict[str, Any]) -> DatasetSource:
        """Replaces a dataset's connector config. Raises ValueError if it isn't registered."""
        self._require_postgres()
        with psycopg.connect(self.dsn, autocommit=True) as con:
            row = con.execute(
                "UPDATE dataset_sources SET config = %s, updated_at = NOW() WHERE name = %s "
                "RETURNING name, connector_type, config, created_by, created_at, updated_at",
                (Jsonb(config), name),
            ).fetchone()
        if row is None:
            raise ValueError(f"No dataset named {name!r}.")
        return DatasetSource(*row)

    def remove(self, name: str) -> None:
        """Unregisters a dataset (its data is left alone)."""
        if self.available:
            with psycopg.connect(self.dsn, autocommit=True) as con:
                con.execute("DELETE FROM dataset_sources WHERE name = %s", (name,))

    def _init_db(self) -> None:
        with psycopg.connect(self.dsn, autocommit=True) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS dataset_sources (
                    name            VARCHAR PRIMARY KEY,
                    connector_type  VARCHAR NOT NULL,
                    config          JSONB NOT NULL,
                    created_by      VARCHAR,
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    updated_at      TIMESTAMPTZ DEFAULT NOW()
                )
                """)

    def _require_postgres(self) -> None:
        if not self.available:
            raise ValueError("Datasets can't be registered: Postgres is unavailable.")

    def _select(self, name: str | None = None) -> list[DatasetSource]:
        with psycopg.connect(self.dsn, autocommit=True) as con:
            if name is None:
                rows = con.execute(
                    "SELECT name, connector_type, config, created_by, created_at, updated_at "
                    "FROM dataset_sources ORDER BY name"
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT name, connector_type, config, created_by, created_at, updated_at "
                    "FROM dataset_sources WHERE name = %s",
                    (name,),
                ).fetchall()
        return [DatasetSource(*row) for row in rows]

    def _insert(self, source: DatasetSource) -> DatasetSource:
        with psycopg.connect(self.dsn, autocommit=True) as con:
            row = con.execute(
                "INSERT INTO dataset_sources (name, connector_type, config, created_by) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (name) DO NOTHING "
                "RETURNING name, connector_type, config, created_by, created_at, updated_at",
                (source.name, source.connector_type, Jsonb(source.config), source.created_by),
            ).fetchone()
        if row is not None:
            return DatasetSource(*row)
        existing = self._select(source.name)
        if not existing:  # removed concurrently
            raise ValueError(f"Couldn't register dataset {source.name!r}.")
        return existing[0]

    def _local_folders(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if NAME_RE.match(p.name) and is_local_folder(p)
        )

    def _present(self, source: DatasetSource) -> bool:
        # A local dataset is its folder: a row for a folder not under this root
        # (deleted, or written under another root) doesn't count.
        path = Path(source.config.get("path", ""))
        if source.connector_type != LOCAL_TYPE or path.is_absolute():
            return True
        return is_local_folder(self.root / path)

    @staticmethod
    def _local_source(name: str, created_by: str | None = None) -> DatasetSource:
        # The path is relative to the datasets root, so the root can move.
        return DatasetSource(name, LOCAL_TYPE, {"path": name, "file_pattern": "*.parquet"}, created_by)
