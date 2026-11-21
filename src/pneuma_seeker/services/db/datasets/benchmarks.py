"""
Registers the benchmarks in data_src/ (KramaBench and DataAgentBench, see
data_src/README.md), mounted read-only at /data in the core container:

- Datasets: each KramaBench benchmark as a CSV folder read in place
  (<benchmark>/dataset), and each DataAgentBench database from its SQLite or
  DuckDB file, or its Postgres database (data_src/docker-compose.yml, reached as
  host.docker.internal). A registration of the same name, such as a KramaBench
  Parquet folder, is replaced; its data stays on disk.
- Descriptions from each metadata.csv into the memory layer: DataAgentBench's
  database, table and column descriptions, KramaBench's table descriptions. A
  dataset whose data and metadata.csv rows are unchanged since its descriptions
  were written, and whose descriptions are all still there, is skipped (see the
  benchmark_descriptions table).
- Discovery indices and join profiles (the column sketches join discovery
  compares, see ir_system.join_profiles): built for every dataset whose index
  or profiles are missing or stale.

Datasets an earlier run registered that are not benchmark datasets anymore are
unregistered, with their descriptions. Safe to rerun.

    python -m pneuma_seeker.services.db.datasets.benchmarks [--root DIR] [--postgres-host HOST] [--verbose]
    python -m pneuma_seeker.services.db.datasets.benchmarks status

Progress shows as bars and a summary at the end; --verbose also shows the services' logs.
"""

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

import duckdb
import psycopg
from tqdm import tqdm

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.ir_system.join_profiles import ColumnBar, JoinProfiles
from pneuma_seeker.services.core.ir_system.retriever.impl.new_discovery import NewDiscovery
from pneuma_seeker.services.db.datasets.benchmark_files import metadata_rows, sources, stray_files
from pneuma_seeker.services.db.datasets.connectors import build
from pneuma_seeker.services.db.datasets.connectors.base import clean_names
from pneuma_seeker.services.db.datasets.registry import DatasetSource
from pneuma_seeker.services.db.pneuma_db import PneumaDB
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger

REGISTERED_BY = "benchmarks"
DESCRIPTION_TYPES = ("dataset_metadata", "table_metadata", "column_metadata")
DESCRIPTIONS_VERSION = "d1"  # bump to rewrite every dataset's descriptions after a change in how they are written


class BenchmarkSetup:
    """Registers the benchmark datasets of one data_src/ root in Pneuma-Seeker."""

    def __init__(self, config: Config, logger: logging.Logger, root: Path, postgres_host: str) -> None:
        self.config, self.logger, self.root = config, logger, root
        self.pneuma_db = PneumaDB(config, logger)
        self.memory = self.pneuma_db.memory_manager
        self.registry = self.pneuma_db.dataset_manager.registry
        if self.memory is None:
            raise SystemExit("Pneuma-Seeker's Postgres is unavailable.")
        self.sources = sources(root, postgres_host)

    def run(self) -> None:
        """Registers every benchmark dataset, writes the descriptions and builds the indices."""
        if stray := stray_files(self.root):
            raise SystemExit("These would be read as tables; the descriptions are in <benchmark>/metadata.csv. "
                             "Remove them first:\n  " + "\n  ".join(str(p) for p in stray))
        if stale := self.prune():
            print(f"unregistered (no longer benchmark datasets): {', '.join(stale)}")
        states = Counter()
        with tqdm(self.sources.items(), desc="register", unit="dataset") as bar:
            for name, src in bar:
                bar.set_postfix_str(name)
                states[self.register(name, src)] += 1
        print("datasets: " + ", ".join(f"{n} {state}" for state, n in sorted(states.items())))
        unmatched, written = self.write_descriptions()
        print("descriptions: " + ", ".join(f"{n} {state}" for state, n in sorted(written.items()))
              + f"; {len(unmatched)} unmatched" + (f": {unmatched[:10]}" if unmatched else ""))
        self.build_indices()

    def register(self, name: str, src: dict) -> str:
        """
        Registers a dataset unless it is registered the same way, after checking it can
        be read; returns "registered" or "already registered".
        """
        connector = build(src["type"], src["config"])
        ok, error, tables = connector.health_check()
        if not ok:
            raise SystemExit(f"{name}: {error}")
        current = self.registry.get(name)
        if (current and current.created_by == REGISTERED_BY and current.connector_type == src["type"]
                and current.config == connector.stored_config()):
            return "already registered"
        self.registry.replace(DatasetSource(name, src["type"], connector.stored_config(), REGISTERED_BY))
        return "registered"

    def prune(self) -> list[str]:
        """Unregisters datasets an earlier run registered that are not benchmark datasets anymore."""
        stale = [s.name for s in self.registry.list_sources()
                 if s.created_by == REGISTERED_BY and s.name not in self.sources]
        with psycopg.connect(self.memory.dsn, autocommit=True) as con:
            self._ensure_descriptions_table(con)
            for name in stale:
                self.registry.remove(name)
                con.execute("DELETE FROM memory_entries WHERE dataset_name = %s AND memory_type = ANY(%s)",
                            (name, list(DESCRIPTION_TYPES)))
                con.execute("DELETE FROM benchmark_descriptions WHERE dataset_name = %s", (name,))
        return stale

    @staticmethod
    def _ensure_descriptions_table(con: psycopg.Connection) -> None:
        con.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_descriptions (
                dataset_name VARCHAR PRIMARY KEY,
                fingerprint  VARCHAR NOT NULL,
                entries      INTEGER NOT NULL,
                unmatched    JSONB NOT NULL,
                written_at   TIMESTAMPTZ DEFAULT NOW()
            )
        """)

    def _description_entries(self, con: psycopg.Connection, name: str) -> int:
        return con.execute(
            "SELECT count(*) FROM memory_entries WHERE dataset_name = %s AND memory_type = ANY(%s)",
            (name, list(DESCRIPTION_TYPES)),
        ).fetchone()[0]

    def write_descriptions(self) -> tuple[list[str], Counter]:
        """
        Writes each dataset's descriptions unless they are current; returns (the rows
        naming a table or column that does not exist, {"written"/"unchanged": count}).
        """
        unmatched: list[str] = []
        states: Counter = Counter()
        with psycopg.connect(self.memory.dsn, autocommit=True) as con:
            self._ensure_descriptions_table(con)
            with tqdm(self.sources.items(), desc="descriptions", unit="dataset") as bar:
                for name, src in bar:
                    bar.set_postfix_str(name)
                    rows = [r for r in metadata_rows(self.root, src["benchmark"])
                            if r["database"] == src["database"] and r["description"].strip()]
                    fingerprint = hashlib.sha256(json.dumps(
                        [DESCRIPTIONS_VERSION, self.pneuma_db.dataset_fingerprint(name), rows], sort_keys=True
                    ).encode()).hexdigest()
                    stored = con.execute(
                        "SELECT fingerprint, entries, unmatched FROM benchmark_descriptions WHERE dataset_name = %s",
                        (name,),
                    ).fetchone()
                    if stored and stored[0] == fingerprint and stored[1] == self._description_entries(con, name):
                        unmatched += stored[2]
                        states["unchanged"] += 1
                        continue
                    missing = self._write_dataset_descriptions(name, rows)
                    unmatched += missing
                    con.execute(
                        """
                        INSERT INTO benchmark_descriptions (dataset_name, fingerprint, entries, unmatched)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (dataset_name) DO UPDATE SET fingerprint = EXCLUDED.fingerprint,
                            entries = EXCLUDED.entries, unmatched = EXCLUDED.unmatched, written_at = NOW()
                        """,
                        (name, fingerprint, self._description_entries(con, name), json.dumps(missing)),
                    )
                    states["written"] += 1
        return unmatched, states

    def _write_dataset_descriptions(self, name: str, rows: list[dict]) -> list[str]:
        """Writes one dataset's descriptions from its metadata.csv rows; returns the unmatched ones."""
        unmatched = []
        names = self._names(name)
        table_rows, column_rows = [], []
        for row in rows:
            table, column, description = row["table_name"], row["column_name"], row["description"].strip()
            if not table:
                self._set_dataset_description(name, description)
                continue
            found = names.get(table.lower()) or names.get(Path(table).stem.lower())
            if found is None:
                unmatched.append(f"table {name}.{table}")
                continue
            clean_table, columns = found
            if not column:
                table_rows.append({"key_text": clean_table, "content": description})
            elif column.lower() in columns:
                column_rows.append({"key_text": f"{clean_table}.{columns[column.lower()]}",
                                    "content": description})
            else:
                unmatched.append(f"column {name}.{table}.{column}")
        self.memory.replace_dataset_rows(name, "table_metadata", table_rows, REGISTERED_BY)
        self.memory.replace_dataset_rows(name, "column_metadata", column_rows, REGISTERED_BY)
        return unmatched

    def build_indices(self) -> None:
        """Builds each dataset's discovery index and join profiles unless current, so no chat waits for them."""
        profiles = JoinProfiles(self.config, DBAPI(self.config, self.logger, pneuma_db=self.pneuma_db))
        indices, built_profiles = Counter(), Counter()
        with tqdm(self.sources, desc="index and profile", unit="dataset", position=0) as bar:
            for name in bar:
                bar.set_postfix_str(f"{name} (index)")
                discovery = NewDiscovery("indexing_service", f"index-{name}", self.config,
                                         DBAPI(self.config, self.logger, pneuma_db=self.pneuma_db), None)  # type: ignore[arg-type]
                before = discovery.index_status(name)
                discovery._index(name)  # builds only when missing or stale
                after = discovery.index_status(name)
                indices["current" if before and after and before.get("built_at") == after.get("built_at") else "built"] += 1
                bar.set_postfix_str(f"{name} (join profiles)")
                columns = ColumnBar(name)
                try:
                    built = profiles.build(name, progress=columns)
                finally:
                    columns.close()
                built_profiles["built" if built else "current"] += 1
        print("discovery indices: " + ", ".join(f"{n} {state}" for state, n in sorted(indices.items())))
        print("join profiles: " + ", ".join(f"{n} {state}" for state, n in sorted(built_profiles.items())))

    def status(self) -> None:
        """Prints how many benchmark datasets are registered as expected, and their descriptions."""
        registered = {s.name: (s.connector_type, s.config) for s in self.registry.list_sources()}
        expected = {n: (s["type"], build(s["type"], s["config"]).stored_config()) for n, s in self.sources.items()}
        different = [n for n in self.sources if registered.get(n) != expected[n]]
        with psycopg.connect(self.memory.dsn, autocommit=True) as con:
            counts = dict(con.execute(
                "SELECT memory_type, count(*) FROM memory_entries WHERE dataset_name = ANY(%s) GROUP BY memory_type",
                (list(self.sources),),
            ).fetchall())
        print(f"datasets: {len(self.sources) - len(different)} of {len(self.sources)} registered from {self.root}"
              + (f" (missing or different: {', '.join(different)})" if different else ""))
        print(f"descriptions: {counts.get('dataset_metadata', 0)} dataset, {counts.get('table_metadata', 0)} table, "
              f"{counts.get('column_metadata', 0)} column; tribal knowledge: {counts.get('tribal_knowledge', 0)}")
        profiles = JoinProfiles(self.config, DBAPI(self.config, self.logger, pneuma_db=self.pneuma_db))
        without = [n for n in self.sources if profiles.status(n) != "current"]
        print(f"join profiles: {len(self.sources) - len(without)} of {len(self.sources)} current"
              + (f" (missing or stale: {', '.join(without)})" if without else ""))

    def _names(self, name: str) -> dict[str, tuple[str, dict[str, str]]]:
        """
        Returns {lowercased source table: (table as Pneuma-Seeker names it, {lowercased
        source column: column as Pneuma-Seeker names it})}, since Pneuma-Seeker cleans names.
        """
        connector = self.pneuma_db.dataset_manager.connector(name)
        con = duckdb.connect()
        try:
            relations = connector.relations(con, "dataset")
            out = {}
            for (table, relation), clean_table in zip(relations.items(), clean_names(list(relations))):
                columns = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]
                out[table.lower()] = (clean_table, {c.lower(): n for c, n in zip(columns, clean_names(columns))})
            return out
        finally:
            con.close()

    def _set_dataset_description(self, name: str, content: str) -> None:
        entry = self.memory.get_by_key("dataset_metadata", name, name)
        if entry is None:
            self.memory.create_entry("dataset_metadata", content, REGISTERED_BY, dataset_name=name, key_text=name)
        elif entry.content != content:
            self.memory.update_entry(entry.memory_id, content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=["setup", "status"], default="setup")
    parser.add_argument("--root", default="/data", help="the data_src/ folder (default: /data, its mount in core)")
    parser.add_argument("--postgres-host", default="host.docker.internal",
                        help="host of DataAgentBench's Postgres databases (default: host.docker.internal)")
    parser.add_argument("--verbose", action="store_true", help="also show the services' logs")
    args = parser.parse_args()
    if not args.verbose:
        logging.disable(logging.INFO)  # the bars and the summary show the progress
    setup = BenchmarkSetup(Config(), setup_logger("Benchmark Setup"), Path(args.root), args.postgres_host)
    if args.command == "setup":
        setup.run()
    setup.status()


if __name__ == "__main__":
    main()
