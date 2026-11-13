"""
Migrates datasets from the old layout (<root>/<ds>/<ds>.db plus metadata.csv) to
the current one: each table becomes <root>/<ds>/<table>/<table>.parquet, and
table descriptions move into the memory layer (Postgres). The old files are
deleted only after every table's row count is verified.

    python -m pneuma_seeker.services.db.datasets.migrate [--root DIR] [--keep-legacy] [--dry-run]
"""

import argparse
import shutil
from pathlib import Path

import duckdb
from pandas import read_csv

from pneuma_seeker.services.db.datasets.connectors import quote_ident, sql_literal
from pneuma_seeker.services.db.datasets.uploads import dataset_lock, staging_dir
from pneuma_seeker.services.db.pneuma_db import PneumaDB
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger
from pneuma_seeker.shared.str_processor import clean_column_table_name


def legacy_datasets(root: Path) -> list[Path]:
    """Returns the dataset folders still in the old layout."""
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / f"{p.name}.db").is_file())


def migrate_dataset(dataset_dir: Path, root: Path) -> dict[str, int]:
    """Converts a dataset's .db file to Parquet tables. Returns the row count per table."""
    db_file = dataset_dir / f"{dataset_dir.name}.db"
    counts: dict[str, int] = {}
    with dataset_lock(root, dataset_dir.name):
        staging = staging_dir(root)
        try:
            con = duckdb.connect(db_file.as_posix(), read_only=True)
            try:
                tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
                for table in tables:
                    out = staging / table / f"{table}.parquet"
                    out.parent.mkdir()
                    con.execute(
                        f"COPY {quote_ident(table)} TO {sql_literal(out.as_posix())} (FORMAT parquet, COMPRESSION zstd)"
                    )
                    (expected,) = con.execute(f"SELECT count(*) FROM {quote_ident(table)}").fetchone()  # type: ignore[misc]
                    (written,) = con.execute(f"SELECT count(*) FROM read_parquet({sql_literal(out.as_posix())})").fetchone()  # type: ignore[misc]
                    if expected != written:
                        raise RuntimeError(f"{dataset_dir.name}.{table}: wrote {written} of {expected} rows")
                    counts[table] = int(written)
            finally:
                con.close()
            for table in counts:
                target = dataset_dir / table
                if target.exists():
                    shutil.rmtree(target)
                (staging / table).rename(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return counts


def import_descriptions(pneuma_db: PneumaDB, dataset_name: str, metadata_file: Path) -> int:
    """
    Stores a metadata CSV's table descriptions (table_name, description) in the
    memory layer, keeping ones already there. Returns how many were added.
    """
    memory = pneuma_db.memory_manager
    if not metadata_file.is_file() or memory is None:
        return 0
    metadata = read_csv(metadata_file)
    if not {"table_name", "description"} <= set(metadata.columns):
        return 0
    added = 0
    for name, description in zip(metadata["table_name"], metadata["description"]):
        table = clean_column_table_name(str(name))
        if not isinstance(description, str) or not description.strip():
            continue
        if memory.get_by_key("table_metadata", dataset_name, table) is None:
            memory.create_entry(
                "table_metadata", description.strip(), "import",
                dataset_name=dataset_name, key_text=table,
            )
            added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="datasets root (default: Config.DATASETS_ROOT)")
    parser.add_argument("--keep-legacy", action="store_true", help="keep <ds>.db and metadata.csv")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be migrated")
    args = parser.parse_args()

    config = Config()
    root = Path(args.root or config.DATASETS_ROOT)
    pending = legacy_datasets(root)
    print(f"{len(pending)} dataset(s) to migrate under {root}: {', '.join(p.name for p in pending) or '-'}")
    if args.dry_run or not pending:
        return
    pneuma_db = PneumaDB(config, setup_logger("Dataset Migration"), dataset_db_path=root.as_posix())
    if pneuma_db.memory_manager is None:
        raise SystemExit("Postgres is unavailable: start it first, so table descriptions can be moved there.")
    for dataset_dir in pending:
        counts = migrate_dataset(dataset_dir, root)
        added = import_descriptions(pneuma_db, dataset_dir.name, dataset_dir / "metadata.csv")
        pneuma_db.dataset_manager.registry.register_local(dataset_dir.name, created_by="migration")
        if not args.keep_legacy:
            for name in (f"{dataset_dir.name}.db", f"{dataset_dir.name}.db.wal", "metadata.csv"):
                (dataset_dir / name).unlink(missing_ok=True)
        print(f"{dataset_dir.name}: {len(counts)} tables, {sum(counts.values())} rows, {added} descriptions added")


if __name__ == "__main__":
    main()
