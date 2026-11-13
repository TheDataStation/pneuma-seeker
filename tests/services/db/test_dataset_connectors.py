import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import duckdb

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from unittest.mock import MagicMock

from pneuma_seeker.services.db.datasets import download, migrate, uploads
from pneuma_seeker.services.db.datasets.connectors import DatasetUnavailable, build
from pneuma_seeker.services.db.datasets.registry import DatasetRegistry, DatasetSource
from pneuma_seeker.services.db.pneuma_db import PneumaDB
from pneuma_seeker.shared.config import Config


def offline_config() -> Config:
    """Returns a config whose Postgres is unreachable, so tests never touch a real database."""
    config = Config()
    config.POSTGRES_HOST, config.POSTGRES_PORT = "127.0.0.1", 1
    return config


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name: str, text: str) -> Path:
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path


def attached(connector, alias: str = "ds") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    connector.attach(con, alias)
    return con


class FileConnectorTests(TempDirTestCase):
    def test_parquet_dir_exposes_one_cleaned_view_per_table_folder(self):
        (self.tmp / "pq" / "My Table").mkdir(parents=True)
        (self.tmp / "pq" / "empty").mkdir()
        for i in (1, 2):  # a table may be split across files
            duckdb.sql(f"COPY (SELECT {i} AS \"Some Col\") TO '{self.tmp}/pq/My Table/part{i}.parquet' (FORMAT parquet)")
        connector = build("parquet_dir", {"path": str(self.tmp / "pq")})
        self.assertEqual(connector.table_names(), ["my_table"])
        con = attached(connector)
        self.assertEqual(sorted(con.execute('SELECT some_col FROM ds."my_table"').fetchall()), [(1,), (2,)])

    def test_csv_dir_reads_csv_and_tsv_files(self):
        self.write("csv/Sales 2024.csv", "Amount ($),Region\n10,east\n")
        self.write("csv/t.tsv", "a\tb\n1\t2\n")
        connector = build("csv_dir", {"path": str(self.tmp / "csv"), "file_pattern": "*.*sv"})
        con = attached(connector)
        self.assertEqual(connector.table_names(), ["sales_2024", "t"])
        self.assertEqual(con.execute('SELECT amount_usd, region FROM ds."sales_2024"').fetchall(), [(10, "east")])
        self.assertEqual(con.execute('SELECT b FROM ds."t"').fetchall(), [(2,)])

    def test_duckdb_and_sqlite_files_are_attached_read_only(self):
        db = duckdb.connect(str(self.tmp / "x.duckdb"))
        db.execute('CREATE TABLE "Big Table" AS SELECT 1 AS "A B"')
        db.close()
        sq = sqlite3.connect(self.tmp / "x.sqlite")
        sq.execute('CREATE TABLE Foo ("Bar Baz" INT)')
        sq.execute("INSERT INTO Foo VALUES (5)")
        sq.commit()
        sq.close()
        for kind, file, table, column, value in (
            ("duckdb", "x.duckdb", "big_table", "a_b", 1),
            ("sqlite", "x.sqlite", "foo", "bar_baz", 5),
        ):
            con = attached(build(kind, {"path": str(self.tmp / file)}))
            self.assertEqual(con.execute(f'SELECT {column} FROM ds."{table}"').fetchall(), [(value,)])
            with self.assertRaises(duckdb.Error):
                con.execute("CREATE TABLE ds__source.main.nope AS SELECT 1")

    def test_missing_sources_are_unavailable(self):
        for kind, config in (
            ("parquet_dir", {"path": str(self.tmp / "nope")}),
            ("csv_dir", {"path": str(self.tmp / "nope")}),
            ("duckdb", {"path": str(self.tmp / "nope.duckdb")}),
        ):
            connector = build(kind, config)
            with self.assertRaises(DatasetUnavailable):
                attached(connector)
            ok, error, count = connector.health_check()
            self.assertFalse(ok)
            self.assertIn("not found", error)

    def test_fingerprint_changes_with_the_files(self):
        self.write("csv/a.csv", "x\n1\n")
        connector = build("csv_dir", {"path": str(self.tmp / "csv")})
        before = connector.fingerprint()
        self.write("csv/b.csv", "x\n2\n")
        self.assertNotEqual(connector.fingerprint(), before)

    def test_file_patterns_cannot_leave_the_folder(self):
        for pattern in ("../*.csv", "/etc/*"):
            with self.assertRaises(ValueError):
                build("csv_dir", {"path": str(self.tmp), "file_pattern": pattern})

    def test_lockdown_limits_reads_to_the_source(self):
        self.write("csv/a.csv", "x\n1\n")
        outside = self.write("secret.csv", "s\n1\n")
        connector = build("csv_dir", {"path": str(self.tmp / "csv")})
        con = attached(connector)
        connector.lockdown(con)
        self.assertEqual(con.execute('SELECT x FROM ds."a"').fetchall(), [(1,)])
        with self.assertRaises(duckdb.Error):
            con.execute(f"SELECT * FROM read_csv('{outside}')")


class PostgresConnectorTests(unittest.TestCase):
    config = {"host": "db.example", "port": 5433, "dbname": "sales", "user": "a@b", "password": "p@ss/word"}

    def test_uri_is_percent_encoded_and_secrets_are_masked(self):
        connector = build("postgres", {**self.config, "schema": "crm"})
        self.assertEqual(connector.connection_uri(), "postgresql://a%40b:p%40ss%2Fword@db.example:5433/sales")
        self.assertEqual(connector.redacted_config()["password"], "••••••••")
        self.assertEqual(connector.redacted_config()["schema"], "crm")
        self.assertNotIn("p@ss/word", connector.scrub(f"failed: {connector.connection_uri()} p@ss/word"))

    def test_unreachable_server_is_reported_without_the_password(self):
        connector = build("postgres", {**self.config, "host": "127.0.0.1", "port": 1})
        ok, error, _ = connector.health_check()
        self.assertFalse(ok)
        self.assertNotIn("p@ss/word", error)

    @unittest.skipUnless(os.getenv("PNEUMA_TEST_POSTGRES_DSN"), "set PNEUMA_TEST_POSTGRES_DSN to test a live Postgres")
    def test_live_database_tables_are_views(self):
        from urllib.parse import urlparse

        dsn = urlparse(os.environ["PNEUMA_TEST_POSTGRES_DSN"])
        connector = build("postgres", {
            "host": dsn.hostname, "port": dsn.port or 5432, "dbname": dsn.path.lstrip("/"),
            "user": dsn.username, "password": dsn.password or "",
        })
        ok, error, _ = connector.health_check()
        self.assertTrue(ok, error)


class UploadTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "datasets"

    def add(self, file: Path, **kwargs) -> uploads.UploadResult:
        return uploads.add_file(self.root, "ds", file, file.name, **kwargs)

    def table(self, name: str) -> list[tuple]:
        return duckdb.sql(f"SELECT * FROM read_parquet('{self.root}/ds/{name}/{name}.parquet')").fetchall()

    def test_formats_become_parquet_tables_with_cleaned_names(self):
        result = self.add(self.write("Orders 2024.csv", "Order ID,Total ($)\n1,9.5\n"), create=True)
        self.assertTrue(result.created_dataset)
        self.assertEqual([(t.name, t.rows, t.columns) for t in result.tables], [("orders_2024", 1, ["order_id", "total_usd"])])
        self.add(self.write("t.tsv", "a\tb\n1\t2\n"))
        duckdb.sql(f"COPY (SELECT 3 AS c) TO '{self.tmp}/p.parquet' (FORMAT parquet)")
        self.add(self.tmp / "p.parquet")
        duckdb.sql("INSTALL excel; LOAD excel")
        duckdb.sql(f"COPY (SELECT 4 AS \"D E\") TO '{self.tmp}/book.xlsx' (FORMAT xlsx, HEADER true)")
        self.add(self.tmp / "book.xlsx")
        self.assertEqual(sorted(p.name for p in (self.root / "ds").iterdir()), ["orders_2024", "p", "sheet1", "t"])
        self.assertEqual(self.table("t"), [(1, 2)])
        self.assertEqual(self.table("p"), [(3,)])
        self.assertEqual(self.table("sheet1"), [(4,)])

    def test_existing_tables_need_replace(self):
        self.add(self.write("t.csv", "a\n1\n"), create=True)
        with self.assertRaises(uploads.UploadRefused) as refused:
            self.add(self.write("t.csv", "a\n2\n"))
        self.assertEqual(refused.exception.kind, "table_exists")
        self.assertEqual(self.table("t"), [(1,)])
        result = self.add(self.write("t.csv", "a\n2\n"), replace=True)
        self.assertTrue(result.tables[0].replaced)
        self.assertEqual(self.table("t"), [(2,)])

    def test_refusals_change_nothing(self):
        with self.assertRaises(uploads.UploadRefused) as refused:
            self.add(self.write("t.csv", "a\n1\n"))  # no dataset yet and create not set
        self.assertEqual(refused.exception.kind, "not_found")
        for name, kind in (("x.xls", "invalid"), ("x.json", "invalid")):
            with self.assertRaises(uploads.UploadRefused) as refused:
                self.add(self.write(name, "{}"), create=True)
            self.assertEqual(refused.exception.kind, kind)
        with self.assertRaises(uploads.UploadRefused):
            self.add(self.write("bad.xlsx", "not a workbook"), create=True)
        with self.assertRaises(uploads.UploadRefused):
            uploads.add_file(self.root, "Bad Name", self.write("t.csv", "a\n1\n"), "t.csv", create=True)
        self.assertFalse((self.root / "ds").exists())
        self.assertEqual(list((self.root / ".staging").iterdir()), [])

    def test_delete_table_and_dataset(self):
        self.add(self.write("a.csv", "x\n1\n"), create=True)
        self.add(self.write("b.csv", "x\n1\n"))
        uploads.delete_table(self.root, "ds", "a")
        self.assertEqual([p.name for p in (self.root / "ds").iterdir()], ["b"])
        with self.assertRaises(uploads.UploadRefused):
            uploads.delete_table(self.root, "ds", "a")
        uploads.delete_dataset_files(self.root, "ds")
        self.assertFalse((self.root / "ds").exists())


class RegistryAndManagerTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.db = PneumaDB(
            offline_config(), logging.getLogger("t"),
            dataset_db_path=str(self.tmp / "datasets"), workspace_db_path=str(self.tmp / "ws"),
        )
        self.manager = self.db.dataset_manager

    def tearDown(self):
        self.db.close_all_connections()
        super().tearDown()

    def test_without_postgres_local_folders_are_the_datasets(self):
        registry = DatasetRegistry(offline_config(), logging.getLogger("t"), self.tmp / "datasets")
        self.assertFalse(registry.available)
        uploads.add_file(self.tmp / "datasets", "local_ds", self.write("t.csv", "a\n1\n"), "t.csv", create=True)
        (self.tmp / "datasets" / "not_a_dataset").mkdir()
        self.assertEqual([s.name for s in registry.list_sources()], ["local_ds"])
        source = registry.get("local_ds")
        self.assertEqual((source.connector_type, source.config["path"]), ("parquet_dir", "local_ds"))
        self.assertIsNone(registry.get("not_a_dataset"))
        with self.assertRaises(ValueError):
            registry.add(DatasetSource("ext", "csv_dir", {"path": str(self.tmp)}))

    def test_local_and_connected_datasets(self):
        self.manager.add_file("up", self.write("t.csv", "a\n1\n"), "t.csv", create=True)
        source = self.manager.get_source("up")
        self.assertTrue(self.manager.is_local(source))
        self.assertFalse(self.manager.is_local(DatasetSource("ext", "csv_dir", {"path": str(self.tmp)})))
        self.assertFalse(self.manager.is_local(DatasetSource("odd", "parquet_dir", {"path": "/elsewhere"})))

    def test_files_cannot_be_added_to_a_connected_source(self):
        self.manager.registry.get = lambda name: DatasetSource(name, "postgres", {})  # type: ignore[method-assign]
        with self.assertRaises(uploads.UploadRefused):
            self.manager.add_file("ext", self.write("t.csv", "a\n1\n"), "t.csv")

    def test_adding_a_source_checks_name_type_and_reachability(self):
        for name, kind, config in (
            ("Bad Name", "csv_dir", {"path": str(self.tmp)}),
            ("ok_name", "mystery", {}),
            ("ok_name", "csv_dir", {"path": str(self.tmp / "missing")}),
        ):
            with self.assertRaises(ValueError):
                self.manager.add_source(name, kind, config, None)


class MigrationTests(TempDirTestCase):
    def test_legacy_db_becomes_parquet_tables_with_matching_counts(self):
        root = self.tmp / "datasets"
        (root / "old").mkdir(parents=True)
        con = duckdb.connect(str(root / "old" / "old.db"))
        con.execute("CREATE TABLE a AS SELECT * FROM range(5) t(x)")
        con.execute("CREATE TABLE b AS SELECT 'v' AS y")
        con.close()
        self.assertEqual([p.name for p in migrate.legacy_datasets(root)], ["old"])
        counts = migrate.migrate_dataset(root / "old", root)
        self.assertEqual(counts, {"a": 5, "b": 1})
        connector = build("parquet_dir", {"path": str(root / "old")})
        self.assertEqual(connector.table_names(), ["a", "b"])
        self.assertTrue((root / "old" / "old.db").exists())  # removed only by main(), after every dataset converts


class DownloadTests(TempDirTestCase):
    def test_pack_and_install_round_trip_with_descriptions(self):
        root, archives = self.tmp / "datasets", self.tmp / "archives"
        uploads.add_file(root, "env", self.write("Rain.csv", "year,jul\n2020,1.9\n"), "Rain.csv", create=True)
        self.write("data_src/env/metadata.csv", "table_name,description\nRain,July rainfall\nmissing,\n")
        archive = download.pack(root, "env", archives, self.tmp / "data_src")

        fresh = self.tmp / "fresh"
        metadata = download.install(archive, fresh, "env", self.tmp)
        self.assertEqual(build("parquet_dir", {"path": str(fresh / "env")}).table_names(), ["rain"])
        self.assertFalse((fresh / "env" / "metadata.csv").exists())

        memory = MagicMock()
        memory.get_by_key.return_value = None
        added = migrate.import_descriptions(MagicMock(memory_manager=memory), "env", metadata)
        self.assertEqual(added, 1)
        memory.create_entry.assert_called_once_with(
            "table_metadata", "July rainfall", "import", dataset_name="env", key_text="rain"
        )

    def test_existing_descriptions_are_kept(self):
        metadata = self.write("metadata.csv", "table_name,description\nrain,New\n")
        memory = MagicMock()
        memory.get_by_key.return_value = MagicMock(content="Edited in the UI")
        self.assertEqual(migrate.import_descriptions(MagicMock(memory_manager=memory), "env", metadata), 0)
        memory.create_entry.assert_not_called()

    def test_drive_folder_listing_is_parsed(self):
        page = (
            '<div class="flip-entry"><a href="https://drive.google.com/file/d/1FsUarNNbwsJ9XC5lv/view?usp=drive_web" '
            'target="_blank"><div class="flip-entry-title">archeology.zip</div></a></div>'
            '<div class="flip-entry"><a href="https://drive.google.com/file/d/1SXRG4HUy-evW2eLBE/view?usp=drive_web">'
            '<div class="flip-entry-info"><div class="flip-entry-title">legal &amp; more.zip</div></div></a></div>'
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = page.encode()
        with unittest.mock.patch.object(download.urllib.request, "urlopen", return_value=response) as urlopen:
            files = download.drive_folder_files("https://drive.google.com/drive/folders/1xZhw6ITGIq2st?usp=sharing")
        self.assertEqual(files, {"archeology.zip": "1FsUarNNbwsJ9XC5lv", "legal & more.zip": "1SXRG4HUy-evW2eLBE"})
        self.assertIn("embeddedfolderview?id=1xZhw6ITGIq2st", urlopen.call_args[0][0])

    def test_unsafe_archives_are_refused(self):
        bad = self.tmp / "bad.zip"
        import zipfile

        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("../evil.txt", "x")
        with self.assertRaises(SystemExit):
            download.install(bad, self.tmp / "fresh", "env", self.tmp)
        self.assertFalse((self.tmp / "evil.txt").exists())


if __name__ == "__main__":
    unittest.main()
