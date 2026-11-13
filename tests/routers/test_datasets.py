import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from pneuma_seeker.routers import datasets
from pneuma_seeker.routers.auth import get_current_user
from pneuma_seeker.services.db.pneuma_db import PneumaDB
from pneuma_seeker.services.db.users.models import UserRecord
from pneuma_seeker.shared.config import Config


class DatasetsRouterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        config = Config()
        config.POSTGRES_HOST, config.POSTGRES_PORT = "127.0.0.1", 1  # never a real database
        config.DATASETS_ROOT = str(self.tmp / "datasets")
        config.RETRIEVAL_INDEX_ROOT = str(self.tmp / "indices")
        config.UPLOAD_MAX_BYTES = 1024
        self.db = PneumaDB(
            config, logging.getLogger("t"),
            dataset_db_path=config.DATASETS_ROOT, workspace_db_path=str(self.tmp / "ws"),
        )
        self.is_admin = True
        self.patches = [
            patch.object(datasets, "pneuma_db", self.db),
            patch.object(datasets, "config", config),
            patch.object(datasets, "_resolve_dataset_permissions", lambda user: (self.is_admin, {
                "dataset:access:shared": "true",
            })),
        ]
        for p in self.patches:
            p.start()
        app = FastAPI()
        app.include_router(datasets.router)
        app.dependency_overrides[get_current_user] = lambda: UserRecord("u1", "u@x.io", "u", "g1", True)
        self.client = TestClient(app)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.db.close_all_connections()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def upload(self, dataset: str, name: str, body: bytes, **params):
        return self.client.put(f"/datasets/{dataset}/upload/{name}", content=body, params=params)

    def test_upload_creates_a_dataset_and_refuses_clashes_until_replace(self):
        response = self.upload("shared", "Sales 2024.csv", b"Amount ($),Region\n10,east\n", create=True)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["tables"], [{"name": "sales_2024", "rows": 1, "columns": 2, "replaced": False}])
        self.assertTrue((self.tmp / "datasets" / "shared" / "sales_2024" / "sales_2024.parquet").is_file())

        self.assertEqual(self.upload("shared", "Sales 2024.csv", b"a\n1\n").status_code, 409)
        replaced = self.upload("shared", "Sales 2024.csv", b"a\n1\n", replace=True)
        self.assertTrue(replaced.json()["tables"][0]["replaced"])

        listed = self.client.get("/datasets").json()
        self.assertEqual(listed["datasets"][0]["name"], "shared")
        self.assertTrue(listed["datasets"][0]["is_local"])
        self.assertTrue(listed["can_manage"])

    def test_upload_refusals(self):
        self.assertEqual(self.upload("shared", "t.csv", b"a\n1\n").status_code, 404)  # no dataset, no create
        self.assertEqual(self.upload("shared", "t.json", b"{}", create=True).status_code, 400)
        self.assertEqual(self.upload("shared", "t.csv", b"x" * 2048, create=True).status_code, 413)
        self.assertEqual(self.upload("Bad Name", "t.csv", b"a\n1\n", create=True).status_code, 400)
        self.assertEqual(list((self.tmp / "datasets" / ".staging").iterdir()), [])

    def test_delete_table_and_remove_dataset(self):
        self.upload("shared", "a.csv", b"x\n1\n", create=True)
        self.upload("shared", "b.csv", b"x\n1\n")
        self.assertEqual(self.client.delete("/datasets/shared/tables/a").status_code, 200)
        self.assertEqual(self.client.delete("/datasets/shared/tables/a").status_code, 404)
        self.assertEqual(self.client.delete("/datasets/shared").status_code, 200)
        self.assertFalse((self.tmp / "datasets" / "shared").exists())
        self.assertEqual(self.client.get("/datasets").json()["datasets"], [])

    def test_testing_a_source_reports_reachability(self):
        (self.tmp / "csvs").mkdir()
        (self.tmp / "csvs" / "t.csv").write_text("a\n1\n")
        ok = self.client.post("/datasets/sources/test", json={"connector_type": "csv_dir", "config": {"path": str(self.tmp / "csvs")}})
        self.assertEqual(ok.json(), {"ok": True, "error": None, "table_count": 1})
        missing = self.client.post("/datasets/sources/test", json={"connector_type": "csv_dir", "config": {"path": "/nope"}})
        self.assertFalse(missing.json()["ok"])
        unknown = self.client.post("/datasets/sources/test", json={"connector_type": "mystery", "config": {}})
        self.assertEqual(unknown.status_code, 400)

    def test_index_is_rebuilt_on_demand_for_m1(self):
        datasets.config.TABLE_DISCOVERY_METHOD = "m1"
        self.upload("shared", "rain.csv", b"year,jul\n2020,1.9\n", create=True)
        self.assertEqual(self.client.get("/datasets/shared/index").json()["state"], "missing")
        started = self.client.post("/datasets/shared/index")  # TestClient runs the background task before returning
        self.assertEqual(started.status_code, 200)
        status = self.client.get("/datasets/shared/index").json()
        self.assertEqual((status["method"], status["state"]), ("m1", "built"))
        self.assertIsNotNone(status["built_at"])
        self.assertEqual(self.client.post("/datasets/nope/index").status_code, 404)
        self.is_admin = False
        self.assertEqual(self.client.post("/datasets/shared/index").status_code, 403)

    def test_non_admins_only_list_what_they_may_access(self):
        self.upload("shared", "t.csv", b"a\n1\n", create=True)
        self.upload("private", "t.csv", b"a\n1\n", create=True)
        self.is_admin = False
        listed = self.client.get("/datasets").json()
        self.assertEqual([d["name"] for d in listed["datasets"]], ["shared"])
        self.assertFalse(listed["can_manage"])
        for response in (
            self.upload("shared", "u.csv", b"a\n1\n"),
            self.client.delete("/datasets/shared"),
            self.client.post("/datasets/sources/test", json={"connector_type": "csv_dir", "config": {"path": "/"}}),
            self.client.get("/datasets/shared/source"),
        ):
            self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
