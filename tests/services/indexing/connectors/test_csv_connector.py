import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../src")
    ),
)

from pneuma_seeker.services.indexing.connectors.csv_connector import CSVConnector


class TestCSVConnector(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

        pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}).to_csv(
            os.path.join(self.tmpdir, "users.csv"), index=False
        )
        pd.DataFrame({"order_id": [10], "amount": [100]}).to_csv(
            os.path.join(self.tmpdir, "orders.csv"), index=False
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_check_connection_with_directory(self):
        connector = CSVConnector({"type": "csv", "directory_path": self.tmpdir})
        self.assertTrue(connector.check_connection())

    def test_discover_streams(self):
        connector = CSVConnector({"type": "csv", "directory_path": self.tmpdir})
        streams = connector.discover()
        names = [item["stream"] for item in streams]
        self.assertIn("users", names)
        self.assertIn("orders", names)

    def test_read_stream_records(self):
        connector = CSVConnector({"type": "csv", "directory_path": self.tmpdir})
        rows = list(connector.read("users"))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], 1)
        self.assertEqual(rows[0]["name"], "Alice")

    def test_read_unknown_stream_raises(self):
        connector = CSVConnector({"type": "csv", "directory_path": self.tmpdir})
        with self.assertRaises(KeyError):
            list(connector.read("unknown"))


if __name__ == "__main__":
    unittest.main()