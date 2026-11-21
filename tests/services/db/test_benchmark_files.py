import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from pneuma_seeker.services.db.datasets import benchmark_files as bf


class BenchmarkFilesTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.write("archeology/archeology_tabular.json", [{"id": "a-1", "query": "Q?", "answer": 1.5}])
        self.write("archeology/metadata.csv", "table_name,description\ncities,All cities\n")
        for benchmark in ("books", "local"):
            self.write(f"{benchmark}/benchmark.json", [{"id": "q1", "question": "Which?", "answer": "x"}])
            self.write(f"{benchmark}/databases.json", {
                "review_database": {"type": "sqlite", "file": "review.db"},
                f"{benchmark}_info": {"type": "postgres", "file": "info.sql", "port": 5451, "dbname": "db",
                                      "user": "u", "password": "p"},
            })

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, rel: str, content) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content))

    def test_questions_of_both_suites_share_one_shape(self):
        with patch.object(bf, "KRAMABENCH", ["archeology"]), patch.object(bf, "DAB", ["books", "local"]):
            krama = bf.load_questions(self.root, "archeology")
            dab = bf.load_questions(self.root, "books", ["q1"])
            with self.assertRaises(ValueError):
                bf.load_questions(self.root, "books", ["q9"])
            with self.assertRaises(ValueError):
                bf.load_questions(self.root, "nope")
        self.assertEqual(krama, [{"question_id": "a-1", "question": "Q?", "ground_truth": 1.5, "dataset": "archeology"}])
        self.assertEqual(dab[0]["question"], "Which?")

    def test_database_names_shared_across_benchmarks_get_prefixed(self):
        with patch.object(bf, "KRAMABENCH", ["archeology"]), patch.object(bf, "DAB", ["books", "local"]):
            srcs = bf.sources(self.root, "pg-host")
        self.assertEqual(sorted(srcs), ["archeology", "books_info", "books_review_database",
                                        "local_info", "local_review_database"])
        self.assertEqual(srcs["archeology"]["type"], "csv_dir")
        self.assertEqual(srcs["books_info"]["config"]["host"], "pg-host")
        self.assertEqual(srcs["local_review_database"]["config"]["path"],
                         str(self.root / "local" / "dataset" / "review.db"))

    def test_kramabench_metadata_rows_get_the_benchmark_as_database(self):
        rows = bf.metadata_rows(self.root, "archeology")
        self.assertEqual(rows, [{"table_name": "cities", "description": "All cities",
                                 "database": "archeology", "column_name": ""}])

    def test_a_metadata_csv_inside_dataset_is_reported(self):
        self.write("archeology/dataset/metadata.csv", "table_name,description\n")
        with patch.object(bf, "KRAMABENCH", ["archeology"]):
            self.assertEqual(bf.stray_files(self.root), [self.root / "archeology" / "dataset" / "metadata.csv"])


if __name__ == "__main__":
    unittest.main()
