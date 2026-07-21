import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
)

import numpy as np
import pandas as pd

from pneuma_seeker.services.core.action_set.impl.semantic_join import SemanticJoin
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.shared.config import Config


class SemanticJoinTests(unittest.TestCase):
    """Unit tests for SemanticJoin."""

    def setUp(self) -> None:
        self.config = Config()
        self.logger = MagicMock()
        self.db_api = MagicMock()
        self.lm_api = MagicMock()
        self.semantic_join = SemanticJoin(
            "user_id", "chat_id", self.config, self.logger, self.db_api, self.lm_api
        )

    def test_apply_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            self.semantic_join.apply(
                {
                    "left_table_id": 1,
                    "right_table_id": "right",
                    "relevant_left_cols": ["name"],
                    "relevant_right_cols": ["name"],
                    "joined_table_id": "joined",
                }
            )

        with self.assertRaises(ValueError):
            self.semantic_join.apply(
                {
                    "left_table_id": "left",
                    "right_table_id": None,
                    "relevant_left_cols": ["name"],
                    "relevant_right_cols": ["name"],
                    "joined_table_id": "joined",
                }
            )

        with self.assertRaises(ValueError):
            self.semantic_join.apply(
                {
                    "left_table_id": "left",
                    "right_table_id": "right",
                    "relevant_left_cols": "name",
                    "relevant_right_cols": ["name"],
                    "joined_table_id": "joined",
                }
            )

        with self.assertRaises(ValueError):
            self.semantic_join.apply(
                {
                    "left_table_id": "left",
                    "right_table_id": "right",
                    "relevant_left_cols": ["name"],
                    "relevant_right_cols": "name",
                    "joined_table_id": "joined",
                }
            )

    def _build_execute_query_side_effect(
        self,
        left_table_id: str,
        right_table_id: str,
        joined_table_id: str,
        left_table: pd.DataFrame,
        right_table: pd.DataFrame,
        inserted_pairs: list[tuple[int, int, float]],
    ):
        def _parse_limit_offset(sql: str) -> tuple[int, int]:
            limit_match = re.search(r"LIMIT\s+(\d+)", sql, re.IGNORECASE)
            offset_match = re.search(r"OFFSET\s+(\d+)", sql, re.IGNORECASE)
            limit = int(limit_match.group(1)) if limit_match else len(left_table)
            offset = int(offset_match.group(1)) if offset_match else 0
            return limit, offset

        def _slice_table(df: pd.DataFrame, sql: str) -> pd.DataFrame:
            limit, offset = _parse_limit_offset(sql)
            sliced = df.iloc[offset : offset + limit].copy()
            sliced.insert(0, "rowid", sliced.index.astype(int))
            return sliced

        def _select_full_rows(df: pd.DataFrame, sql: str) -> pd.DataFrame:
            rowids_match = re.search(r"rowid\s+IN\s*\(([^)]+)\)", sql)
            if not rowids_match:
                return pd.DataFrame(columns=["rowid", *df.columns])
            rowids = [int(x.strip()) for x in rowids_match.group(1).split(",")]
            sliced = df.iloc[rowids].copy()
            sliced.insert(0, "rowid", sliced.index.astype(int))
            return sliced

        def _select_joined_output() -> pd.DataFrame:
            rows: list[dict[str, object]] = []
            for left_rowid, right_rowid, score in inserted_pairs:
                lrow = left_table.iloc[left_rowid]
                rrow = right_table.iloc[right_rowid]
                row = {
                    **{f"left_{c}": lrow[c] for c in left_table.columns},
                    **{f"right_{c}": rrow[c] for c in right_table.columns},
                    "similarity_score": float(score),
                }
                rows.append(row)
            columns = (
                [f"left_{c}" for c in left_table.columns]
                + [f"right_{c}" for c in right_table.columns]
                + ["similarity_score"]
            )
            return pd.DataFrame(rows, columns=columns)

        def side_effect(user_id, chat_id, sql, sql_params=()):
            normalized = " ".join(sql.strip().split())
            if normalized.upper().startswith("SHOW TABLES"):
                return pd.DataFrame(
                    {"name": [left_table_id, right_table_id, joined_table_id]}
                )
            if f'FROM "{left_table_id}"' in normalized and "LIMIT 0" in normalized:
                return left_table.head(0)
            if f'FROM "{right_table_id}"' in normalized and "LIMIT 0" in normalized:
                return right_table.head(0)
            if f'FROM "{left_table_id}"' in normalized and normalized.startswith(
                "SELECT rowid"
            ):
                return _slice_table(left_table, normalized)
            if f'FROM "{right_table_id}"' in normalized and normalized.startswith(
                "SELECT rowid"
            ):
                return _slice_table(right_table, normalized)
            if (
                f'FROM "{left_table_id}"' in normalized
                and "WHERE rowid IN" in normalized
            ):
                return _select_full_rows(left_table, normalized)
            if (
                f'FROM "{right_table_id}"' in normalized
                and "WHERE rowid IN" in normalized
            ):
                return _select_full_rows(right_table, normalized)
            if normalized.startswith("INSERT INTO"):
                values_match = re.search(r"VALUES\s+(.*)\)\s+AS\s+v", normalized)
                if values_match:
                    values_part = values_match.group(1)
                    tuples = re.findall(
                        r"\(([-\d]+),\s*([-\d]+),\s*([\d\.eE+-]+)\)",
                        values_part,
                    )
                    for left_id, right_id, score in tuples:
                        inserted_pairs.append(
                            (int(left_id), int(right_id), float(score))
                        )
                return pd.DataFrame()
            if normalized.startswith("CREATE OR REPLACE TABLE"):
                return pd.DataFrame()
            if normalized.startswith("DROP TABLE"):
                return pd.DataFrame()
            if normalized.startswith("ALTER TABLE"):
                return pd.DataFrame()
            if normalized.startswith(f'SELECT * FROM "{joined_table_id}"'):
                return _select_joined_output()
            return pd.DataFrame()

        return side_effect

    def test_join_empty_dataframe_returns_empty(self):
        left_table_id = "left_table"
        right_table_id = "right_table"
        joined_table_id = "joined_table"
        left_table = pd.DataFrame({"name": []})
        right_table = pd.DataFrame({"name": ["x"]})
        inserted_pairs: list[tuple[int, int, float]] = []

        self.db_api.execute_query = MagicMock(
            side_effect=self._build_execute_query_side_effect(
                left_table_id,
                right_table_id,
                joined_table_id,
                left_table,
                right_table,
                inserted_pairs,
            )
        )

        result = self.semantic_join.join(
            left_table_id,
            right_table_id,
            relevant_left_cols=["name"],
            relevant_right_cols=["name"],
            joined_table_id=joined_table_id,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(
            list(result.columns),
            ["left_name", "right_name", "similarity_score"],
        )
        self.assertEqual(len(result), 0)

    def test_join_basic_no_llm_with_mock_embeddings(self):
        left_table_id = "left_table"
        right_table_id = "right_table"
        joined_table_id = "joined_table"
        left_df = pd.DataFrame({"name": ["apple", "banana"]})
        right_df = pd.DataFrame({"name": ["apple", "orange"]})
        inserted_pairs: list[tuple[int, int, float]] = []

        left_texts = ["name: apple", "name: banana"]
        right_texts = ["name: apple", "name: orange"]

        left_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        right_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        def encode_side_effect(texts):
            if texts == left_texts:
                return left_emb
            if texts == right_texts:
                return right_emb
            raise AssertionError("Unexpected texts passed to encode")

        self.lm_api.encode = MagicMock(side_effect=encode_side_effect)

        self.db_api.execute_query = MagicMock(
            side_effect=self._build_execute_query_side_effect(
                left_table_id,
                right_table_id,
                joined_table_id,
                left_df,
                right_df,
                inserted_pairs,
            )
        )

        result = self.semantic_join.join(
            left_table_id,
            right_table_id,
            relevant_left_cols=["name"],
            relevant_right_cols=["name"],
            joined_table_id=joined_table_id,
            top_k=1,
            mode="embedding",
            use_llm=False,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["left_name"], "apple")
        self.assertEqual(result.iloc[0]["right_name"], "apple")
        self.assertEqual(result.iloc[1]["left_name"], "banana")
        self.assertEqual(result.iloc[1]["right_name"], "orange")

    def test_join_with_llm_filter(self):
        left_table_id = "left_table"
        right_table_id = "right_table"
        joined_table_id = "joined_table"
        left_df = pd.DataFrame({"name": ["apple"]})
        right_df = pd.DataFrame({"name": ["apple", "orange"]})
        inserted_pairs: list[tuple[int, int, float]] = []

        left_texts = ["name: apple"]
        right_texts = ["name: apple", "name: orange"]

        left_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        right_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        def encode_side_effect(texts):
            if texts == left_texts:
                return left_emb
            if texts == right_texts:
                return right_emb
            raise AssertionError("Unexpected texts passed to encode")

        self.lm_api.encode = MagicMock(side_effect=encode_side_effect)
        self.lm_api.chat = MagicMock(return_value=["[1, 0]"])

        self.db_api.execute_query = MagicMock(
            side_effect=self._build_execute_query_side_effect(
                left_table_id,
                right_table_id,
                joined_table_id,
                left_df,
                right_df,
                inserted_pairs,
            )
        )

        result = self.semantic_join.join(
            left_table_id,
            right_table_id,
            relevant_left_cols=["name"],
            relevant_right_cols=["name"],
            joined_table_id=joined_table_id,
            top_k=2,
            mode="embedding",
            use_llm=True,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["left_name"], "apple")
        self.assertEqual(result.iloc[0]["right_name"], "apple")


class SemanticJoinRealDBTests(unittest.TestCase):
    """
    Integration tests against a real DuckDB-backed DBAPI (mirrors the
    EntityResolutionTests setup) so mode behavior is observable end-to-end
    instead of through a mocked SQL executor.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.SEMANTIC_JOIN_MODE = "jarowinkler"
        self.config.SEMANTIC_JOIN_BATCH_SIZE = 30
        self.logger = MagicMock()
        self.lm_api = MagicMock()
        self.db_api = DBAPI(
            self.config,
            self.logger,
            dataset_db_path=str(Path(self.temp_dir) / "datasets"),
            workspace_db_path=str(Path(self.temp_dir) / "workspaces"),
        )
        self.user_id = "user_id"
        self.chat_id = "chat_id"
        self.action = SemanticJoin(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.lm_api,
        )

    def tearDown(self) -> None:
        self.db_api.pneuma_db.close_all_connections()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_table(self, table_name: str, col_name: str, values: list) -> None:
        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f'CREATE OR REPLACE TABLE "{table_name}" AS '
            f'SELECT unnest({values!r}) AS "{col_name}";',
        )

    def _joined_pairs(self, joined_table_id: str) -> dict[str, str]:
        result = self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f'SELECT left_name, right_name FROM "{joined_table_id}";',
        )
        return dict(zip(result["left_name"], result["right_name"]))

    # ------------------------------------------------------------------
    # jarowinkler mode (default) — no embeddings, scalable
    # ------------------------------------------------------------------

    def test_jarowinkler_is_the_config_default_mode(self):
        self.assertEqual(Config().SEMANTIC_JOIN_MODE, "jarowinkler")

    def test_jarowinkler_mode_matches_typo_variants_without_calling_embeddings(self):
        self._create_table("left_tbl", "name", ["Amazn", "Gooogle"])
        self._create_table("right_tbl", "name", ["Amazon", "Google", "Meta"])

        self.action.apply(
            {
                "left_table_id": "left_tbl",
                "right_table_id": "right_tbl",
                "relevant_left_cols": ["name"],
                "relevant_right_cols": ["name"],
                "joined_table_id": "joined_tbl",
                "top_k": 1,
            }
        )

        pairs = self._joined_pairs("joined_tbl")
        self.assertEqual(pairs["Amazn"], "Amazon")
        self.assertEqual(pairs["Gooogle"], "Google")
        # The whole point of the default mode: it never touches the LM API.
        self.lm_api.encode.assert_not_called()

    # ------------------------------------------------------------------
    # embedding mode
    # ------------------------------------------------------------------

    def test_embedding_mode_uses_cosine_similarity(self):
        self._create_table("left_tbl", "name", ["close_to_A", "close_to_B"])
        self._create_table("right_tbl", "name", ["EntityA", "EntityB"])

        vec_map = {
            "name: EntityA": np.array([1.0, 0.0], dtype=np.float32),
            "name: EntityB": np.array([0.0, 1.0], dtype=np.float32),
            "name: close_to_A": np.array([0.95, 0.05], dtype=np.float32),
            "name: close_to_B": np.array([0.05, 0.95], dtype=np.float32),
        }
        self.lm_api.encode.side_effect = lambda texts: np.array(
            [vec_map[t] for t in texts], dtype=np.float32
        )

        self.action.apply(
            {
                "left_table_id": "left_tbl",
                "right_table_id": "right_tbl",
                "relevant_left_cols": ["name"],
                "relevant_right_cols": ["name"],
                "joined_table_id": "joined_tbl",
                "mode": "embedding",
                "top_k": 1,
            }
        )

        pairs = self._joined_pairs("joined_tbl")
        self.assertEqual(pairs["close_to_A"], "EntityA")
        self.assertEqual(pairs["close_to_B"], "EntityB")
        self.assertTrue(self.lm_api.encode.called)

    # ------------------------------------------------------------------
    # hybrid mode
    # ------------------------------------------------------------------

    def test_hybrid_mode_falls_back_to_jarowinkler_when_embeddings_tie(self):
        # Both right candidates get an identical embedding, so cosine similarity
        # cannot distinguish them. If hybrid is truly blending in the JaroWinkler
        # component, the syntactically closer candidate ("Amazon") must win.
        self._create_table("left_tbl", "name", ["Amzn"])
        self._create_table("right_tbl", "name", ["Amazon", "Zonbay"])

        tied_vec = np.array([1.0, 0.0], dtype=np.float32)
        self.lm_api.encode.side_effect = lambda texts: np.array(
            [tied_vec for _ in texts], dtype=np.float32
        )

        self.action.apply(
            {
                "left_table_id": "left_tbl",
                "right_table_id": "right_tbl",
                "relevant_left_cols": ["name"],
                "relevant_right_cols": ["name"],
                "joined_table_id": "joined_tbl",
                "mode": "hybrid",
                "alpha": 0.5,
                "top_k": 1,
            }
        )

        pairs = self._joined_pairs("joined_tbl")
        self.assertEqual(pairs["Amzn"], "Amazon")
        self.assertTrue(self.lm_api.encode.called)

    # ------------------------------------------------------------------
    # Batching: tables too large to load in one shot
    # ------------------------------------------------------------------

    def test_batching_over_100_row_tables_with_small_batch_size(self):
        # Simulates tables that cannot be loaded instantaneously by forcing a
        # small SEMANTIC_JOIN_BATCH_SIZE, so both the left and right sides are
        # paginated across many round trips (10 left batches x 10 right
        # batches), exercising the cross-batch top-k merge logic.
        n = 100
        left_values = [f"Item_{i:03d}" for i in range(n)]
        right_values = [f"Item_{i:03d}_dup" for i in range(n)]
        self._create_table("left_tbl", "name", left_values)
        self._create_table("right_tbl", "name", right_values)

        self.config.SEMANTIC_JOIN_BATCH_SIZE = 10

        self.action.apply(
            {
                "left_table_id": "left_tbl",
                "right_table_id": "right_tbl",
                "relevant_left_cols": ["name"],
                "relevant_right_cols": ["name"],
                "joined_table_id": "joined_tbl",
                "top_k": 1,
            }
        )

        pairs = self._joined_pairs("joined_tbl")
        self.assertEqual(len(pairs), n)
        for i in range(n):
            left_val = f"Item_{i:03d}"
            self.assertEqual(pairs[left_val], f"{left_val}_dup")
        self.lm_api.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
