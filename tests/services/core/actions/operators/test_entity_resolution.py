import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
)

from pneuma_seeker.services.core.action_set.impl.entity_resolution import (
    EntityResolution,
    _UNRESOLVED_SENTINEL,
)
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.shared.config import Config


class EntityResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.DATA_SOURCES = ["test_ds"]
        self.config.ENTITY_RESOLUTION_THRESHOLD = 0.85
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
        self.action = EntityResolution(
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

    def _create_workspace_table(self, table_name: str, values: list[str]) -> None:
        escaped = ", ".join(f"('{v.replace(chr(39), chr(39)+chr(39))}')" for v in values)
        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT unnest([{", ".join(repr(v) for v in values)}]) AS merchant;',
        )

    # ------------------------------------------------------------------
    # Unsupervised mode
    # ------------------------------------------------------------------

    def test_unsupervised_maps_identical_strings_to_themselves(self):
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            "CREATE OR REPLACE TABLE src AS SELECT unnest(['Apple', 'Google', 'Meta']) AS name;",
        )

        result = self.action.apply({
            "source_table_id": "src",
            "target_column": "name",
            "output_mapping_table_id": "name_map",
        })

        self.assertIn("original_value", result.columns)
        self.assertIn("canonical_value", result.columns)

        mapping = self.db_api.execute_query(
            self.user_id, self.chat_id,
            'SELECT original_value, canonical_value FROM "name_map" ORDER BY original_value;',
        )
        self.assertEqual(len(mapping), 3)
        # Each distinct string maps to itself (nothing is similar enough to cluster)
        orig_to_canon = dict(zip(mapping["original_value"], mapping["canonical_value"]))
        self.assertEqual(orig_to_canon["Apple"], "Apple")
        self.assertEqual(orig_to_canon["Google"], "Google")
        self.assertEqual(orig_to_canon["Meta"], "Meta")

    def test_unsupervised_clusters_typo_variants(self):
        # "Amazn" and "Amzon" are very close to "Amazon" — all should cluster together
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            "CREATE OR REPLACE TABLE src AS SELECT unnest(['Amazon', 'Amazn', 'Amzon', 'Google']) AS name;",
        )

        self.action.apply({
            "source_table_id": "src",
            "target_column": "name",
            "output_mapping_table_id": "name_map",
            "threshold": 0.80,
        })

        mapping = self.db_api.execute_query(
            self.user_id, self.chat_id,
            'SELECT original_value, canonical_value FROM "name_map";',
        )
        orig_to_canon = dict(zip(mapping["original_value"], mapping["canonical_value"]))

        # All Amazon variants must share the same canonical form
        amazon_canonicals = {orig_to_canon[k] for k in ["Amazon", "Amazn", "Amzon"]}
        self.assertEqual(len(amazon_canonicals), 1, "Amazon variants should share one canonical form")

        # Google must be separate
        self.assertNotIn(orig_to_canon["Google"], amazon_canonicals)

    def test_unsupervised_shortest_string_becomes_canonical(self):
        # Sort-by-length means "IBM" (3 chars) is processed before "IBM Corp" (8) and "IBM Corp." (9)
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            "CREATE OR REPLACE TABLE src AS SELECT unnest(['IBM Corp', 'IBM Corp.', 'IBM']) AS name;",
        )

        self.action.apply({
            "source_table_id": "src",
            "target_column": "name",
            "output_mapping_table_id": "name_map",
            "threshold": 0.80,
        })

        mapping = self.db_api.execute_query(
            self.user_id, self.chat_id,
            'SELECT original_value, canonical_value FROM "name_map";',
        )
        orig_to_canon = dict(zip(mapping["original_value"], mapping["canonical_value"]))

        # "IBM" is shortest → becomes the representative; the others map to it
        self.assertEqual(orig_to_canon["IBM"], "IBM")
        self.assertEqual(orig_to_canon["IBM Corp"], "IBM")
        self.assertEqual(orig_to_canon["IBM Corp."], "IBM")

    def test_unsupervised_output_table_persisted_in_db(self):
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            "CREATE OR REPLACE TABLE src AS SELECT unnest(['Foo', 'Bar']) AS val;",
        )

        self.action.apply({
            "source_table_id": "src",
            "target_column": "val",
            "output_mapping_table_id": "val_map",
        })

        tables = self.db_api.execute_query(self.user_id, self.chat_id, "SHOW TABLES;")
        self.assertIn("val_map", tables["name"].tolist())

    def test_unsupervised_all_unique_values_covered(self):
        # Every unique non-null value in the source column must appear in the mapping
        values = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            f"CREATE OR REPLACE TABLE src AS SELECT unnest({values!r}) AS name;",
        )

        self.action.apply({
            "source_table_id": "src",
            "target_column": "name",
            "output_mapping_table_id": "name_map",
        })

        mapping = self.db_api.execute_query(
            self.user_id, self.chat_id,
            'SELECT original_value FROM "name_map";',
        )
        self.assertEqual(set(mapping["original_value"].tolist()), set(values))

    def test_unsupervised_empty_table_produces_empty_mapping(self):
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            "CREATE OR REPLACE TABLE src (name VARCHAR);",
        )

        result = self.action.apply({
            "source_table_id": "src",
            "target_column": "name",
            "output_mapping_table_id": "name_map",
        })

        self.assertEqual(len(result), 0)

    def test_unsupervised_nulls_excluded_from_mapping(self):
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            "CREATE OR REPLACE TABLE src AS SELECT unnest(['Apple', NULL, 'Google']) AS name;",
        )

        self.action.apply({
            "source_table_id": "src",
            "target_column": "name",
            "output_mapping_table_id": "name_map",
        })

        mapping = self.db_api.execute_query(
            self.user_id, self.chat_id,
            'SELECT original_value FROM "name_map";',
        )
        # NULL is excluded; only 'Apple' and 'Google' appear
        self.assertEqual(set(mapping["original_value"].tolist()), {"Apple", "Google"})

    def test_unsupervised_idempotent_overwrite(self):
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            "CREATE OR REPLACE TABLE src AS SELECT unnest(['X', 'Y']) AS name;",
        )
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            'CREATE OR REPLACE TABLE "existing_map" AS SELECT \'old\' AS original_value, \'old\' AS canonical_value;',
        )

        self.action.apply({
            "source_table_id": "src",
            "target_column": "name",
            "output_mapping_table_id": "existing_map",
        })

        mapping = self.db_api.execute_query(
            self.user_id, self.chat_id,
            'SELECT original_value FROM "existing_map" ORDER BY original_value;',
        )
        self.assertEqual(set(mapping["original_value"].tolist()), {"X", "Y"})

    # ------------------------------------------------------------------
    # Supervised mode
    # ------------------------------------------------------------------

    def test_supervised_maps_close_strings_to_canonical_entity(self):
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            "CREATE OR REPLACE TABLE src AS SELECT unnest(['Amazn', 'Amaznn', 'Google', 'Gooogle']) AS name;",
        )

        self.action.apply({
            "source_table_id": "src",
            "target_column": "name",
            "output_mapping_table_id": "name_map",
            "canonical_entities": ["Amazon", "Google"],
            "threshold": 0.80,
        })

        mapping = self.db_api.execute_query(
            self.user_id, self.chat_id,
            'SELECT original_value, canonical_value FROM "name_map";',
        )
        orig_to_canon = dict(zip(mapping["original_value"], mapping["canonical_value"]))

        self.assertEqual(orig_to_canon["Amazn"], "Amazon")
        self.assertEqual(orig_to_canon["Amaznn"], "Amazon")
        self.assertEqual(orig_to_canon["Google"], "Google")
        self.assertEqual(orig_to_canon["Gooogle"], "Google")

    def test_supervised_unmatched_values_get_sentinel(self):
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            "CREATE OR REPLACE TABLE src AS SELECT unnest(['Amazon', 'Zyxwvuts']) AS name;",
        )

        self.action.apply({
            "source_table_id": "src",
            "target_column": "name",
            "output_mapping_table_id": "name_map",
            "canonical_entities": ["Google", "Meta"],
            "threshold": 0.85,
        })

        mapping = self.db_api.execute_query(
            self.user_id, self.chat_id,
            'SELECT original_value, canonical_value FROM "name_map";',
        )
        orig_to_canon = dict(zip(mapping["original_value"], mapping["canonical_value"]))

        # Neither value resembles Google or Meta → both get the sentinel
        self.assertEqual(orig_to_canon["Amazon"], _UNRESOLVED_SENTINEL)
        self.assertEqual(orig_to_canon["Zyxwvuts"], _UNRESOLVED_SENTINEL)

    def test_supervised_all_unique_values_covered(self):
        values = ["Apple Inc", "Appel", "Micro Soft", "Zara"]
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            f"CREATE OR REPLACE TABLE src AS SELECT unnest({values!r}) AS name;",
        )

        self.action.apply({
            "source_table_id": "src",
            "target_column": "name",
            "output_mapping_table_id": "name_map",
            "canonical_entities": ["Apple", "Microsoft"],
            "threshold": 0.80,
        })

        mapping = self.db_api.execute_query(
            self.user_id, self.chat_id,
            'SELECT original_value FROM "name_map";',
        )
        self.assertEqual(set(mapping["original_value"].tolist()), set(values))

    def test_supervised_uses_config_threshold_when_not_provided(self):
        # With a very high default threshold (1.0), nothing matches
        self.config.ENTITY_RESOLUTION_THRESHOLD = 1.0
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            "CREATE OR REPLACE TABLE src AS SELECT 'Amazon' AS name;",
        )

        self.action.apply({
            "source_table_id": "src",
            "target_column": "name",
            "output_mapping_table_id": "name_map",
            "canonical_entities": ["Amazn"],  # close but not identical
        })

        mapping = self.db_api.execute_query(
            self.user_id, self.chat_id,
            'SELECT canonical_value FROM "name_map";',
        )
        self.assertEqual(mapping.iloc[0]["canonical_value"], _UNRESOLVED_SENTINEL)

    # ------------------------------------------------------------------
    # Return value
    # ------------------------------------------------------------------

    def test_apply_returns_dataframe_with_correct_columns(self):
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            "CREATE OR REPLACE TABLE src AS SELECT unnest(['A', 'B', 'C']) AS name;",
        )

        result = self.action.apply({
            "source_table_id": "src",
            "target_column": "name",
            "output_mapping_table_id": "name_map",
        })

        self.assertIn("original_value", result.columns)
        self.assertIn("canonical_value", result.columns)

    def test_apply_returns_at_most_5_rows(self):
        # Source has 10 distinct values; return value is capped at 5 (LIMIT 5)
        values = [f"Entity{i}" for i in range(10)]
        self.db_api.execute_query(
            self.user_id, self.chat_id,
            f"CREATE OR REPLACE TABLE src AS SELECT unnest({values!r}) AS name;",
        )

        result = self.action.apply({
            "source_table_id": "src",
            "target_column": "name",
            "output_mapping_table_id": "name_map",
        })

        self.assertLessEqual(len(result), 5)

    # ------------------------------------------------------------------
    # get_description
    # ------------------------------------------------------------------

    def test_get_description_mentions_action_name(self):
        from pneuma_seeker.shared.schemas.core.action import ActionNames
        desc = self.action.get_description()
        self.assertIn(ActionNames.ENTITY_RESOLUTION.value, desc)

    def test_get_description_mentions_both_modes(self):
        desc = self.action.get_description()
        self.assertIn("Unsupervised", desc)
        self.assertIn("Supervised", desc)


if __name__ == "__main__":
    unittest.main()
