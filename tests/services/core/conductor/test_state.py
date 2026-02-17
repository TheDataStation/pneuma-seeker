# tests/pneuma_seeker/core/conductor/test_state.py
import os
import sys
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
)

import pandas as pd

from pneuma_seeker.services.core.conductor.state import ConductorState
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType, Table


class InformationNeedStateTests(unittest.TestCase):
    def setUp(self):
        self.state = ConductorState()
        self.config = Config("../../../.env.test")

    def test_initial_state(self):
        self.assertEqual(self.state.T, {})
        self.assertFalse(self.state.is_T_materialized)
        self.assertEqual(self.state.column_descriptions, {})
        self.assertEqual(self.state.S, "")
        self.assertFalse(self.state.is_S_executed)

    def test_add_table_and_get_current_state(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        table = Table("tbl1", RetrieverType.USER, df, {}, path="dummy.tbl1")
        self.state.T["tbl1"] = table
        self.state.is_T_materialized = True
        self.state.column_descriptions["tbl1"] = {"a": "col a", "b": "col b"}
        self.state.S = "SELECT *"
        self.state.is_S_executed = True

        snapshot = self.state.get_current_state_instance(
            self.config.TABLE_MAX_ROWS_DISPLAY
        )
        self.assertIn("tbl1", snapshot["T"])
        self.assertEqual(snapshot["is_T_materialized"], True)
        self.assertEqual(snapshot["column_descriptions"]["tbl1"]["a"], "col a")
        self.assertEqual(snapshot["S"], "SELECT *")
        self.assertTrue(snapshot["is_S_executed"])

    def test_str_representation(self):
        df = pd.DataFrame({"x": [10]})
        table = Table("tblX", RetrieverType.USER, df, {}, path="dummy.tblX")
        self.state.T["tblX"] = table
        s = str(self.state)
        self.assertIn("Target tables", s)
        self.assertIn("tblX", s)


if __name__ == "__main__":
    unittest.main()
