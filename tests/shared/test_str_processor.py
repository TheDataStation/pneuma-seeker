# tests/pneuma_seeker/shared/test_str_processor.py
import os
import unittest
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import pandas as pd

from pneuma_seeker.shared.str_processor import clean_column_table_name, dataframe_to_preview_str

class StrProcessorTests(unittest.TestCase):
    def test_basic_cleaning(self):
        self.assertEqual(clean_column_table_name("My Column"), "my_column")
        self.assertEqual(clean_column_table_name("Column-Name"), "column_name")
        self.assertEqual(clean_column_table_name("Col(1)"), "col_1")
        self.assertEqual(clean_column_table_name("Col$#@!"), "colusdnum")

    def test_multiple_underscores(self):
        self.assertEqual(clean_column_table_name("A__B--C"), "a_b_c")

    def test_leading_trailing_underscores(self):
        self.assertEqual(clean_column_table_name("_A B_"), "a_b")

    def test_empty_string(self):
        self.assertEqual(clean_column_table_name(""), "")


class PreviewTests(unittest.TestCase):
    def test_a_small_result_shows_every_cell_without_padding(self):
        df = pd.DataFrame({"order_id": ["order_1", "order_22"], "rating": [4.5, None], "n": [3, 10]})
        self.assertEqual(dataframe_to_preview_str(df), "order_id | rating | n\norder_1 | 4.5 | 3\norder_22 | NULL | 10")

    def test_a_long_cell_is_cut_and_says_how_much(self):
        df = pd.DataFrame({"total": [200], "notes": ["x" * 50_000]})
        text = dataframe_to_preview_str(df)
        self.assertIn("200 | " + "x" * 300 + "… (+49,700 chars)", text)
        self.assertLess(len(text), 400)

    def test_a_long_result_is_cut_at_a_row_and_says_how_much(self):
        df = pd.DataFrame({"id": range(1000), "text": ["y" * 100] * 1000})
        text = dataframe_to_preview_str(df, max_chars=2_000)
        rows = text.splitlines()
        self.assertLessEqual(len("\n".join(rows[:-1])), 2_000)
        self.assertRegex(rows[-1], r"^… \(showing \d+ of 1000 rows; the result is [\d,]+ characters\.")
        self.assertTrue(rows[-2].startswith(str(len(rows) - 3) + " | "))  # whole rows only

    def test_one_row_too_long_for_the_result_is_still_shown_cut(self):
        df = pd.DataFrame({f"c{i}": ["z" * 300] for i in range(100)})
        text = dataframe_to_preview_str(df, max_chars=1_000)
        self.assertIn("showing 1 of 1 rows", text)
        self.assertLess(len(text), 1_200)


if __name__ == "__main__":
    unittest.main()
