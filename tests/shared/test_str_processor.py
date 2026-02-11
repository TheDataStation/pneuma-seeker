# tests/pneuma_seeker/shared/test_str_processor.py
import os
import unittest
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from pneuma_seeker.shared.str_processor import clean_column_table_name

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

if __name__ == "__main__":
    unittest.main()
