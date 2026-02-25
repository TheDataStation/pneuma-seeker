# tests/pneuma_seeker/shared/test_parser.py
import os
import unittest
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)
from pneuma_seeker.shared import parser


class ParserTests(unittest.TestCase):
    def test_parse_json_valid(self):
        js = '{"a":1, "b":2}'
        self.assertEqual(parser.parse_json(js), {"a": 1, "b": 2})

    def test_parse_json_with_code_block(self):
        js = '```json\n{"a":1}```'
        self.assertEqual(parser.parse_json(js), {"a": 1})

    def test_parse_json_with_prefix_suffix_text(self):
        js = 'Sure. {"a": 1, "b": 2} Thanks.'
        self.assertEqual(parser.parse_json(js), {"a": 1, "b": 2})

    def test_parse_json_invalid_type(self):
        with self.assertRaises(ValueError):
            parser.parse_json(123)  # type: ignore

    def test_parse_json_invalid_json(self):
        with self.assertRaises(ValueError):
            parser.parse_json("{invalid_json}")

    def test_parse_sql_and_code(self):
        self.assertEqual(parser.parse_sql("```sql SELECT 1```"), " SELECT 1")
        self.assertEqual(parser.parse_code("```python x=1```"), " x=1")

    def test_augmented_literal_eval(self):
        self.assertEqual(parser.augmented_literal_eval("```python [1,2]```"), [1, 2])
        with self.assertRaises(SyntaxError):
            parser.augmented_literal_eval("```python [1,2```")  # invalid python


if __name__ == "__main__":
    unittest.main()
