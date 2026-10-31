# tests/pneuma_seeker/provenance/test_text_usage.py
import os
import sys
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from pneuma_seeker.provenance.text_usage import find_transcribed_values, uses_document

WEB_TEXT = (
    "The largest French cities are Paris (population 2,140,526) and "
    "Marseille (population 873,076). Lyon follows with 522,969 inhabitants."
)


class TextUsageTests(unittest.TestCase):
    def test_detects_values_transcribed_into_sql_inside_python(self):
        code = (
            'db_api.execute_query(user_id, chat_id, "CREATE OR REPLACE TABLE t AS '
            "SELECT * FROM (VALUES ('Paris', 2140526), ('Marseille', 873076)) v(city, population)\")"
        )
        self.assertEqual(
            uses_document(code, WEB_TEXT),
            ["paris", "marseille", "2140526", "873076"],
        )

    def test_numbers_match_regardless_of_formatting(self):
        self.assertEqual(
            find_transcribed_values("x = [522969.0, 873076]", WEB_TEXT),
            ["522969", "873076"],
        )

    def test_single_coincidental_value_is_not_usage(self):
        self.assertEqual(uses_document("SELECT * FROM x WHERE city = 'Paris'", WEB_TEXT), [])

    def test_ignores_identifiers_small_round_numbers_and_partial_words(self):
        code = "SELECT \"population\", 'Par' FROM x WHERE n > 1000 LIMIT 100 -- 'Lyon follows'"
        self.assertEqual(
            find_transcribed_values(code, WEB_TEXT, identifiers={"population"}),
            ["lyon follows"],
        )
        self.assertEqual(find_transcribed_values("'arse'", WEB_TEXT), [])

    def test_empty_inputs(self):
        self.assertEqual(uses_document("", WEB_TEXT), [])
        self.assertEqual(uses_document("'Paris', 'Lyon'", ""), [])


if __name__ == "__main__":
    unittest.main()
