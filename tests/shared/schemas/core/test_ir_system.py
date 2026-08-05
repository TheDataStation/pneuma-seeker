import os
import sys
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
)

import pandas as pd

from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
    TableContext,
    Text,
    convert_retrieval_results_to_str,
)


class IRSystemSchemaTests(unittest.TestCase):
    """Unit tests for IR System schema utilities."""

    def test_convert_retrieval_results_to_str_empty_list(self):
        """Test converting empty list returns empty string."""
        result = convert_retrieval_results_to_str([])
        self.assertEqual(result, "")

    def test_convert_retrieval_results_to_str_single_text_document(self):
        """Test converting single Text document in normal mode."""
        doc = Text(
            doc_id="doc1",
            retriever_type=RetrieverType.ENUMERATOR,
            content="Test content",
            metadata={"source": "test"},
        )
        result = convert_retrieval_results_to_str([doc])
        self.assertIn("doc1", result)
        self.assertIn("Test content", result)
        self.assertIn("- ```", result)

    def test_convert_retrieval_results_to_str_multi_topic_mode_single_topic(self):
        """Test converting documents with single topic in multi-topic mode."""
        docs: list[AbstractDocument] = [
            Text(
                doc_id="doc1",
                retriever_type=RetrieverType.WEB_SEARCH,
                content="Content 1",
                metadata={"source": "web", "topic": "python"},
            ),
            Text(
                doc_id="doc2",
                retriever_type=RetrieverType.WEB_SEARCH,
                content="Content 2",
                metadata={"source": "web", "topic": "python"},
            ),
        ]
        result = convert_retrieval_results_to_str(docs)
        self.assertIn("- Topic: python", result)
        self.assertIn("doc1", result)
        self.assertIn("doc2", result)
        # Should have one topic header
        self.assertEqual(result.count("- Topic:"), 1)

    def test_convert_retrieval_results_to_str_multi_topic_mode_multiple_topics(self):
        """Test converting documents with multiple topics in multi-topic mode."""
        docs: list[AbstractDocument] = [
            Text(
                doc_id="doc1",
                retriever_type=RetrieverType.WEB_SEARCH,
                content="Python content",
                metadata={"topic": "python"},
            ),
            Text(
                doc_id="doc2",
                retriever_type=RetrieverType.WEB_SEARCH,
                content="Java content",
                metadata={"topic": "java"},
            ),
            Text(
                doc_id="doc3",
                retriever_type=RetrieverType.WEB_SEARCH,
                content="More Python",
                metadata={"topic": "python"},
            ),
        ]
        result = convert_retrieval_results_to_str(docs)
        self.assertIn("- Topic: python", result)
        self.assertIn("- Topic: java", result)
        self.assertIn("doc1", result)
        self.assertIn("doc2", result)
        self.assertIn("doc3", result)
        # Should have two topic headers
        self.assertEqual(result.count("- Topic:"), 2)

    def test_convert_retrieval_results_to_str_multi_topic_mode_missing_topic(self):
        """Test documents without topic metadata default to 'unknown' in multi-topic mode."""
        docs: list[AbstractDocument] = [
            Text(
                doc_id="doc1",
                retriever_type=RetrieverType.ENUMERATOR,
                content="Content without topic",
                metadata={"source": "test"},
            ),
        ]
        result = convert_retrieval_results_to_str(docs)
        self.assertIn("- Topic: unknown", result)
        self.assertIn("doc1", result)

    def test_convert_retrieval_results_to_str_with_table_document(self):
        """Test converting Table document."""
        table = Table(
            doc_id="table1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            metadata={"table_name": "test_table", "dataset_name": "test_db"},
        )
        result = convert_retrieval_results_to_str([table])
        self.assertIn("table1", result)
        self.assertIn("a", result)
        self.assertIn("b", result)
        # Table __str__ should include column names
        self.assertIn("col:", result)

    def test_convert_retrieval_results_to_str_with_table_context_document(self):
        """Test converting TableContext document."""
        context = TableContext(
            doc_id="context1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content="This table contains user data",
            metadata={"table_name": "users", "dataset_name": "db", "type": "description"},
        )
        result = convert_retrieval_results_to_str([context])
        self.assertIn("context1", result)
        self.assertIn("This table contains user data", result)

    def test_convert_retrieval_results_to_str_with_user_provided_document(self):
        """Test converting a user-provided (external) document."""
        doc = Text(
            doc_id="ext1",
            retriever_type=RetrieverType.USER,
            content="Important knowledge",
            metadata={"source": "user"},
        )
        result = convert_retrieval_results_to_str([doc])
        self.assertIn("ext1", result)
        self.assertIn("Important knowledge", result)

    def test_convert_retrieval_results_to_str_mixed_documents_multi_topic(self):
        """Test converting mixed document types in multi-topic mode."""
        docs = [
            Text(
                doc_id="text1",
                retriever_type=RetrieverType.WEB_SEARCH,
                content="Web content",
                metadata={"topic": "web"},
            ),
            Table(
                doc_id="table1",
                retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                content=pd.DataFrame({"x": [1, 2]}),
                metadata={"topic": "database", "table_name": "t1", "dataset_name": "d1"},
            ),
            TableContext(
                doc_id="ctx1",
                retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                content="Context info",
                metadata={"topic": "web", "table_name": "t2", "dataset_name": "d2", "type": "desc"},
            ),
        ]
        result = convert_retrieval_results_to_str(docs)
        self.assertIn("- Topic: web", result)
        self.assertIn("- Topic: database", result)
        self.assertIn("text1", result)
        self.assertIn("table1", result)
        self.assertIn("ctx1", result)

    def test_convert_retrieval_results_to_str_empty_table(self):
        """Test converting Table with no rows."""
        table = Table(
            doc_id="empty_table",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=pd.DataFrame({"a": [], "b": []}),
            metadata={"table_name": "empty", "dataset_name": "test"},
        )
        result = convert_retrieval_results_to_str([table])
        self.assertIn("empty_table", result)
        self.assertIn("col:", result)
        # Should not have sample rows
        self.assertNotIn("sample row", result)

    def test_abstract_document_equality(self):
        """Test AbstractDocument equality based on doc_id and retriever_type."""
        doc1 = Text(
            doc_id="doc1",
            retriever_type=RetrieverType.ENUMERATOR,
            content="Content",
            metadata={},
        )
        doc2 = Text(
            doc_id="doc1",
            retriever_type=RetrieverType.ENUMERATOR,
            content="Different content",
            metadata={},
        )
        doc3 = Text(
            doc_id="doc1",
            retriever_type=RetrieverType.WEB_SEARCH,
            content="Content",
            metadata={},
        )
        
        self.assertEqual(doc1, doc2)  # Same id and retriever_type
        self.assertNotEqual(doc1, doc3)  # Different retriever_type

    def test_abstract_document_hash(self):
        """Test AbstractDocument can be used in sets/dicts."""
        doc1 = Text(
            doc_id="doc1",
            retriever_type=RetrieverType.ENUMERATOR,
            content="Content",
            metadata={},
        )
        doc2 = Text(
            doc_id="doc1",
            retriever_type=RetrieverType.ENUMERATOR,
            content="Different content",
            metadata={},
        )
        
        doc_set = {doc1, doc2}
        self.assertEqual(len(doc_set), 1)  # Should deduplicate

    def test_table_str_with_many_rows(self):
        """Test Table __str__ (default 'one_row' strategy) shows exactly 1 sample row."""
        table = Table(
            doc_id="big_table",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=pd.DataFrame({"a": range(100), "b": range(100, 200)}),
            metadata={"table_name": "big", "dataset_name": "test"},
        )
        result = str(table)
        self.assertEqual(result.count("sample row"), 1)

    def _mixed_format_table(self) -> Table:
        """Table with a categorical column mixing majority '#'-shaped ranks and
        one rare compound-shaped value, plus a numeric and a date column."""
        return Table(
            doc_id="mixed_table",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=pd.DataFrame(
                {
                    "rank": ["1", "1", "2", "3", "4 or 5"],
                    "price": [9.99, 19.99, 29.99, 39.99, 249.00],
                    "signup_date": [
                        "2019-01-04",
                        "2020-05-01",
                        "2021-07-15",
                        "2022-03-09",
                        "2024-11-30",
                    ],
                }
            ),
            metadata={"table_name": "mixed", "dataset_name": "test"},
        )

    def test_table_to_str_column_profile_strategy(self):
        """column_profile shows no sample row, and surfaces the rare rank format."""
        table = self._mixed_format_table()
        result = table.to_str(strategy="column_profile")
        self.assertNotIn("sample row", result)
        self.assertIn("etc.", result)
        self.assertIn("distinct", result)
        self.assertIn("4 or 5", result)
        self.assertIn("range", result)

    def test_table_to_str_one_row_strategy(self):
        """one_row does simple random sampling (random_state=42) of exactly 1 row."""
        table = self._mixed_format_table()
        result = table.to_str(strategy="one_row")
        self.assertEqual(result.count("sample row"), 1)

    def test_table_to_str_five_rows_strategy(self):
        """five_rows does simple random sampling (random_state=42) of up to 5 rows."""
        table = self._mixed_format_table()
        result = table.to_str(strategy="five_rows")
        self.assertEqual(result.count("sample row"), 5)

    def test_table_to_str_zero_rows_strategy(self):
        """zero_rows shows no rows, and inlines an example value in the header
        for non-numeric columns only."""
        table = self._mixed_format_table()
        result = table.to_str(strategy="zero_rows")
        self.assertNotIn("sample row", result)
        self.assertIn("rank (object; e.g., 1)", result)
        self.assertIn("signup_date (object; e.g., 2019-01-04)", result)
        self.assertIn("price (float64)", result)
        self.assertNotIn("price (float64; e.g.,", result)

    def test_table_to_str_five_rows_strategy_capped_by_table_size(self):
        """five_rows never samples more rows than the table has."""
        table = Table(
            doc_id="small_table",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=pd.DataFrame({"a": [1, 2]}),
            metadata={"table_name": "small", "dataset_name": "test"},
        )
        result = table.to_str(strategy="five_rows")
        self.assertEqual(result.count("sample row"), 2)
