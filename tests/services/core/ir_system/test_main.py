import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
)

import pandas as pd

from pneuma_seeker.services.core.ir_system.main import Retriever
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
    TableContext,
    Text,
)


class IRSystemTests(unittest.TestCase):
    """Unit tests for IRSystem."""

    def setUp(self) -> None:
        self.config = Config()
        self.logger = MagicMock()
        self.db_api = MagicMock()
        self.lm_api = MagicMock()
        self.ir_system = Retriever(
            "uX", "cX", self.config, self.logger, self.db_api, self.lm_api
        )

    def test_index_documents_calls_retriever_index(self):
        """Test that index_documents calls the retriever's index method."""
        mock_retriever = MagicMock()
        self.ir_system.retriever_factory.get_retriever = MagicMock(
            return_value=mock_retriever
        )

        documents: list[AbstractDocument] = [
            Text(
                doc_id="doc1",
                retriever_type=RetrieverType.DOCUMENT_DB,
                content="Test content",
                metadata={"source": "test"},
            )
        ]

        self.ir_system.index_documents(RetrieverType.DOCUMENT_DB, documents)

        self.ir_system.retriever_factory.get_retriever.assert_called_once_with(
            RetrieverType.DOCUMENT_DB
        )
        mock_retriever.index.assert_called_once_with(documents)

    def test_retrieve_documents_returns_documents(self):
        """Test that retrieve_documents returns documents from retriever."""
        mock_documents = [
            Text(
                doc_id="doc1",
                retriever_type=RetrieverType.DOCUMENT_DB,
                content="Test content 1",
                metadata={"source": "test"},
            ),
            Text(
                doc_id="doc2",
                retriever_type=RetrieverType.DOCUMENT_DB,
                content="Test content 2",
                metadata={"source": "test"},
            ),
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve = MagicMock(return_value=mock_documents)
        self.ir_system.retriever_factory.get_retriever = MagicMock(
            return_value=mock_retriever
        )

        result = self.ir_system.retrieve_documents(
            RetrieverType.DOCUMENT_DB, "test query", k=10
        )

        self.assertEqual(result, mock_documents)
        self.assertEqual(len(result), 2)
        mock_retriever.retrieve.assert_called_once_with(
            "test query", 10, False, None
        )

    def test_retrieve_documents_with_sample_only(self):
        """Test retrieve_documents with sample_only flag."""
        mock_documents = [
            Table(
                doc_id="table1",
                retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                content=pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
                metadata={"table_name": "table1", "dataset_name": "test_dataset"},
            )
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve = MagicMock(return_value=mock_documents)
        self.ir_system.retriever_factory.get_retriever = MagicMock(
            return_value=mock_retriever
        )

        result = self.ir_system.retrieve_documents(
            RetrieverType.PNEUMA_RETRIEVER,
            "test query",
            k=5,
            sample_only=True,
            sample_size=100,
        )

        self.assertEqual(result, mock_documents)
        mock_retriever.retrieve.assert_called_once_with(
            "test query", 5, True, 100
        )

    def test_retrieve_multi_topic_documents_returns_dict(self):
        """Test that retrieve_multi_topic_documents returns a dictionary of results."""
        mock_documents_1 = [
            Text(
                doc_id="doc1",
                retriever_type=RetrieverType.WEB_SEARCH,
                content="Content about topic 1",
                metadata={"source": "web"},
            )
        ]
        mock_documents_2 = [
            Text(
                doc_id="doc2",
                retriever_type=RetrieverType.WEB_SEARCH,
                content="Content about topic 2",
                metadata={"source": "web"},
            )
        ]

        def retrieve_side_effect(prompt, k, sample_only, sample_size):
            if "topic 1" in prompt:
                return mock_documents_1
            elif "topic 2" in prompt:
                return mock_documents_2
            return []

        mock_retriever = MagicMock()
        mock_retriever.retrieve = MagicMock(side_effect=retrieve_side_effect)
        self.ir_system.retriever_factory.get_retriever = MagicMock(
            return_value=mock_retriever
        )

        prompts = ["query about topic 1", "query about topic 2"]
        result = self.ir_system.retrieve_multi_topic_documents(
            RetrieverType.WEB_SEARCH, prompts, k=5
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertIn("query about topic 1", result)
        self.assertIn("query about topic 2", result)
        self.assertEqual(result["query about topic 1"], mock_documents_1)
        self.assertEqual(result["query about topic 2"], mock_documents_2)
        self.assertEqual(mock_retriever.retrieve.call_count, 2)

    def test_retrieve_multi_topic_documents_with_empty_prompts(self):
        """Test retrieve_multi_topic_documents with empty prompts list."""
        mock_retriever = MagicMock()
        self.ir_system.retriever_factory.get_retriever = MagicMock(
            return_value=mock_retriever
        )

        result = self.ir_system.retrieve_multi_topic_documents(
            RetrieverType.DOCUMENT_DB, [], k=10
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)
        mock_retriever.retrieve.assert_not_called()

    def test_retrieve_documents_with_table_context(self):
        """Test retrieving TableContext documents."""
        mock_documents = [
            TableContext(
                doc_id="context1",
                retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                content="This table contains user information",
                metadata={
                    "table_name": "users",
                    "dataset_name": "test_db",
                    "type": "description",
                },
            )
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve = MagicMock(return_value=mock_documents)
        self.ir_system.retriever_factory.get_retriever = MagicMock(
            return_value=mock_retriever
        )

        result = self.ir_system.retrieve_documents(
            RetrieverType.PNEUMA_RETRIEVER, "user information", k=5
        )

        self.assertEqual(result, mock_documents)
        self.assertIsInstance(result[0], TableContext)

    def test_retrieve_documents_returns_empty_list(self):
        """Test that retrieve_documents returns empty list when no documents found."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = MagicMock(return_value=[])
        self.ir_system.retriever_factory.get_retriever = MagicMock(
            return_value=mock_retriever
        )

        result = self.ir_system.retrieve_documents(
            RetrieverType.WEB_SEARCH, "nonexistent query", k=10
        )

        self.assertEqual(result, [])
        self.assertEqual(len(result), 0)

    def test_index_documents_with_table_documents(self):
        """Test indexing Table documents."""
        mock_retriever = MagicMock()
        self.ir_system.retriever_factory.get_retriever = MagicMock(
            return_value=mock_retriever
        )

        documents: list[AbstractDocument] = [
            Table(
                doc_id="table1",
                retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                content=pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}),
                metadata={"table_name": "users", "dataset_name": "test_db"},
            ),
            Table(
                doc_id="table2",
                retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                content=pd.DataFrame({"order_id": [101, 102], "amount": [100, 200]}),
                metadata={"table_name": "orders", "dataset_name": "test_db"},
            ),
        ]

        self.ir_system.index_documents(RetrieverType.PNEUMA_RETRIEVER, documents)

        mock_retriever.index.assert_called_once_with(documents)

    def test_retrieve_multi_topic_documents_with_sample_params(self):
        """Test retrieve_multi_topic_documents with sample parameters."""
        mock_documents = [
            Text(
                doc_id="doc1",
                retriever_type=RetrieverType.WEB_CRAWL,
                content="Crawled content",
                metadata={"url": "http://example.com"},
            )
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve = MagicMock(return_value=mock_documents)
        self.ir_system.retriever_factory.get_retriever = MagicMock(
            return_value=mock_retriever
        )

        prompts = ["query1", "query2"]
        result = self.ir_system.retrieve_multi_topic_documents(
            RetrieverType.WEB_CRAWL,
            prompts,
            k=3,
            sample_only=True,
            sample_size=50,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(mock_retriever.retrieve.call_count, 2)
        # Verify each call had the correct parameters
        for call in mock_retriever.retrieve.call_args_list:
            args, kwargs = call
            self.assertEqual(args[1], 3)  # k
            self.assertEqual(args[2], True)  # sample_only
            self.assertEqual(args[3], 50)  # sample_size
