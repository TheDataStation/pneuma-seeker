from abc import ABC
from enum import Enum
from typing import Any

import json

from pandas import DataFrame


class RetrieverType(Enum):
    """
    Represents all types of data in Pneuma-Seeker.
    """

    PNEUMA_RETRIEVER = "Pneuma"  # Internal tables
    CONDUCTOR = "Conductor"
    ENUMERATOR = "Enumerator"
    MATERIALIZER = "Materializer"
    DOCUMENT_DB = "Document DB"
    WEB_SEARCH = "Web Search"
    USER = "User"  # External tables
    WEB_CRAWL = "Web Crawl"


class AbstractDocument(ABC):
    """
    Represents (abstractly) the unit of information in Processor.
    """

    def __init__(
        self,
        doc_id: str,
        retriever_type: RetrieverType,
        content: Any,
        metadata: dict[str, str],
        path: str | None = None,
        last_node_id: (
            str | None
        ) = None,  # Keep track of last transformation that returns this data
    ):
        self.doc_id = doc_id
        self.retriever_type = retriever_type
        self.content = content
        self.metadata = metadata
        self.path = path
        self.last_node_id = last_node_id

    def __eq__(self, other):
        if not isinstance(other, AbstractDocument):
            return NotImplemented
        return (self.doc_id, self.retriever_type) == (
            other.doc_id,
            other.retriever_type,
        )

    def __hash__(self):
        return hash((self.doc_id, self.retriever_type))

    def __str__(self) -> str:
        return f"ID: {self.doc_id} ; Content: {self.content}"


class Knowledge(AbstractDocument):
    """
    Represents some form of knowledge from users.

    - retriever_type: RetrieverType.KNOWLEDGE_BASE
    - content: str
    - metadata: {"type": "local/global", "user": "..."}
    """

    def __init__(
        self,
        doc_id: str,
        retriever_type: RetrieverType,
        content: str,
        metadata: dict[str, str],
        path: str | None = None,
        last_node_id: str | None = None,
    ):
        super().__init__(doc_id, retriever_type, content, metadata, path, last_node_id)


class Table(AbstractDocument):
    """
    Represents a table.

    - retriever_type: RetrieverType.PNEUMA (Pneuma is the current table discovery system)
    - content: DataFrame
    - metadata: {"table_name": "...", "dataset_name": "..."}
    """

    def __init__(
        self,
        doc_id: str,
        retriever_type: RetrieverType,
        content: DataFrame,
        metadata: dict[str, str],
        path: str | None = None,
        last_node_id: str | None = None,
    ):
        super().__init__(doc_id, retriever_type, content, metadata, path, last_node_id)

    def __str__(self) -> str:
        table: DataFrame = self.content
        table_id = self.doc_id
        if "dataset_name" in self.metadata:
            table_id += f" (dataset: {self.metadata['dataset_name']})"

        duckdb_col_types: dict[str, str] = {}
        col_types_raw = self.metadata.get("column_types")
        if isinstance(col_types_raw, str) and col_types_raw.strip():
            try:
                parsed = json.loads(col_types_raw)
                if isinstance(parsed, dict):
                    duckdb_col_types = {
                        str(k): str(v) for k, v in parsed.items() if v is not None
                    }
            except Exception:
                duckdb_col_types = {}

        cols = " | ".join(
            f"{col} ({duckdb_col_types.get(col, dtype)})"
            for col, dtype in zip(table.columns, table.dtypes)
        )

        if "description" in self.metadata and "keywords_existence" in self.metadata:
            header = f"Table {table_id} ({self.metadata['description']}; include these keywords: {self.metadata['keywords_existence']}):\ncol: {cols}"
        elif "description" in self.metadata:
            header = f"Table {table_id} ({self.metadata['description']}):\ncol: {cols}"
        elif "keywords_existence" in self.metadata:
            header = (
                f"Table {table_id} "
                f"(include these keywords: {self.metadata['keywords_existence']}):\n"
                f"col: {cols}"
            )
        else:
            header = f"Table {table_id}:\ncol: {cols}"

        lines = [header]

        if len(table) > 0:
            sample_rows = table.sample(min(5, len(table)), random_state=42).sort_index()
            for idx, (_, row) in enumerate(sample_rows.iterrows(), start=1):
                row_str = " | ".join(str(row[col]) for col in table.columns)
                lines.append(f"sample row {idx}: {row_str}")

        return "\n".join(lines)


class TableContext(AbstractDocument):
    """
    Represents table context.

    - retriever_type: RetrieverType.PNEUMA (Pneuma is the current table discovery system)
    - content: str
    - metadata: {"table_name": "...", "dataset_name": "...", "type": "..."}
    """

    def __init__(
        self,
        doc_id: str,
        retriever_type: RetrieverType,
        content: str,
        metadata: dict[str, str],
        path: str | None = None,
        last_node_id: str | None = None,
    ):
        super().__init__(doc_id, retriever_type, content, metadata, path, last_node_id)


class Text(AbstractDocument):
    """
    Represents textual document.
    """

    def __init__(
        self,
        doc_id: str,
        retriever_type: RetrieverType,
        content: str,
        metadata: dict[str, str],
        path: str | None = None,
        last_node_id: str | None = None,
    ):
        super().__init__(doc_id, retriever_type, content, metadata, path, last_node_id)


def convert_retrieval_results_to_str(retrieval_results: list[AbstractDocument]):
    representation = ""
    seen_docs: set[str] = set()
    topic_documents: dict[str, list[AbstractDocument]] = {}
    for result in retrieval_results:
        topic = result.metadata.get("topic", "unknown")
        if topic not in topic_documents:
            topic_documents[topic] = []
        topic_documents[topic].append(result)

    for topic, docs in topic_documents.items():
        representation += f"- Topic: {topic}\n"
        for doc in docs:
            if doc.doc_id in seen_docs:
                representation += f"  - Table {doc.doc_id}\n"
            else:
                representation += f"  - ```{str(doc)}```\n"
                seen_docs.add(doc.doc_id)
    return representation.strip()
