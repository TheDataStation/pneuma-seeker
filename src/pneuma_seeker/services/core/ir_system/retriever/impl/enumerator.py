import re

import json

from pneuma_seeker.services.core.ir_system.retriever.interface import (
    AbstractRetriever,
)
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
)


def _normalize_table_name(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="replace")
    return str(value)


class Enumerator(AbstractRetriever):
    """Represents a table enumerator."""

    @property
    def retriever_type(self) -> RetrieverType:
        """
        Defines the type of the retriever.
        """
        return RetrieverType.ENUMERATOR

    def load(self):
        """
        Loads the retriever, including its dependencies (e.g., its model).
        """
        pass

    def retrieve(
        self,
        query: str,
        k: int,
        sample_only: bool,
        sample_size: int | None = None,
    ) -> list[AbstractDocument]:
        """
        Retrieves a list of documents given a query, where the query is a regex pattern.
        """
        results: list[AbstractDocument] = []
        self.db_api.link_dataset_tables(
            self.user_id, self.chat_id, self.config.DATA_SOURCES[0]
        )
        all_table_names = self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f"SHOW TABLES FROM {self.config.DATA_SOURCES[0]}",
        )["name"].values.tolist()
        normalized_table_names: list[str] = []
        for raw_name in all_table_names:
            normalized = _normalize_table_name(raw_name)
            if normalized is not None:
                normalized_table_names.append(normalized)

        regex: re.Pattern[str] = re.compile(query)
        match_table_names: list[str] = [
            table_name
            for table_name in normalized_table_names
            if regex.match(table_name) is not None
        ]

        for table_name in match_table_names:
            query_table = f"""
            SELECT * FROM {self.config.DATA_SOURCES[0]}."{table_name}"
            """
            if sample_only:
                if sample_size is None or sample_size <= 0:
                    sample_size = 5
                query_table += f" LIMIT {sample_size}"
            actual_table = self.db_api.execute_query(
                self.user_id, self.chat_id, query_table
            )

            # Fetch authoritative column types from DuckDB.
            duckdb_col_types: dict[str, str] = {}
            try:
                col_types_df = self.db_api.execute_query(
                    self.user_id,
                    self.chat_id,
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                                        WHERE table_name = ?
                                            AND (table_schema = ? OR table_catalog = ?)
                    ORDER BY ordinal_position
                    """.strip(),
                    (
                        table_name,
                        self.config.DATA_SOURCES[0],
                        self.config.DATA_SOURCES[0],
                    ),
                )
                duckdb_col_types = {
                    str(col): str(dtype)
                    for col, dtype in zip(
                        col_types_df["column_name"].tolist(),
                        col_types_df["data_type"].tolist(),
                    )
                }
            except Exception:
                duckdb_col_types = {}

            table_description = self.db_api.get_table_description(
                self.config.DATA_SOURCES[0], table_name
            )
            table_metadata: dict[str, str] = {
                "description": table_description,
                "dataset_name": self.config.DATA_SOURCES[0],
            }
            if duckdb_col_types:
                table_metadata["column_types"] = json.dumps(
                    duckdb_col_types, ensure_ascii=False
                )

            results.append(
                Table(
                    doc_id=table_name,
                    retriever_type=RetrieverType.ENUMERATOR,
                    content=actual_table,
                    metadata=table_metadata,
                    path=f'{self.config.DATA_SOURCES[0]}."{table_name}"',
                )
            )
        return results

    def index(self, documents: list[AbstractDocument]):
        """
        Indexes a list of documents to the retriever.
        """
        pass
