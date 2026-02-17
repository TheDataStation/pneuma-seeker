import re

from pneuma_seeker.services.core.ir_system.retriever.abstract_retriever import (
    AbstractRetriever,
)
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
)


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
        regex = re.compile(query)
        match_table_names = [
            table_name for table_name in all_table_names if regex.match(table_name)
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

            table_description = self.db_api.get_table_description(
                self.config.DATA_SOURCES[0], table_name
            )
            table_metadata: dict[str, str] = {
                "description": table_description,
                "dataset_name": self.config.DATA_SOURCES[0],
            }

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
