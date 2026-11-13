from logging import Logger

from pandas import DataFrame

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.ir_system.retriever.impl.pneuma_retriever import (
    PneumaRetriever,
)
from pneuma_seeker.services.db.datasets.connectors import build
from pneuma_seeker.services.indexing.metadata_store import IndexingMetadataStore
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
    TableContext,
    Text,
)
from pneuma_seeker.shared.str_processor import clean_column_table_name


class IndexingService:
    """
    Builds the Pneuma retriever's indices for a registered dataset (an LLM call
    per column, so it's run on demand). M1 needs none: it indexes a dataset on
    its first search.
    """

    def __init__(self, config: Config, logger: Logger, db_api: DBAPI | None = None):
        self.config = config
        self.logger = logger
        self.db_api = db_api or DBAPI(config, logger)
        self.language_model_api = LanguageModelAPI(config, logger)
        self.retriever = PneumaRetriever(
            user_id="indexing_service",
            chat_id="indexing_service",
            config=config,
            db_api=self.db_api,
            language_model_api=self.language_model_api,
        )
        self.metadata_store = IndexingMetadataStore()

    def start_indexing_run(self, dataset_name: str) -> str:
        """Records a new indexing run of a registered dataset and returns its ID. Raises ValueError."""
        source = self.db_api.pneuma_db.get_dataset_source(dataset_name)
        if source is None:
            raise ValueError(f"No dataset named {dataset_name!r}.")
        connector = build(source.connector_type, source.config)
        return self.metadata_store.record_run_started(
            dataset_name=dataset_name,
            source_type=source.connector_type,
            source_config=connector.redacted_config(),
            snapshot_id=self.db_api.dataset_fingerprint(dataset_name),
        )

    def run_indexing_job(
        self,
        dataset_name: str,
        run_id: str,
        schema_summaries: DataFrame | None = None,
        overwrite: bool = True,
    ) -> None:
        """Builds the dataset's indices and records whether the run succeeded."""
        try:
            tables = self.db_api.list_dataset_tables(dataset_name)
            if not tables:
                raise ValueError(f"Dataset {dataset_name!r} has no readable tables.")
            documents = self.__build_documents(dataset_name, tables)
            if schema_summaries is not None:
                self.retriever.index_with_existing_schema_summaries(
                    documents,
                    existing_schema_summaries=self.__schema_summaries_to_docs(dataset_name, schema_summaries),
                    overwrite=overwrite,
                )
            else:
                self.retriever.index(documents, overwrite)
            self.metadata_store.mark_run_succeeded(
                run_id=run_id, indexed_stream_count=len(tables), indexed_table_count=len(tables)
            )
        except Exception as exception:
            self.metadata_store.mark_run_failed(run_id, str(exception))
            raise

    def get_latest_index_metadata(self, dataset_name: str) -> dict | None:
        """Returns the latest indexing run of a dataset, or None."""
        return self.metadata_store.get_latest_run(dataset_name)

    def close(self):
        self.metadata_store.close()

    def __build_documents(self, dataset_name: str, tables: list[str]) -> list[AbstractDocument]:
        # The retriever reads each table itself, so the documents only name them.
        documents: list[AbstractDocument] = []
        for table_name in tables:
            metadata = {"table_name": table_name, "dataset_name": dataset_name}
            documents.append(
                Table(
                    doc_id=f"{dataset_name}/{table_name}",
                    retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                    content=DataFrame(),
                    metadata=metadata,
                )
            )
            description = self.db_api.get_table_description(dataset_name, table_name)
            if description:
                documents.append(
                    TableContext(
                        doc_id=f"context_{dataset_name}/{table_name}",
                        retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                        content=description,
                        metadata={**metadata, "type": "description"},
                    )
                )
        return documents

    def __schema_summaries_to_docs(self, dataset_name: str, schema_summaries: DataFrame) -> list[Text]:
        """Converts schema summaries (table_name, summary[, column_name]) into one document per table."""
        missing = {"table_name", "summary"} - set(schema_summaries.columns)
        if missing:
            raise ValueError(f"schema_summaries is missing required columns: {sorted(missing)}")
        order = ["table_name", "column_name"] if "column_name" in schema_summaries.columns else ["table_name"]
        schema_summaries = schema_summaries.sort_values(by=order, kind="stable")
        docs: list[Text] = []
        for (table_name,), group in schema_summaries.groupby(["table_name"], sort=False):
            cleaned = clean_column_table_name(str(table_name))
            docs.append(
                Text(
                    doc_id=f"{dataset_name}/{cleaned}_schema_summary",
                    retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                    content=" | ".join(group["summary"].astype(str).tolist()),
                    metadata={"table_name": cleaned, "dataset_name": dataset_name},
                )
            )
        return docs
