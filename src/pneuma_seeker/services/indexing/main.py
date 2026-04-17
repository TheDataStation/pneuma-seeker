from datetime import UTC, datetime
from logging import Logger
from pathlib import Path
import tempfile
from typing import Any

from pandas import DataFrame

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.ir_system.retriever.impl.pneuma_retriever import PneumaRetriever
from pneuma_seeker.services.indexing.connectors.base import SourceConnector
from pneuma_seeker.services.indexing.connectors.csv_connector import CSVConnector
from pneuma_seeker.services.indexing.connectors.postgresql_connector import PostgreSQLConnector
from pneuma_seeker.services.indexing.metadata_store import IndexingMetadataStore
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument, RetrieverType, Table, TableContext, Text
from pneuma_seeker.shared.str_processor import clean_column_table_name


class IndexingService:
    """Orchestrates data registration and retriever indexing."""

    def __init__(
        self,
        config: Config,
        logger: Logger,
    ):
        self.config = config
        self.logger = logger

        self.db_api = DBAPI(config, logger)
        self.language_model_api = LanguageModelAPI(config, logger)
        self.retriever = PneumaRetriever(
            user_id="indexing_service",
            chat_id="indexing_service",
            config=config,
            db_api=self.db_api,
            language_model_api=self.language_model_api,
        )
        self.metadata_store = IndexingMetadataStore()

        self._connector_registry: dict[str, type[SourceConnector]] = {
            "csv": CSVConnector,
            "postgres": PostgreSQLConnector,
            "postgresql": PostgreSQLConnector,
        }

    def register_connector(
        self, source_type: str, connector_cls: type[SourceConnector]
    ):
        self._connector_registry[source_type.lower()] = connector_cls

    def create_connector(self, connector_config: dict[str, Any]) -> SourceConnector:
        source_type = str(connector_config.get("type", "")).lower()
        if not source_type:
            raise ValueError("Connector configuration requires 'type'.")

        connector_cls = self._connector_registry.get(source_type)
        if connector_cls is None:
            raise ValueError(f"Unsupported connector type: {source_type}")

        return connector_cls(connector_config)

    def index_dataset(
        self,
        dataset_name: str,
        connector_config: dict[str, Any],
        metadata_available: bool = False,
        schema_summaries: DataFrame | None = None,
    ) -> str:
        connector = self.create_connector(connector_config)
        snapshot_id = self._build_snapshot_id(connector_config)

        run_id = self.metadata_store.record_run_started(
            dataset_name=dataset_name,
            source_type=connector.source_type,
            source_config=connector_config,
            snapshot_id=snapshot_id,
        )

        try:
            if not connector.check_connection():
                raise RuntimeError("Failed to connect to data source.")

            streams = connector.discover()
            if not streams:
                raise ValueError("No streams discovered from the source.")

            documents = self._build_documents(
                dataset_name=dataset_name,
                connector=connector,
                streams=streams,
                metadata_available=metadata_available,
            )

            table_docs = [doc for doc in documents if isinstance(doc, Table)]

            schema_summary_docs: list[Text] | None = None
            if schema_summaries is not None:
                schema_summary_docs = self._schema_summaries_to_docs(
                    dataset_name=dataset_name,
                    schema_summaries=schema_summaries,
                )

            self._register_dataset_for_querying(dataset_name, connector, table_docs)

            if schema_summary_docs is not None:
                self.retriever.index_with_existing_schema_summaries(
                    documents,
                    existing_schema_summaries=schema_summary_docs,
                )
            else:
                self.retriever.index(documents)

            self.metadata_store.mark_run_succeeded(
                run_id=run_id,
                indexed_stream_count=len(streams),
                indexed_table_count=len(table_docs),
            )
            return run_id
        except Exception as exception:
            self.metadata_store.mark_run_failed(run_id, str(exception))
            raise

    def get_latest_index_metadata(self, dataset_name: str) -> dict | None:
        return self.metadata_store.get_latest_run(dataset_name)

    def close(self):
        self.metadata_store.close()

    def _build_snapshot_id(self, connector_config: dict[str, Any]) -> str:
        explicit_snapshot_id = connector_config.get("snapshot_id")
        if explicit_snapshot_id:
            return str(explicit_snapshot_id)

        return datetime.now(tz=UTC).isoformat(timespec="seconds")

    def _stream_name(self, stream_info: dict[str, Any]) -> str:
        raw = (
            stream_info.get("table_name")
            or stream_info.get("stream")
            or stream_info.get("name")
        )
        if raw is None:
            raise ValueError("Each discovered stream must include a stream name.")
        return clean_column_table_name(str(raw).replace(".", "_"))

    def _build_documents(
        self,
        dataset_name: str,
        connector: SourceConnector,
        streams: list[dict[str, Any]],
        metadata_available: bool,
    ) -> list[AbstractDocument]:
        documents: list[AbstractDocument] = []

        for stream_info in streams:
            source_stream = str(
                stream_info.get("stream")
                or stream_info.get("source_table_name")
                or stream_info.get("table_name")
            )
            table_name = self._stream_name(stream_info)

            rows = list(connector.read(source_stream))
            table_df = DataFrame(rows)

            # Preserve schema shape for empty streams when columns are known.
            if table_df.empty and isinstance(stream_info.get("columns"), list):
                table_df = DataFrame(columns=stream_info["columns"])

            documents.append(
                Table(
                    doc_id=f"{dataset_name}/{table_name}",
                    retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                    content=table_df,
                    metadata={
                        "table_name": table_name,
                        "dataset_name": dataset_name,
                        "source_stream": source_stream,
                    },
                )
            )

            description = stream_info.get("description")
            if metadata_available and description:
                documents.append(
                    TableContext(
                        doc_id=f"context_{dataset_name}/{table_name}",
                        retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                        content=str(description),
                        metadata={
                            "table_name": table_name,
                            "dataset_name": dataset_name,
                            "type": "description",
                        },
                    )
                )

        return documents

    def _schema_summaries_to_docs(
        self,
        dataset_name: str,
        schema_summaries: DataFrame,
    ) -> list[Text]:
        required_cols = {"table_name", "summary"}
        missing = required_cols - set(schema_summaries.columns)
        if missing:
            raise ValueError(
                f"schema_summaries is missing required columns: {sorted(missing)}"
            )

        if "column_name" in schema_summaries.columns:
            schema_summaries = schema_summaries.sort_values(
                by=["table_name", "column_name"], kind="stable"
            )
        else:
            schema_summaries = schema_summaries.sort_values(
                by=["table_name"], kind="stable"
            )

        docs: list[Text] = []
        grouped = schema_summaries.groupby(["table_name"], sort=False)
        for table_name, group in grouped:
            cleaned = clean_column_table_name(str(table_name))
            docs.append(
                Text(
                    doc_id=f"{dataset_name}/{cleaned}_schema_summary",
                    retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                    content=" | ".join(group["summary"].astype(str).tolist()),
                    metadata={
                        "table_name": cleaned,
                        "dataset_name": dataset_name,
                    },
                )
            )

        return docs

    def _register_dataset_for_querying(
        self,
        dataset_name: str,
        connector: SourceConnector,
        table_docs: list[Table],
    ):
        if connector.source_type == "postgres":
            connection_string = getattr(connector, "connection_string", None)
            if not connection_string:
                raise ValueError(
                    "PostgreSQL connector must expose a connection_string for registration."
                )
            self.db_api.register_postgres_dataset(dataset_name, connection_string)
            return

        with tempfile.TemporaryDirectory(prefix=f"indexing_{dataset_name}_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            for table_doc in table_docs:
                table_name = table_doc.metadata["table_name"]
                csv_path = tmpdir_path / f"{table_name}.csv"
                table_doc.content.to_csv(csv_path, index=False)

            self.db_api.ingest_dataset(dataset_name, tmpdir)
