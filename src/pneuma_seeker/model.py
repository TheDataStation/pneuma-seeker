from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EndpointTag(Enum):
    INDEXING = "indexing"
    CORE = "core"


class IndexDatasetRequest(BaseModel):
    dataset_name: str = Field(min_length=1)
    connector_config: dict[str, Any]
    metadata_available: bool = False
    schema_summaries: list[dict[str, Any]] | None = None


class IndexDatasetResponse(BaseModel):
    run_id: str
    dataset_name: str
    latest_metadata: dict[str, Any] | None


class DatasetMetadataResponse(BaseModel):
    dataset_name: str
    latest_metadata: dict[str, Any]
