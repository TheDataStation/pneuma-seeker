from logging import Logger

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pandas import DataFrame

from pneuma_seeker.models import DatasetMetadataResponse, EndpointTag, IndexDatasetRequest, IndexDatasetResponse
from pneuma_seeker.services.indexing.main import IndexingService
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger


router = APIRouter(
    prefix="/index",
    tags=[EndpointTag.INDEXING],
)


def get_config() -> Config:
    """Returns a Config instance initialized from the .env file."""
    return Config("../../../.env")


def get_logger() -> Logger:
    """Returns a Logger instance for the Indexing Router."""
    return setup_logger("Indexing Router")


def get_indexing_service(
    config: Config = Depends(get_config), 
    logger: Logger = Depends(get_logger)
) -> IndexingService:
    """Returns an IndexingService instance initialized with the provided Config and Logger."""
    return IndexingService(
        config,
        logger,
    )


@router.post("/", response_model=IndexDatasetResponse, tags=[EndpointTag.INDEXING])
def index_dataset(
    payload: IndexDatasetRequest,
    background_tasks: BackgroundTasks,
    indexing_service: IndexingService = Depends(get_indexing_service)
) -> IndexDatasetResponse:
    """Handles requests to start indexing a dataset, schedules the indexing job as a background task, and returns initial metadata."""
    schema_summaries_df: DataFrame | None = None
    if payload.schema_summaries is not None:
        schema_summaries_df = DataFrame(payload.schema_summaries)

    try:
        run_id = indexing_service.start_indexing_run(
            dataset_name=payload.dataset_name,
            connector_config=payload.connector_config,
        )
    except ValueError as exception:
        raise HTTPException(status_code=400, detail=str(exception)) from exception
    except RuntimeError as exception:
        raise HTTPException(status_code=502, detail=str(exception)) from exception
    except Exception as exception:
        raise HTTPException(
            status_code=500, detail=f"Failed to start indexing: {exception}"
        ) from exception

    background_tasks.add_task(
        indexing_service.run_indexing_job,
        dataset_name=payload.dataset_name,
        connector_config=payload.connector_config,
        run_id=run_id,
        schema_summaries=schema_summaries_df,
        overwrite=payload.overwrite,
    )

    return IndexDatasetResponse(
        run_id=run_id,
        dataset_name=payload.dataset_name,
        latest_metadata=indexing_service.get_latest_index_metadata(
            payload.dataset_name
        ),
    )


@router.get(
    "/{dataset_name}/latest",
    response_model=DatasetMetadataResponse,
    tags=[EndpointTag.INDEXING],
)
def get_latest_metadata_endpoint(
    dataset_name: str,
    indexing_service: IndexingService = Depends(get_indexing_service)
) -> DatasetMetadataResponse:
    """Handles requests to fetch the latest indexing metadata for a given dataset name."""
    metadata = indexing_service.get_latest_index_metadata(dataset_name)
    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail=f"No indexing metadata found for dataset '{dataset_name}'.",
        )

    return DatasetMetadataResponse(dataset_name=dataset_name, latest_metadata=metadata)
