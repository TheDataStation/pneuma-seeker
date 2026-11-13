import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pandas import DataFrame

from pneuma_seeker.models import (
    DatasetSourceRequest,
    EndpointTag,
    IndexDatasetRequest,
    TestDatasetSourceRequest,
    UpdateDatasetSourceRequest,
)
from pneuma_seeker.routers.auth import get_current_user
from pneuma_seeker.routers.chat import (
    _resolve_dataset_permissions,
    config,
    logger,
    pneuma_db,
)
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.ir_system.retriever.impl.new_discovery import NewDiscovery
from pneuma_seeker.services.db.datasets import uploads
from pneuma_seeker.services.db.datasets.connectors import CONNECTOR_TYPES
from pneuma_seeker.services.db.datasets.registry import DatasetSource
from pneuma_seeker.services.db.users.models import UserRecord
from pneuma_seeker.services.indexing.main import IndexingService
from pneuma_seeker.services.indexing.metadata_store import IndexingMetadataStore

router = APIRouter(prefix="/datasets", tags=[EndpointTag.DATASETS])

_CHUNK = 1024 * 1024


def require_admin(current_user: UserRecord = Depends(get_current_user)) -> UserRecord:
    """Raises 403 unless the current user is an administrator."""
    is_admin, _ = _resolve_dataset_permissions(current_user)
    if not is_admin:
        raise HTTPException(status_code=403, detail="Only administrators can manage datasets.")
    return current_user


def _describe(source: DatasetSource) -> dict:
    return {
        "name": source.name,
        "connector_type": source.connector_type,
        "connector_label": CONNECTOR_TYPES[source.connector_type].label,
        "is_local": pneuma_db.dataset_manager.is_local(source),
    }


def _refused(exc: uploads.UploadRefused) -> HTTPException:
    return HTTPException(status_code=uploads.STATUS.get(exc.kind, 400), detail=exc.message)


@router.get("")
def list_datasets(current_user: UserRecord = Depends(get_current_user)):
    """Lists the datasets the current user may access, with their connector."""
    is_admin, permissions = _resolve_dataset_permissions(current_user)
    accessible = set(pneuma_db.get_accessible_datasets(is_admin, permissions))
    return {
        "datasets": [_describe(s) for s in pneuma_db.list_datasets() if s.name in accessible],
        "can_manage": is_admin,
    }


@router.get("/connectors")
def list_connectors(_: UserRecord = Depends(require_admin)):
    """Lists the connector types a dataset can be added with."""
    return {
        "connectors": [
            {"type": cls.type, "label": cls.label, "secret_fields": sorted(cls.SECRET_FIELDS)}
            for cls in CONNECTOR_TYPES.values()
        ]
    }


@router.get("/uploads/limits")
def upload_limits(_: UserRecord = Depends(require_admin)):
    """Returns the upload limits and accepted file types."""
    return {"max_file_bytes": config.UPLOAD_MAX_BYTES, "accepted": sorted(uploads.FORMATS)}


@router.put("/{dataset_name}/upload/{file_name}")
async def upload_file(
    dataset_name: str,
    file_name: str,
    request: Request,
    create: bool = False,
    replace: bool = False,
    current_user: UserRecord = Depends(require_admin),
):
    """
    Adds the tables of one file, sent as the raw request body, to a local
    dataset (creating it with `create`); `replace` overwrites tables of the same name.
    """
    try:
        uploads.file_format(file_name)
    except uploads.UploadRefused as exc:
        raise _refused(exc)
    root = Path(config.DATASETS_ROOT)
    declared = int(request.headers.get("content-length") or 0)
    if declared > config.UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail=_too_large())
    if declared and shutil.disk_usage(root).free < 2 * declared:
        raise HTTPException(status_code=507, detail="Not enough disk space for this upload.")

    staging = uploads.staging_dir(root)
    try:
        staged = staging / Path(file_name).name
        size = 0
        with open(staged, "wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > config.UPLOAD_MAX_BYTES:
                    raise HTTPException(status_code=413, detail=_too_large())
                out.write(chunk)
        try:
            result = await run_in_threadpool(
                pneuma_db.add_file_to_dataset,
                dataset_name,
                staged,
                file_name,
                create=create,
                replace=replace,
                created_by=current_user.user_id,
            )
        except uploads.UploadRefused as exc:
            raise _refused(exc)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    logger.info(f"[Datasets] Uploaded {file_name!r} to {dataset_name!r}: {[t.name for t in result.tables]}")
    return {
        "dataset_name": result.dataset_name,
        "created_dataset": result.created_dataset,
        "tables": [
            {"name": t.name, "rows": t.rows, "columns": len(t.columns), "replaced": t.replaced}
            for t in result.tables
        ],
    }


@router.delete("/{dataset_name}/tables/{table_name}")
def delete_table(dataset_name: str, table_name: str, _: UserRecord = Depends(require_admin)):
    """Deletes a table of a local dataset, with its descriptions."""
    try:
        pneuma_db.delete_dataset_table(dataset_name, table_name)
    except uploads.UploadRefused as exc:
        raise _refused(exc)
    return {"deleted": table_name}


@router.delete("/{dataset_name}")
def remove_dataset(dataset_name: str, _: UserRecord = Depends(require_admin)):
    """Removes a dataset: a local one's files are deleted, a connected source is left alone."""
    try:
        pneuma_db.remove_dataset(dataset_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"removed": dataset_name}


@router.get("/{dataset_name}/source")
def get_source(dataset_name: str, _: UserRecord = Depends(require_admin)):
    """Returns a dataset's connector settings (secrets masked) and whether it can be read."""
    source = pneuma_db.get_dataset_source(dataset_name)
    if source is None:
        raise HTTPException(status_code=404, detail=f"No dataset named {dataset_name!r}.")
    connector = pneuma_db.dataset_manager.connector(dataset_name)
    ok, error, tables = connector.health_check()
    return {
        **_describe(source),
        "config": connector.redacted_config(),
        "health": {"ok": ok, "error": error, "table_count": tables},
    }


@router.post("/sources/test")
def test_source(payload: TestDatasetSourceRequest, _: UserRecord = Depends(require_admin)):
    """Checks whether a source can be read, without registering it."""
    config_ = payload.config
    if payload.name:
        source = pneuma_db.get_dataset_source(payload.name)
        if source is not None:
            secrets = CONNECTOR_TYPES[source.connector_type].SECRET_FIELDS
            config_ = {**config_, **{f: source.config.get(f, "") for f in secrets if not config_.get(f)}}
    try:
        ok, error, tables = pneuma_db.dataset_manager.test_source(payload.connector_type, config_)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": ok, "error": error, "table_count": tables}


@router.post("/sources")
def add_source(payload: DatasetSourceRequest, current_user: UserRecord = Depends(require_admin)):
    """Registers a dataset over an existing source (a folder, database file or Postgres database)."""
    try:
        source = pneuma_db.dataset_manager.add_source(
            payload.name, payload.connector_type, payload.config, current_user.user_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _describe(source)


@router.put("/sources/{dataset_name}")
def update_source(dataset_name: str, payload: UpdateDatasetSourceRequest, _: UserRecord = Depends(require_admin)):
    """Replaces a connected dataset's settings; a blank password keeps the stored one."""
    try:
        source = pneuma_db.dataset_manager.update_source(dataset_name, payload.config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _describe(source)


# Index builds started in this process: dataset -> {"error", "done", "total"}
# ("error" is None while building; progress is known for M1 only).
_jobs: dict[str, dict] = {}


def _discovery(dataset_name: str) -> NewDiscovery:
    # Each dataset gets its own workspace connection, so builds of different datasets run side by side.
    db_api = DBAPI(config, logger, pneuma_db=pneuma_db)
    return NewDiscovery("indexing_service", f"index-{dataset_name}", config, db_api, None)  # type: ignore[arg-type]


@router.post("/{dataset_name}/index")
def rebuild_index(
    dataset_name: str,
    background_tasks: BackgroundTasks,
    payload: IndexDatasetRequest | None = None,
    _: UserRecord = Depends(require_admin),
):
    """
    Rebuilds the dataset's retrieval index in the background, for the configured
    TABLE_DISCOVERY_METHOD: M1's (otherwise rebuilt on its own when files change,
    but not when a connected database's contents do) or the Pneuma retriever's.
    """
    if pneuma_db.get_dataset_source(dataset_name) is None:
        raise HTTPException(status_code=404, detail=f"No dataset named {dataset_name!r}.")
    job = _jobs.get(dataset_name)
    if job is not None and job["error"] is None:
        return {"dataset_name": dataset_name, "method": config.TABLE_DISCOVERY_METHOD, "started": False}
    job: dict = {"error": None, "done": 0, "total": None}

    def progress(done: int, total: int) -> None:
        job["done"], job["total"] = done, total

    if config.TABLE_DISCOVERY_METHOD == "m1":
        def build() -> None:
            _discovery(dataset_name).rebuild_index(dataset_name, progress)
    else:
        payload = payload or IndexDatasetRequest()
        service = IndexingService(config, logger, DBAPI(config, logger, pneuma_db=pneuma_db))
        run_id = service.start_indexing_run(dataset_name)
        summaries = DataFrame(payload.schema_summaries) if payload.schema_summaries is not None else None

        def build() -> None:
            try:
                service.run_indexing_job(dataset_name, run_id, schema_summaries=summaries, overwrite=payload.overwrite)
            finally:
                service.close()

    def run() -> None:
        try:
            build()
            _jobs.pop(dataset_name, None)
        except Exception as exc:
            job["error"] = str(exc) or type(exc).__name__
            logger.error(f"[Datasets] Indexing {dataset_name!r} failed: {exc}")

    _jobs[dataset_name] = job
    background_tasks.add_task(run)
    return {"dataset_name": dataset_name, "method": config.TABLE_DISCOVERY_METHOD, "started": True}


@router.get("/{dataset_name}/index")
def index_status(dataset_name: str, _: UserRecord = Depends(require_admin)):
    """
    Returns the dataset's index state for the configured method: "building"
    (with `progress` {done, total} when known), "failed" (with `error`),
    "built" (with `built_at`) or "missing".
    """
    method = config.TABLE_DISCOVERY_METHOD
    state = {"dataset_name": dataset_name, "method": method, "built_at": None, "error": None, "progress": None}
    job = _jobs.get(dataset_name)
    if job is not None:
        if job["error"] is not None:
            return {**state, "state": "failed", "error": job["error"]}
        progress = {"done": job["done"], "total": job["total"]} if job["total"] else None
        return {**state, "state": "building", "progress": progress}
    if method == "m1":
        meta = _discovery(dataset_name).index_status(dataset_name)
        return {**state, "state": "built" if meta else "missing", "built_at": (meta or {}).get("built_at")}
    store = IndexingMetadataStore()
    try:
        run = store.get_latest_run(dataset_name)
    finally:
        store.close()
    if not run:
        return {**state, "state": "missing"}
    status = str(run.get("status", "")).upper()
    if status == "FAILED":
        return {**state, "state": "failed", "error": run.get("error_message")}
    built_at = run.get("completed_at")
    return {**state, "state": "built" if status == "SUCCEEDED" else "building", "built_at": str(built_at) if built_at else None}


def _too_large() -> str:
    return f"The file is larger than the upload limit ({config.UPLOAD_MAX_BYTES // 1024**2} MiB)."
