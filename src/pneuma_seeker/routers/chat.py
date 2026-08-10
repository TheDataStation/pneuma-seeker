# src/pneuma_seeker/routers/chat.py
from asyncio import sleep, create_task
from datetime import datetime
from io import BytesIO, StringIO
from json import dumps
from queue import Empty, Queue
from re import match
from threading import Thread
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED

from anyio import to_thread
from shutil import rmtree

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)

from pneuma_seeker.models import ChatHistoryResponse, EndpointTag, PermissionKey
from pneuma_seeker.routers.auth import get_current_user, get_current_user_permissions
from pneuma_seeker.services.core.conductor.models import (
    ConductorResponse,
    ConductorResponseType,
)
from pneuma_seeker.services.db.pneuma_db import PneumaDB
from pneuma_seeker.services.db.users.manager import UserDB
from pneuma_seeker.services.db.users.models import UserRecord
from pneuma_seeker.session_manager import SessionManager
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType
from pneuma_seeker.shared.logger import setup_logger
from pneuma_seeker.shared.parser import parse_json
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import LLMOption
from pneuma_seeker.shared.schemas.language_model.role import Role
from pneuma_seeker.shared.table_serializer import serialize_dataframe

router = APIRouter(
    prefix="/chat",
    tags=[EndpointTag.CHAT],
)

config = Config("../../../.env")
logger = setup_logger("Chat Router")
pneuma_db = PneumaDB(config, logger)
user_db = UserDB(config, logger)
session_manager = SessionManager(
    config,
    logger,
    pneuma_db,
)


_MEMORY_EXTRACTION_SYS_PROMPT = """You are extracting durable, reusable knowledge from a single chat turn in a data-analysis assistant.

Identify at most a few short, atomic pieces of knowledge worth remembering long-term:
- "global": undocumented assumptions, data quirks, or pitfalls about the dataset that would help OTHER users interpret it correctly.
- "local": a personal preference this specific user expressed about how they like analyses done.

Only extract something if it is clearly durable and reusable in future, unrelated turns — not a one-off detail specific to this single question.

Output ONLY a JSON object of the form:
{"entries": [{"scope": "local"|"global", "content": "one short sentence"}]}

If nothing qualifies, return {"entries": []}. Never include more than 3 entries."""


def _extract_memory_safely(
    chat_session,
    user_message: str,
    final_response: str,
    user_id: str,
    group_id: str | None,
    dataset_name: str,
    chat_id: str,
) -> None:
    """
    Best-effort, fire-and-forget extraction of tribal knowledge / user preferences
    from a completed turn. Never allowed to raise — a malformed LLM JSON response
    or any other failure here must not affect the user-facing chat flow, which has
    already completed by the time this runs.
    """
    try:
        messages = [
            LLMMessage(role=Role.SYSTEM.value, content=_MEMORY_EXTRACTION_SYS_PROMPT),
            LLMMessage(
                role=Role.USER.value,
                content=f"User message:\n{user_message}\n\nAssistant response:\n{final_response}",
            ),
        ]
        raw = "".join(
            chat_session.language_model_api.chat(messages, LLMOption(json_mode=True))
        )
        parsed = parse_json(raw)
        entries = parsed.get("entries", []) if isinstance(parsed, dict) else []
        if not entries:
            return
        chat_session.conductor.db_api.record_memory_candidates(
            entries, user_id, group_id, dataset_name, chat_id
        )
    except Exception as e:
        logger.info(f"[Memory Extraction] Skipped due to error: {e}")


def _resolve_dataset_permissions(
    current_user: UserRecord,
) -> tuple[bool, dict[str, str]]:
    """Returns (is_admin, group_permissions) for the current user."""
    group_permissions = get_current_user_permissions(current_user)
    admin_value = group_permissions.get(PermissionKey.ADMIN.value, False)
    is_admin = (
        admin_value.lower() == "true"
        if isinstance(admin_value, str)
        else bool(admin_value)
    )
    return is_admin, group_permissions


def require_dataset_access(dataset_name: str, current_user: UserRecord) -> None:
    """Raises 403 unless `current_user` may access `dataset_name`.

    Dataset access is permission-gated per group (see PermissionKey.DATASET_ACCESS_PREFIX),
    but that's only enforced by the frontend's dataset picker unless every endpoint that
    reads dataset_name from the request also calls this — the picker doesn't stop a direct
    API call from requesting a dataset the caller was never granted.
    """
    is_admin, group_permissions = _resolve_dataset_permissions(current_user)
    if not pneuma_db.is_dataset_accessible(dataset_name, is_admin, group_permissions):
        raise HTTPException(
            status_code=403,
            detail=f"Access to dataset '{dataset_name}' is not permitted.",
        )


@router.post("/")
async def chat(request: Request, current_user: UserRecord = Depends(get_current_user)):
    body: dict[str, Any] = await request.json()
    user_id = current_user.user_id
    chat_id = body.get("chat_id", "default_chat")
    dataset_name: str | None = body.get("dataset_name")
    latest_user_message: str | None = (
        body.get("message") or body.get("user_message") or body.get("content")
    )
    files = body.get("files", [])
    plan_mode = bool(body.get("plan_mode", False))
    use_memory = bool(body.get("use_memory", True)) and config.ENABLE_MEMORY_LAYER

    if dataset_name is None:
        raise HTTPException(
            status_code=400, detail="Missing data source (dataset_name)"
        )
    if latest_user_message is None or not isinstance(latest_user_message, str):
        raise HTTPException(status_code=400, detail="Missing user message")

    require_dataset_access(dataset_name, current_user)

    memory_group_ids = (
        [g.group_id for g in user_db.list_group_ancestors(current_user.group_id)]
        if current_user.group_id
        else []
    )

    chat_session = await session_manager.get_chat_session_async(
        user_id, chat_id, dataset_name
    )
    # The session may have been created dataset-less by another endpoint (e.g.
    # GET /state racing ahead of this being the chat's first-ever message) —
    # this request's dataset_name is authoritative and always corrects it.
    chat_session.set_dataset_name(dataset_name)

    async def event_stream():
        start = datetime.now().timestamp()

        yield stream_payload("log", "Pneuma connected. Starting processing...")
        await sleep(0)

        response_queue: Queue[str | None] = Queue()

        def send_to_frontend(resp: ConductorResponse) -> None:
            if resp.type == ConductorResponseType.LOG:
                response_queue.put(stream_payload("log", resp.message))
            elif resp.type == ConductorResponseType.FINAL_RESPONSE:
                response_queue.put(stream_payload("assistant", resp.message))
            elif resp.type == ConductorResponseType.PLAN_PROPOSAL:
                response_queue.put(stream_payload("plan", resp.message))
            elif resp.type == ConductorResponseType.DONE:
                response_queue.put(
                    stream_payload(
                        "done",
                        f"Processing done in {datetime.now().timestamp() - start:.2f}s.",
                    )
                )

        def _spawn_memory_extraction_if_enabled():
            """
            Best-effort, non-blocking: extracts tribal knowledge/preferences from
            the just-completed turn on a daemon thread. Never allowed to delay or
            fail the response — see _extract_memory_safely for the try/except.
            """
            if not (config.ENABLE_MEMORY_AUTO_EXTRACTION and use_memory):
                return
            if (
                not chat_session.messages
                or chat_session.messages[-1]["role"] != Role.ASSISTANT.value
            ):
                return
            final_response = chat_session.messages[-1]["content"]
            user_msg = (
                chat_session.messages[-2]["content"]
                if len(chat_session.messages) > 1
                else latest_user_message
            )
            Thread(
                target=_extract_memory_safely,
                args=(
                    chat_session,
                    user_msg,
                    final_response,
                    user_id,
                    current_user.group_id,
                    dataset_name,
                    chat_id,
                ),
                daemon=True,
            ).start()

        def run_chat():
            assert latest_user_message is not None
            try:
                for response in chat_session.chat(
                    latest_user_message,
                    files,
                    frontend_callback=send_to_frontend,
                    plan_mode=plan_mode,
                    use_memory=use_memory,
                    memory_group_ids=memory_group_ids,
                ):
                    send_to_frontend(response)
            finally:
                # Persist here, not in event_stream()'s finally: this thread
                # keeps running to completion even after a client disconnect
                # (abandon_on_cancel=True below), so this is the only point
                # guaranteed to run *after* chat_session.chat() has actually
                # finished and appended the assistant's reply. Persisting from
                # event_stream()'s finally instead would fire immediately on
                # disconnect — mid-Conductor-run, before the reply exists —
                # silently discarding the fully-computed answer once this
                # thread finishes with nothing left to save it.
                chat_session.persist_session(dataset_name)
                _spawn_memory_extraction_if_enabled()
                response_queue.put(None)

        def run_chat_with_profiling():
            from psutil import Process
            from os import getpid
            from gc import collect
            from tracemalloc import start, stop, get_traced_memory
            from time import time

            p = Process(getpid())

            def rss_mb():
                return p.memory_info().rss / 1024 / 1024

            collect()
            start()

            baseline_rss = rss_mb()
            t0 = time()

            logger.info(f"[MEM] baseline RSS: {baseline_rss:.2f} MB")

            peak_rss = baseline_rss

            assert latest_user_message is not None
            try:
                for response in chat_session.chat(
                    latest_user_message,
                    files,
                    frontend_callback=send_to_frontend,
                    plan_mode=plan_mode,
                    use_memory=use_memory,
                    memory_group_ids=memory_group_ids,
                ):
                    cur = rss_mb()
                    peak_rss = max(peak_rss, cur)
                    logger.info(f"[MEM] RSS now: {cur:.2f} MB")
                    send_to_frontend(response)
            finally:
                cur, peak = get_traced_memory()
                stop()
                logger.info(f"[MEM] end RSS: {rss_mb():.2f} MB")
                logger.info(f"[MEM] delta RSS: {rss_mb() - baseline_rss:.2f} MB")
                logger.info(f"[MEM] peak RSS delta: {peak_rss - baseline_rss:.2f} MB")
                logger.info(
                    f"[MEM] trace malloc peak Python alloc: {peak / 1024 / 1024:.2f} MB"
                )
                logger.info(f"[TIME] took {time() - t0:.2f}s")
                # See run_chat()'s finally for why this must happen here.
                chat_session.persist_session(dataset_name)
                _spawn_memory_extraction_if_enabled()
                response_queue.put(None)

        producer = create_task(
            to_thread.run_sync(
                run_chat_with_profiling if config.ENABLE_MEMORY_PROFILING else run_chat,
                abandon_on_cancel=True,
            )
        )

        try:
            while True:
                try:
                    response = await to_thread.run_sync(
                        response_queue.get,
                        True,
                        config.STREAM_HEARTBEAT_INTERVAL_SECONDS,
                    )
                except Empty:
                    # Nothing to report within the interval — most commonly a
                    # single slow LLM call with no intermediate progress to log.
                    # Send an inert keepalive so a proxy/load balancer with an
                    # idle-connection timeout doesn't kill the stream while the
                    # backend is still working; the frontend ignores any sender
                    # it doesn't recognize.
                    yield stream_payload("heartbeat", "")
                    await sleep(0)
                    continue
                except Exception as e:
                    logger.info(f"Exception raised: {e}")
                    break

                if response is None:
                    break

                yield response
                await sleep(0)
        finally:
            # Persistence now happens in run_chat()/run_chat_with_profiling()'s
            # own finally (in the background thread), not here — this block
            # can run before the Conductor has actually finished (e.g. on
            # client disconnect), which would persist a half-finished turn.
            producer.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
    )


@router.get("/datasets")
def get_accessible_datasets(
    current_user: UserRecord = Depends(get_current_user),
):
    is_admin, group_permissions = _resolve_dataset_permissions(current_user)
    datasets = pneuma_db.get_accessible_local_datasets(is_admin, group_permissions)
    return JSONResponse(content={"datasets": datasets})


@router.get("/dataset_tables/{dataset_name}")
def list_dataset_tables(
    dataset_name: str,
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Lists the tables in a dataset directly (no chat session required), for
    browsing a dataset's contents outside of any particular conversation.
    """
    require_dataset_access(dataset_name, current_user)
    try:
        tables = pneuma_db.list_dataset_tables(dataset_name)
    except Exception:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_name}' not found."
        )
    return JSONResponse(content={"tables": sorted(tables)})


@router.get("/dataset_table_query/{dataset_name}", response_class=JSONResponse)
def query_dataset_table(
    dataset_name: str,
    table_id: str,
    limit: int = 50,
    offset: int = 0,
    order_by: str | None = None,
    order_dir: str = "asc",
    search: str | None = None,
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Paginated/sortable/searchable read of a single dataset table, directly
    from the dataset's own DB file — the dataset-browsing counterpart to
    /table_query/{chat_id}, which requires an active chat session.
    """
    if not match(r"^[a-zA-Z0-9_]+$", table_id):
        raise HTTPException(status_code=400, detail="Invalid table_id")
    require_dataset_access(dataset_name, current_user)
    if order_dir.lower() not in ("asc", "desc"):
        order_dir = "asc"
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    try:
        rows_df, total_count, columns = pneuma_db.query_dataset_table(
            dataset_name, table_id, limit, offset, order_by, order_dir, search
        )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Table '{table_id}' not found in dataset '{dataset_name}'.",
        )

    return JSONResponse(
        content={
            "rows": serialize_dataframe(rows_df, limit),
            "total_count": total_count,
            "columns": columns,
        }
    )


@router.get("/retrieve_tables/{chat_id}", response_class=JSONResponse)
async def retrieve_tables(
    chat_id: str,
    query: str,
    dataset_name: str | None = None,
    top_k: int = 10,
    sample_only: bool = True,
    sample_size: int = 5,
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Directly invokes Pneuma-Retriever's table discovery for a given chat
    session and returns the results. Does not mutate the session's own
    retrieval state (conductor.retrieved_tables) — this is a standalone
    inspection/debugging call, separate from the agent's own retrieval
    during a turn.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Missing query")
    top_k = max(1, min(top_k, 50))
    sample_size = max(1, min(sample_size, 500))

    user_id = current_user.user_id
    try:
        chat_session = await session_manager.get_chat_session_async(user_id, chat_id)
    except Exception:
        raise HTTPException(
            status_code=404, detail=f"Chat session '{chat_id}' not found."
        )

    resolved_dataset_name = dataset_name or chat_session.dataset_name
    if not resolved_dataset_name:
        raise HTTPException(
            status_code=400,
            detail="No dataset associated with this chat session; pass dataset_name explicitly.",
        )
    require_dataset_access(resolved_dataset_name, current_user)

    conductor = chat_session.conductor
    try:
        retrieved_docs = await to_thread.run_sync(
            lambda: conductor.action_set.retrieve_documents(
                query,
                RetrieverType.PNEUMA_RETRIEVER,
                resolved_dataset_name,
                top_k,
                sample_only,
                sample_size,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")

    tables = [
        {
            "doc_id": doc.doc_id,
            "metadata": doc.metadata,
            "rows": serialize_dataframe(doc.content, config.TABLE_MAX_ROWS_DISPLAY),
        }
        for doc in retrieved_docs
    ]
    return JSONResponse(
        content={
            "query": query,
            "dataset_name": resolved_dataset_name,
            "top_k": top_k,
            "tables": tables,
        }
    )


@router.get("/execute_code/{chat_id}")
async def execute_code(
    chat_id: str,
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Endpoint to trigger execution of Python code (S) on target tables (T) for a given user and chat.
    """
    user_id = current_user.user_id
    conductor = (
        await session_manager.get_chat_session_async(user_id, chat_id)
    ).conductor
    try:
        try:
            execution_result = conductor.db_api.execute_query(
                user_id,
                chat_id,
                f"SELECT * FROM conductor_s_execution LIMIT {config.TABLE_MAX_ROWS_DISPLAY};",
            )
            if len(execution_result) == 0:
                execution_result = conductor.action_set.execute_code(
                    conductor.state.S, "conductor_s_execution"
                )
        except:
            execution_result = conductor.action_set.execute_code(
                conductor.state.S, "conductor_s_execution"
            )
        return serialize_dataframe(execution_result, config.TABLE_MAX_ROWS_DISPLAY)
    except Exception as e:
        print(f"Error during code execution: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during code execution.",
        )


@router.get(
    "/state/{chat_id}",
    response_class=JSONResponse,
)
async def get_state(chat_id: str, current_user: UserRecord = Depends(get_current_user)):
    chat_session = await session_manager.get_chat_session_async(
        current_user.user_id, chat_id
    )
    conductor = chat_session.conductor
    state = conductor.state.get_current_state_instance(config.TABLE_MAX_ROWS_DISPLAY)

    prov_steps: list[str] = []
    if conductor.state.is_T_materialized:
        prov_explanation_steps_markdown = (
            conductor.materializer.prov_graph.get_graph_explanation()  # type: ignore
        )
        if prov_explanation_steps_markdown:
            prov_steps = [str(step_md) for step_md in prov_explanation_steps_markdown]

    retrieved_tables = {
        doc.doc_id: serialize_dataframe(doc.content, config.TABLE_MAX_ROWS_DISPLAY)
        for doc in conductor.retrieved_tables
    }

    return JSONResponse(
        content={
            "state": state,
            "prov_steps": prov_steps,
            "retrieved_tables": retrieved_tables,
            "dataset_name": chat_session.dataset_name,
            "used_memory_entries": conductor.used_memory_entries,  # type: ignore
        }
    )


@router.get("/chat/{chat_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    chat_id: str, current_user: UserRecord = Depends(get_current_user)
):
    user_id = current_user.user_id
    chat_session = await session_manager.get_chat_session_async(user_id, chat_id)
    return ChatHistoryResponse(
        user_id=user_id,
        chat_id=chat_id,
        messages=chat_session.messages,
        dataset_name=chat_session.dataset_name,
    )


@router.get("/target_views/{chat_id}")
def get_target_views(
    chat_id: str, current_user: UserRecord = Depends(get_current_user)
):
    """
    Download target views (CSV files) for a given user and chat as a ZIP file.
    Example: /tables/u123/c45/all
    """
    user_id = current_user.user_id
    conductor = session_manager.get_chat_session(user_id, chat_id).conductor
    target_table_ids = list(conductor.state.T.keys()) if conductor.state.T else []

    if not target_table_ids:
        raise HTTPException(status_code=404, detail="No target tables found")

    # Create a ZIP file in memory (no temp file needed)
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as zipf:
        for table_id in target_table_ids:
            try:
                df = conductor.db_api.execute_query(
                    user_id,
                    chat_id,
                    f"SELECT * FROM {table_id};",
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load table '{table_id}': {exc}",
                )

            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            zipf.writestr(f"{table_id}.csv", csv_buffer.getvalue())

    zip_buffer.seek(0)

    # Stream the zip file to the client
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={user_id}_{chat_id}_tables.zip"
        },
    )


@router.get("/e2e_script/{chat_id}")
def get_e2e_script(chat_id: str, current_user: UserRecord = Depends(get_current_user)):
    """
    Downloads Materializer code (.py) generated for a given user and chat.
    """
    user_id = current_user.user_id
    chat_session = session_manager.get_chat_session(user_id, chat_id)
    materializer_code = chat_session.conductor.materializer.prov_graph.get_graph_code()  # type: ignore

    file_stream = BytesIO()
    file_stream.write(materializer_code.encode("utf-8"))
    file_stream.seek(0)

    filename = f"materializer_{user_id}_{chat_id}.py"
    return StreamingResponse(
        file_stream,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get(
    "/provenance_nodes/{chat_id}",
    response_class=JSONResponse,
)
async def get_provenance_nodes(
    chat_id: str,
    current_user: UserRecord = Depends(get_current_user),
) -> JSONResponse:
    """
    Return all nodes of the provenance graph for a given user and chat.
    """
    user_id = current_user.user_id

    # Get the provenance graph instance
    chat_session = await session_manager.get_chat_session_async(user_id, chat_id)
    prov_graph = chat_session.conductor.materializer.prov_graph  # type: ignore

    # Convert all nodes to JSON-serializable format
    nodes_json = []
    for node in prov_graph.nodes.values():
        nodes_json.append(
            {
                "id": node.id,
                "source_retriever": getattr(
                    node.source_retriever, "value", str(node.source_retriever)
                ),
                "python_code": node.python_code,
                "description": node.description,
                "parents": [p.id for p in node.parents],
                "children": [c.id for c in node.children],
            }
        )

    return JSONResponse(
        content={
            "user_id": user_id,
            "chat_id": chat_id,
            "node_count": len(nodes_json),
            "nodes": nodes_json,
        }
    )


@router.get("/table_query/{chat_id}", response_class=JSONResponse)
async def query_table(
    chat_id: str,
    table_id: str,
    limit: int = 50,
    offset: int = 0,
    order_by: str | None = None,
    order_dir: str = "asc",
    search: str | None = None,
    dataset_name: str | None = None,
    current_user: UserRecord = Depends(get_current_user),
):
    if not match(r"^[a-zA-Z0-9_]+$", table_id):
        raise HTTPException(status_code=400, detail="Invalid table_id")
    if order_dir.lower() not in ("asc", "desc"):
        order_dir = "asc"
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    user_id = current_user.user_id
    try:
        conductor = (
            await session_manager.get_chat_session_async(user_id, chat_id)
        ).conductor
    except Exception:
        raise HTTPException(
            status_code=404, detail=f"Chat session '{chat_id}' not found."
        )

    db_api = conductor.db_api

    # Qualify the table reference with the dataset schema for retrieved tables.
    # dataset_name is sanitized by stripping quotes; table_id is regex-validated above.
    if dataset_name:
        safe_ds = dataset_name.replace('"', "")
        require_dataset_access(safe_ds, current_user)
        db_api.link_dataset_tables(user_id, chat_id, safe_ds)
        table_ref = f'"{safe_ds}"."{table_id}"'
    else:
        table_ref = f'"{table_id}"'

    try:
        cols_df = db_api.execute_query(
            user_id, chat_id, f"SELECT * FROM {table_ref} LIMIT 0"
        )
    except Exception:
        raise HTTPException(
            status_code=404, detail=f"Table '{table_id}' not found in session."
        )

    columns = list(cols_df.columns)

    where_sql = ""
    sql_params: tuple = ()
    if search and columns:
        conditions = " OR ".join(f'CAST("{col}" AS VARCHAR) ILIKE ?' for col in columns)
        where_sql = f" WHERE {conditions}"
        sql_params = tuple([f"%{search}%"] * len(columns))

    count_df = db_api.execute_query(
        user_id,
        chat_id,
        f"SELECT COUNT(*) AS cnt FROM {table_ref}{where_sql}",
        sql_params,
    )
    total_count = int(count_df.iloc[0]["cnt"])

    order_sql = ""
    if order_by and match(r"^[a-zA-Z0-9_]+$", order_by):
        order_sql = f' ORDER BY "{order_by}" {order_dir.upper()}'

    rows_df = db_api.execute_query(
        user_id,
        chat_id,
        f"SELECT * FROM {table_ref}{where_sql}{order_sql} LIMIT {limit} OFFSET {offset}",
        sql_params,
    )

    return JSONResponse(
        content={
            "rows": serialize_dataframe(rows_df, limit),
            "total_count": total_count,
            "columns": columns,
        }
    )


@router.get("/explain_script/{chat_id}", response_class=JSONResponse)
async def explain_script(
    chat_id: str,
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Lazily generate and cache a plain-language description of the current script (S).
    Returns the cached description on subsequent calls without re-invoking the LLM.
    """
    user_id = current_user.user_id
    try:
        chat_session = await session_manager.get_chat_session_async(user_id, chat_id)
    except Exception:
        raise HTTPException(
            status_code=404, detail=f"Chat session '{chat_id}' not found."
        )

    conductor = chat_session.conductor
    script = conductor.state.S.strip()

    if not script:
        raise HTTPException(status_code=400, detail="No script to explain.")

    # Return cached description without calling the LLM
    if conductor.state.s_description:
        return JSONResponse(content={"description": conductor.state.s_description})

    # Build a brief context of target table schemas for the LLM
    table_context = ""
    if conductor.state.T:
        lines = []
        for table_id, doc in conductor.state.T.items():
            try:
                cols = list(doc.content.columns)[:15]
                lines.append(f"- {table_id}: {', '.join(cols)}")
            except Exception:
                lines.append(f"- {table_id}")
        if lines:
            table_context = "\n\nTarget tables:\n" + "\n".join(lines)

    prompt = (
        "You are explaining a data analysis script to a non-technical user.\n\n"
        "First, write 1-2 sentences describing what the script computes or retrieves "
        "in plain language. Focus on the outcome, not the implementation.\n\n"
        "Then, if applicable, add a short bullet list under the heading '**Assumptions & caveats:**' "
        "covering important assumptions or nuances the user should know "
        "(e.g. statistical model assumptions, data quality requirements, "
        "interpretation caveats, known limitations). "
        "Omit this section entirely if there is nothing meaningful to flag. "
        "Keep the whole response concise."
        f"\n\nScript:\n{script[:2000]}"
        f"{table_context}"
    )

    messages = [LLMMessage(role="user", content=prompt)]
    raw = await to_thread.run_sync(
        lambda: chat_session.language_model_api.chat(messages)
    )
    description = ("".join(raw) if not isinstance(raw, str) else raw).strip()

    conductor.state.s_description = description
    await to_thread.run_sync(
        lambda: conductor.db_api.update_script_description(
            user_id, chat_id, description
        )
    )

    return JSONResponse(content={"description": description})


@router.get("/table_download/{chat_id}")
async def download_table(
    chat_id: str,
    table_id: str,
    dataset_name: str | None = None,
    current_user: UserRecord = Depends(get_current_user),
):
    if not match(r"^[a-zA-Z0-9_]+$", table_id):
        raise HTTPException(status_code=400, detail="Invalid table_id")

    user_id = current_user.user_id
    try:
        conductor = (
            await session_manager.get_chat_session_async(user_id, chat_id)
        ).conductor
    except Exception:
        raise HTTPException(
            status_code=404, detail=f"Chat session '{chat_id}' not found."
        )

    db_api = conductor.db_api

    if dataset_name:
        safe_ds = dataset_name.replace('"', "")
        require_dataset_access(safe_ds, current_user)
        db_api.link_dataset_tables(user_id, chat_id, safe_ds)
        table_ref = f'"{safe_ds}"."{table_id}"'
    else:
        table_ref = f'"{table_id}"'

    _CHUNK = 5_000

    async def generate_csv():
        offset = 0
        first = True
        while True:
            _offset = offset
            _first = first
            df = await to_thread.run_sync(
                lambda: db_api.execute_query(
                    user_id,
                    chat_id,
                    f"SELECT * FROM {table_ref} LIMIT {_CHUNK} OFFSET {_offset}",
                )
            )
            if df.empty:
                break
            buf = StringIO()
            df.to_csv(buf, index=False, header=_first)
            yield buf.getvalue().encode("utf-8")
            first = False
            if len(df) < _CHUNK:
                break
            offset += _CHUNK

    return StreamingResponse(
        generate_csv(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{table_id}.csv"',
        },
    )


@router.get("/sessions", response_class=JSONResponse)
async def list_chats(
    limit: int = 10,
    offset: int = 0,
    query: str | None = None,
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Endpoint to retrieve a paginated list of chat sessions for the authenticated user,
    ordered by their last active timestamp. When query is provided, performs a
    lightweight ILIKE content search across all chat messages.
    """
    logger.info(
        f"Listing chat sessions for user {current_user.user_id} with limit={limit}, offset={offset}, query={query!r}"
    )
    try:
        if query:
            result = pneuma_db.search_chat_sessions(
                user_id=current_user.user_id,
                query=query,
                limit=limit,
                offset=offset,
            )
        else:
            result = pneuma_db.get_user_chat_sessions(
                user_id=current_user.user_id,
                limit=limit,
                offset=offset,
            )
        return JSONResponse(content=result)
    except Exception as e:
        logger.info(f"Error listing chat sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving chat sessions.",
        )


@router.delete("/{chat_id}", response_class=JSONResponse)
async def delete_chat_session(
    chat_id: str,
    background_tasks: BackgroundTasks,
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Endpoint to delete a specific chat session for the authenticated user.
    The DB connection is closed eagerly; the workspace directory is removed
    asynchronously as a background task after the response is sent.
    """
    logger.info(f"Deleting chat session {chat_id} for user {current_user.user_id}")
    try:
        chat_dir = pneuma_db.prepare_chat_deletion(
            user_id=current_user.user_id,
            chat_id=chat_id,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Chat session '{chat_id}' not found.",
        )
    except Exception as e:
        logger.info(f"Error preparing chat session deletion: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while deleting the chat session.",
        )

    session_manager.evict_chat_session(current_user.user_id, chat_id)
    background_tasks.add_task(rmtree, chat_dir)
    return JSONResponse(
        content={"detail": f"Chat session '{chat_id}' deleted successfully."}
    )


def stream_payload(sender: str, text: str) -> str:
    """Formats a message payload for streaming responses."""
    return (
        dumps(
            {
                "sender": sender,
                "text": text,
                "time_stamp": int(datetime.now().timestamp() * 1000),
            }
        )
        + "\n"
    )
