# src/pneuma_seeker/routers/chat.py
from asyncio import sleep, create_task
from datetime import datetime
from io import BytesIO, StringIO
from json import dumps
from queue import Queue
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED

from anyio import to_thread
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)

from pneuma_seeker.models import ChatHistoryResponse, EndpointTag, PermissionKey
from pneuma_seeker.routers.auth import get_current_user, get_current_user_permissions
from pneuma_seeker.services.db.pneuma_db import PneumaDB
from pneuma_seeker.services.db.users.models import UserRecord
from pneuma_seeker.session_manager import SessionManager
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger
from pneuma_seeker.shared.table_serializer import serialize_dataframe

router = APIRouter(
    prefix="/chat",
    tags=[EndpointTag.CHAT],
)

config = Config("../../../.env")
logger = setup_logger("Chat Router")
pneuma_db = PneumaDB(config, logger)
session_manager = SessionManager(
    config,
    logger,
    pneuma_db,
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

    if dataset_name is None:
        raise HTTPException(
            status_code=400, detail="Missing data source (dataset_name)"
        )
    config.DATA_SOURCES = [dataset_name]

    if latest_user_message is None or not isinstance(latest_user_message, str):
        raise HTTPException(status_code=400, detail="Missing user message")

    chat_session = session_manager.get_chat_session(user_id, chat_id)

    async def event_stream():
        start = datetime.now().timestamp()

        yield stream_payload("log", "Pneuma connected. Starting processing...")
        await sleep(0)

        response_queue: Queue[str | None] = Queue()

        def iter_chat_responses():
            assert latest_user_message is not None
            return chat_session.chat(latest_user_message, files)

        def run_chat():
            try:
                for msg in iter_chat_responses():
                    response_queue.put(msg)
            finally:
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

            try:
                for msg in iter_chat_responses():
                    cur = rss_mb()
                    peak_rss = max(peak_rss, cur)
                    logger.info(f"[MEM] RSS now: {cur:.2f} MB")
                    response_queue.put(msg)
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
                    response = await to_thread.run_sync(response_queue.get)

                    if response is None:
                        break

                    if response.startswith("LOG"):
                        payload = stream_payload("log", response)
                    elif response.startswith("DONE"):
                        payload = stream_payload(
                            "done",
                            f"Processing done in {datetime.now().timestamp() - start:.2f}s.",
                        )
                    else:
                        payload = stream_payload("assistant", response)

                    yield payload
                    await sleep(0)
                except Exception as e:
                    logger.info(f"Exception raised: {e}")
                    break
        finally:
            chat_session.persist_session(dataset_name)
            producer.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
    )


@router.get("/datasets")
def get_accessible_datasets(
    current_user: UserRecord = Depends(get_current_user),
):
    group_permissions = get_current_user_permissions(current_user)

    admin_value = group_permissions.get(PermissionKey.ADMIN.value, False)
    if isinstance(admin_value, str):
        is_admin = admin_value.lower() == "true"
    else:
        is_admin = bool(admin_value)

    datasets = pneuma_db.get_accessible_local_datasets(is_admin, group_permissions)
    return JSONResponse(content={"datasets": datasets})


@router.get("/execute_code/{chat_id}")
async def execute_code(
    chat_id: str,
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Endpoint to trigger execution of Python code (S) on target tables (T) for a given user and chat.
    """
    user_id = current_user.user_id
    conductor = session_manager.get_chat_session(user_id, chat_id).conductor
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
    conductor = session_manager.get_chat_session(
        current_user.user_id, chat_id
    ).conductor
    state = conductor.state.get_current_state_instance(config.TABLE_MAX_ROWS_DISPLAY)

    prov_steps: list[str] = []
    if conductor.state.is_T_materialized:
        prov_explanation_steps_markdown = (
            conductor.materializer.prov_graph.get_graph_explanation()
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
        }
    )


@router.get("/chat/{chat_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    chat_id: str, current_user: UserRecord = Depends(get_current_user)
):
    user_id = current_user.user_id
    chat_session = session_manager.get_chat_session(user_id, chat_id)
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
    materializer_code = chat_session.conductor.materializer.prov_graph.get_graph_code()

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
    chat_session = session_manager.get_chat_session(user_id, chat_id)
    prov_graph = chat_session.conductor.materializer.prov_graph

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
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Endpoint to delete a specific chat session for the authenticated user.
    """
    logger.info(f"Deleting chat session {chat_id} for user {current_user.user_id}")
    try:
        pneuma_db.delete_chat_session(
            user_id=current_user.user_id,
            chat_id=chat_id,
        )
        return JSONResponse(
            content={"detail": f"Chat session '{chat_id}' deleted successfully."}
        )
    except Exception as e:
        logger.info(f"Error deleting chat session: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while deleting the chat session.",
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
