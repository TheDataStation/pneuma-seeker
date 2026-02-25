# src/pneuma_seeker/main.py
from asyncio import create_task, sleep
from datetime import datetime
from io import BytesIO, StringIO
from json import dumps
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from anyio import to_thread
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from markdown import markdown
from markdown2 import markdown as markdown_2

from pneuma_seeker.session_manager import SessionManager
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.table_serializer import serialize_dataframe

app = FastAPI(title="Pneuma-Seeker")
logger = setup_logger("Core Service")
config = Config("../../.env")
session_manager = SessionManager(
    config,
    logger,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = (
    Path(__file__).resolve().parents[2]
)  # go up from /src/pneuma_seeker/main.py → project root
TABLES_DIR = BASE_DIR / "data_src" / "target_tables"

templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent / "templates")
)

TITLE_GENERATION_PREFIX = "### Task:\nGenerate a concise, 3-5 word title with an emoji summarizing the chat history.\n### Guidelines:\n- The title should clearly represent the main theme or subject of the conversation.\n- Use emojis that enhance understanding of the topic, but avoid quotation marks or special formatting."


# Helper functions
def now_ms() -> int:
    """Returns the current time in milliseconds."""
    return int(datetime.now().timestamp() * 1000)


def stream_payload(sender: str, text: str) -> str:
    """Formats a message payload for streaming responses."""
    return (
        dumps(
            {
                "sender": sender,
                "text": text,
                "time_stamp": now_ms(),
            }
        )
        + "\n"
    )


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/provenance/nodes/{user_id}/{chat_id}", response_class=JSONResponse)
async def get_provenance_nodes(request: Request, user_id: str, chat_id: str):
    """
    Return all nodes of the provenance graph for a given user and chat.
    """
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


@app.get("/execute_code/{user_id}/{chat_id}")
async def execute_code(user_id: str, chat_id: str):
    """
    Endpoint to trigger execution of Python code (S) on target tables (T) for a given user and chat.
    """
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
        logger.error(f"Error during code execution: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during code execution.",
        )


@app.post("/download_chat_pdf")
async def download_chat_pdf(data: dict):
    from weasyprint import HTML

    model = data["model"]
    messages = data["messages"]
    chat_id = data["chat_id"]

    html_messages = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else model.capitalize()
        color = "#f2f2f2" if msg["role"] == "user" else "#e8f0fe"
        content_html = markdown_2(msg["content"])
        html_messages += f"""
            <div style="margin-bottom: 16px; padding: 10px; border-radius: 10px; background-color: {color}">
                <strong>{role}:</strong><br>{content_html}
            </div>
        """

    full_html = f"""
    <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: sans-serif;
                    margin: 40px;
                    background-color: #ffffff;
                }}
                h1 {{
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <h2>Chat Transcript</h2>
            <h3>Chat ID: {chat_id}</h3>
            {html_messages}
        </body>
    </html>
    """

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        HTML(string=full_html).write_pdf(tmp_file.name)
        return FileResponse(
            tmp_file.name, filename=f"chat_{chat_id}.pdf", media_type="application/pdf"
        )


@app.post("/combined/html/{user_id}/{chat_id}", response_class=HTMLResponse)
async def read_combined_html(request: Request, user_id: str, chat_id: str, data: dict):
    conductor = session_manager.get_chat_session(user_id, chat_id).conductor
    state = conductor.state.get_current_state_instance(config.TABLE_MAX_ROWS_DISPLAY)

    prov_steps: list[str] = ["<strong>T</strong> is not materialized yet."]
    if conductor.state.is_T_materialized:
        prov_explanation_steps_markdown = (
            conductor.materializer.prov_graph.get_graph_explanation()
        )
        if prov_explanation_steps_markdown:
            prov_steps = [
                markdown(
                    step_md,
                    extensions=["fenced_code", "sane_lists"],
                )
                for step_md in prov_explanation_steps_markdown
            ]
        else:
            prov_steps = ["No materialization steps recorded for <strong>T</strong>."]

    messages = data.get("messages", [])
    model = data.get("model", "assistant")

    retrieved_tables = {
        doc.doc_id: serialize_dataframe(doc.content, config.TABLE_MAX_ROWS_DISPLAY)
        for doc in conductor.retrieved_tables
    }

    return templates.TemplateResponse(
        "state_view.html",
        {
            "request": request,
            "state": state,
            "prov_steps": prov_steps,
            "user_id": user_id,
            "chat_id": chat_id,
            "model": model,
            "messages": messages,
            "retrieved_tables": retrieved_tables,
        },
    )


@app.post("/chat")
async def chat(request: Request):
    body: dict[str, Any] = await request.json()
    user_id: str = body.get("user_id", "default_user")
    chat_id: str = body.get("chat_id", "default_chat")
    messages = body.get("messages", [])
    files = body.get("files", [])

    is_title_generation_task = len(messages) > 0 and messages[0].get(
        "content"
    ).startswith(TITLE_GENERATION_PREFIX)
    if is_title_generation_task:
        return {}

    llm_messages: list[LLMMessage] = []
    for msg in messages:
        llm_messages.append(LLMMessage(role=msg["role"], content=msg["content"]))

    chat_session = session_manager.get_chat_session(user_id, chat_id)

    async def event_stream():
        start = datetime.now().timestamp()

        yield stream_payload("log", "Pneuma connected. Starting processing...")
        await sleep(0)

        response_queue: Queue[str | None] = Queue()

        def run_chat():
            try:
                for msg in chat_session.chat(llm_messages, files):
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
                for msg in chat_session.chat(llm_messages, files):
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
                    f"[MEM] tracemalloc peak Python alloc: {peak / 1024 / 1024:.2f} MB"
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
                    break
        finally:
            chat_session.persist_session()
            producer.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
    )


@app.get("/all_tables/{user_id}/{chat_id}")
def download_all_tables(user_id: str, chat_id: str):
    """
    Download all tables (CSV files) for a given user and chat as a ZIP file.
    Example: /tables/u123/c45/all
    """

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


@app.get("/materializer_code/{user_id}/{chat_id}")
def download_materializer_code(user_id: str, chat_id: str):
    """
    Downloads Materializer code (.py) generated for a given user and chat.
    """
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
