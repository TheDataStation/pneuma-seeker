# src/pneuma_seeker/main.py
import re
from asyncio import create_task, sleep
from datetime import datetime
from io import BytesIO, StringIO
from json import dumps, loads
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

import numpy as np
import pandas as pd

from pneuma_seeker.session_manager import SessionManager
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import LLMOption
from pneuma_seeker.shared.table_serializer import serialize_dataframe

def _compute_col_stats_from_db(db_api, user_id: str, chat_id: str, doc) -> dict:
    """Compute full-dataset column statistics for step-1 visualization.

    Returns a dict  {col_name: {total_count, unique_count, null_count,
    is_numeric, is_integer, min, max, histogram, histogram_edges,
    fine_histogram, fine_edges, freq}}  ready for the frontend
    buildColStatsHtml() function.
    """
    col_stats: dict = {}
    try:
        cols = list(doc.content.columns)
        path = doc.path

        # ── 1. Aggregate: total + unique/null count per column ──────────────
        agg_parts = ["COUNT(*) AS __total__"]
        for col in cols:
            safe = f'"{col}"'
            agg_parts.append(f"COUNT(DISTINCT {safe}) AS \"__uniq__{col}\"")
            agg_parts.append(f"(COUNT(*) - COUNT({safe})) AS \"__null__{col}\"")
        agg_df = db_api.execute_query(
            user_id, chat_id, f"SELECT {', '.join(agg_parts)} FROM {path}"
        )
        total = int(agg_df["__total__"].iloc[0])

        for col in cols:
            uniq = int(agg_df[f"__uniq__{col}"].iloc[0])
            null_c = int(agg_df[f"__null__{col}"].iloc[0])
            is_boolean = pd.api.types.is_bool_dtype(doc.content[col])
            # booleans are a subclass of int in pandas — exclude them from numeric
            is_numeric = pd.api.types.is_numeric_dtype(doc.content[col]) and not is_boolean
            is_integer = pd.api.types.is_integer_dtype(doc.content[col]) and not is_boolean

            stats: dict = {
                "total_count": total,
                "unique_count": uniq,
                "null_count": null_c,
                "is_boolean": is_boolean,
                "is_numeric": is_numeric,
                "is_integer": is_integer,
            }

            safe_col = f'"{col}"'
            if is_boolean:
                try:
                    freq_df = db_api.execute_query(
                        user_id, chat_id,
                        f"SELECT {safe_col} AS val, COUNT(*) AS cnt"
                        f" FROM {path} WHERE {safe_col} IS NOT NULL"
                        f" GROUP BY {safe_col} ORDER BY val",
                    )
                    stats["freq"] = {
                        str(r["val"]): int(r["cnt"]) for _, r in freq_df.iterrows()
                    }
                except Exception:
                    pass

            elif is_numeric:
                try:
                    mm_df = db_api.execute_query(
                        user_id, chat_id,
                        f"SELECT MIN({safe_col}) AS mn, MAX({safe_col}) AS mx"
                        f" FROM {path} WHERE {safe_col} IS NOT NULL",
                    )
                    mn = float(mm_df["mn"].iloc[0])
                    mx = float(mm_df["mx"].iloc[0])
                    stats["min"] = mn
                    stats["max"] = mx

                    # Fetch column values for numpy histogram
                    raw_df = db_api.execute_query(
                        user_id, chat_id,
                        f"SELECT {safe_col} AS v FROM {path} WHERE {safe_col} IS NOT NULL",
                    )
                    vals = raw_df["v"].dropna().astype(float).values

                    if mn < mx and len(vals) > 0:
                        counts, edges = np.histogram(vals, bins=10)
                        stats["histogram"] = counts.tolist()
                        stats["histogram_edges"] = edges.tolist()
                        # Fine histogram for drag-to-zoom
                        fine_bins = min(100, max(10, uniq))
                        if fine_bins > 10:
                            fc, fe = np.histogram(vals, bins=fine_bins)
                            stats["fine_histogram"] = fc.tolist()
                            stats["fine_edges"] = fe.tolist()
                    else:
                        stats["histogram"] = [total - null_c]
                        stats["histogram_edges"] = [mn, mx if mx != mn else mn + 1]
                except Exception:
                    pass

            elif uniq <= 12:
                try:
                    freq_df = db_api.execute_query(
                        user_id, chat_id,
                        f"SELECT {safe_col} AS val, COUNT(*) AS cnt"
                        f" FROM {path} WHERE {safe_col} IS NOT NULL"
                        f" GROUP BY {safe_col} ORDER BY cnt DESC LIMIT 12",
                    )
                    stats["freq"] = {
                        str(r["val"]): int(r["cnt"]) for _, r in freq_df.iterrows()
                    }
                except Exception:
                    pass

            col_stats[col] = stats

    except Exception:
        pass

    return col_stats


app = FastAPI(title="Pneuma-Seeker")
logger = setup_logger("Core Service")
config = Config("../../.env")
session_manager = SessionManager(
    config,
    logger,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = (
    Path(__file__).resolve().parents[2]
)  # go up from /src/pneuma_seeker/main.py → project root
TABLES_DIR = BASE_DIR / "data_src" / "target_tables"


def _load_tables_metadata(retrieved_docs, base_dir: Path) -> dict[str, str]:
    """Map each retrieved table to a combined metadata string.

    Combines two sources per table:
    - A short description from ``data_src/{domain}/metadata.csv`` (if present).
    - A longer context block from the best-matching ``.txt`` / ``.text`` file
      in ``metadata/{domain}/``, matched by keyword overlap with the table name.
    """
    import csv as _csv

    _STOP = {"data", "info", "the", "and", "of", "for", "by", "in"}

    # Cache: domain → (short_desc_dict, context_entries)
    domain_cache: dict[str, tuple[dict[str, str], list[tuple[set, str]]]] = {}

    result: dict[str, str] = {}
    for doc in retrieved_docs:
        domain: str = doc.metadata.get("dataset_name", "")
        if not domain:
            continue

        if domain not in domain_cache:
            # Short descriptions from data_src/{domain}/metadata.csv
            short_descs: dict[str, str] = {}
            csv_path = base_dir / "data_src" / domain / "metadata.csv"
            if csv_path.is_file():
                with csv_path.open(encoding="utf-8") as fh:
                    for row in _csv.DictReader(fh):
                        t = row.get("table_name", "").strip()
                        d = row.get("description", "").strip()
                        if t and d:
                            short_descs[t] = d

            # Longer context texts from metadata/{domain}/*.txt|.text
            context_entries: list[tuple[set, str]] = []
            meta_dir = base_dir / "metadata" / domain
            if meta_dir.is_dir():
                for f in meta_dir.iterdir():
                    if f.suffix.lower() in (".txt", ".text", ".md") and f.is_file():
                        words = set(re.sub(r"[_\-]", " ", f.stem).lower().split()) - _STOP
                        text = f.read_text(encoding="utf-8", errors="replace").strip()
                        if words and text:
                            context_entries.append((words, text))

            domain_cache[domain] = (short_descs, context_entries)

        short_descs, context_entries = domain_cache[domain]

        # Best-matching context text
        table_words = set(re.sub(r"[_\-]", " ", doc.doc_id).lower().split()) - _STOP
        best_score, best_context = 0, None
        for (kws, text) in context_entries:
            score = len(kws & table_words)
            if score > best_score:
                best_score, best_context = score, text

        short_desc = short_descs.get(doc.doc_id, "")
        parts: list[str] = []
        if short_desc:
            parts.append(short_desc)
        if best_context:
            parts.append(best_context)
        if parts:
            result[doc.doc_id] = "\n\n".join(parts)

    return result

templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent / "templates")
)

TITLE_GENERATION_PREFIX = "### Task:\nGenerate a concise, 3-5 word title with an emoji summarizing the chat history.\n### Guidelines:\n- The title should clearly represent the main theme or subject of the conversation.\n- Use emojis that enhance understanding of the topic, but avoid quotation marks or special formatting."

_SIDEBAR_BLOCK_RE = re.compile(
    r"```html\s*<!--\s*PNEUMA_STATE_START\s*-->[\s\S]*?<!--\s*PNEUMA_STATE_END\s*-->\s*```",
    re.DOTALL,
)


def _strip_sidebar_blocks(messages: list[dict]) -> list[dict]:
    """Remove embedded sidebar HTML blocks from assistant messages."""
    result = []
    for msg in messages:
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            content = _SIDEBAR_BLOCK_RE.sub("", msg["content"]).strip()
            result.append({**msg, "content": content})
        else:
            result.append(msg)
    return result


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


@app.post("/execute_custom_code/{user_id}/{chat_id}")
async def execute_custom_code(user_id: str, chat_id: str, data: dict):
    """
    Execute a user-supplied script without persisting it to conductor state.
    """
    script = data.get("script", "").strip()
    if not script:
        raise HTTPException(status_code=400, detail="No script provided.")
    conductor = session_manager.get_chat_session(user_id, chat_id).conductor
    try:
        execution_result = conductor.action_set.execute_code(script, "conductor_s_execution")
        return serialize_dataframe(execution_result, config.TABLE_MAX_ROWS_DISPLAY)
    except Exception as e:
        logger.error(f"Error during custom code execution: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during code execution.")


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
    # Infer dataset from model name so the correct session is created even after a restart
    model_name = (data.get("model") or "").lower()
    inferred_dataset = next((ds for ds in config.DATA_SOURCES if ds in model_name), None)
    chat_session = session_manager.get_chat_session(user_id, chat_id, dataset=inferred_dataset)
    conductor = chat_session.conductor
    dataset = chat_session.config.DATA_SOURCES[0] if chat_session.config.DATA_SOURCES else None
    dataset_description: str | None = None
    dataset_sources: str | None = None
    if dataset:
        general_txt = BASE_DIR / "metadata" / dataset / "general.txt"
        if general_txt.is_file():
            raw = general_txt.read_text(encoding="utf-8", errors="replace")
            desc_match = re.search(r"Descrip[a-z]*:\s*(.*?)(?=\nSources:|$)", raw, re.DOTALL | re.IGNORECASE)
            src_match = re.search(r"Sources:\s*(.*)", raw, re.DOTALL | re.IGNORECASE)
            if desc_match:
                dataset_description = desc_match.group(1).strip()
            if src_match:
                dataset_sources = src_match.group(1).strip()
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

    messages = _strip_sidebar_blocks(data.get("messages", []))
    model = data.get("model", "assistant")

    retrieved_tables = {
        doc.doc_id: serialize_dataframe(doc.content, config.TABLE_MAX_ROWS_DISPLAY)
        for doc in conductor.retrieved_tables
    }

    retrieved_tables_col_stats: dict = {}
    for doc in conductor.retrieved_tables:
        try:
            retrieved_tables_col_stats[doc.doc_id] = _compute_col_stats_from_db(
                conductor.db_api, user_id, chat_id, doc
            )
        except Exception:
            pass

    tables_metadata = _load_tables_metadata(conductor.retrieved_tables, BASE_DIR)

    return templates.TemplateResponse(
        "explanation.html",
        {
            "request": request,
            "state": state,
            "prov_steps": prov_steps,
            "user_id": user_id,
            "chat_id": chat_id,
            "model": model,
            "messages": messages,
            "retrieved_tables": retrieved_tables,
            "retrieved_tables_col_stats": retrieved_tables_col_stats,
            "tables_metadata": tables_metadata,
            "backend_url": str(request.base_url).rstrip("/"),
            "dataset": dataset,
            "dataset_description": dataset_description,
            "dataset_sources": dataset_sources,
        },
    )


@app.post("/chat")
async def chat(request: Request):
    body: dict[str, Any] = await request.json()
    user_id: str = body.get("user_id", "default_user")
    chat_id: str = body.get("chat_id", "default_chat")
    dataset: str | None = body.get("dataset", None)
    messages = _strip_sidebar_blocks(body.get("messages", []))
    files = body.get("files", [])

    is_title_generation_task = len(messages) > 0 and messages[0].get(
        "content"
    ).startswith(TITLE_GENERATION_PREFIX)
    if is_title_generation_task:
        return {}

    llm_messages: list[LLMMessage] = []
    for msg in messages:
        llm_messages.append(LLMMessage(role=msg["role"], content=msg["content"]))

    chat_session = session_manager.get_chat_session(user_id, chat_id, dataset=dataset)

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
                        payload = (
                            dumps({
                                "sender": "done",
                                "text": f"Processing done in {datetime.now().timestamp() - start:.2f}s.",
                                "time_stamp": now_ms(),
                                "show_sidebar": bool(chat_session.conductor.retrieved_tables),
                            })
                            + "\n"
                        )
                    else:
                        payload = stream_payload("assistant", response)

                    yield payload
                    await sleep(0)
                except Exception:
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


@app.get("/download_table/{user_id}/{chat_id}/{doc_id}")
def download_retrieved_table(user_id: str, chat_id: str, doc_id: str):
    """Download a full retrieved table as CSV by doc_id."""
    conductor = session_manager.get_chat_session(user_id, chat_id).conductor
    doc = next((d for d in conductor.retrieved_tables if d.doc_id == doc_id), None)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Table '{doc_id}' not found")
    try:
        df = conductor.db_api.execute_query(user_id, chat_id, f"SELECT * FROM {doc.path}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load table: {exc}")
    buf = StringIO()
    df.to_csv(buf, index=False)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={doc_id}.csv"},
    )


@app.get("/download_target_table/{user_id}/{chat_id}/{table_id}")
def download_target_table(user_id: str, chat_id: str, table_id: str):
    """Download a full target table as CSV by table_id."""
    conductor = session_manager.get_chat_session(user_id, chat_id).conductor
    try:
        df = conductor.db_api.execute_query(user_id, chat_id, f"SELECT * FROM {table_id}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load table: {exc}")
    buf = StringIO()
    df.to_csv(buf, index=False)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={table_id}.csv"},
    )


@app.get("/prov_charts/{user_id}/{chat_id}")
async def get_prov_charts(user_id: str, chat_id: str):
    """
    Generate Mermaid diagram charts (table-level + column-level) from the
    provenance graph using LLM, returned as JSON for client-side rendering.
    """
    chat_session = session_manager.get_chat_session(user_id, chat_id)
    conductor = chat_session.conductor
    prov_graph = conductor.materializer.prov_graph

    if not conductor.state.is_T_materialized:
        return JSONResponse(content={"svg": "", "col_flow": {}})

    ordered_nodes = prov_graph.topological_sort()
    mat_nodes = [
        n
        for n in ordered_nodes
        if getattr(n.source_retriever, "value", str(n.source_retriever)) == "Materializer"
    ]

    source_tables_info = {}
    for doc in conductor.retrieved_tables:
        if hasattr(doc.content, "columns"):
            source_tables_info[doc.doc_id] = list(doc.content.columns)[:15]

    target_tables_info = {}
    for tid, df in (conductor.state.T or {}).items():
        target_tables_info[tid] = list(df.columns)[:15]

    # Assign BFS levels to mat_nodes (processed in topological order already)
    mat_id_set = {id(n) for n in mat_nodes}
    node_level: dict[int, int] = {}
    for node in mat_nodes:
        parent_levels = [node_level[id(p)] for p in node.parents if id(p) in mat_id_set and id(p) in node_level]
        node_level[id(node)] = (max(parent_levels) + 1) if parent_levels else 0

    from collections import defaultdict
    level_groups: dict[int, list] = defaultdict(list)
    for node in mat_nodes:
        level_groups[node_level[id(node)]].append(node)
    n_op_stages = (max(level_groups.keys()) + 1) if level_groups else 0

    # Build per-step descriptions grouped by stage
    step_id = 0
    stage_descs = []  # list of (stage_label, [(step_id, node), ...])
    for lvl in sorted(level_groups.keys()):
        nodes_in_lvl = level_groups[lvl]
        entries = []
        for node in nodes_in_lvl:
            step_id += 1
            code_snippet = (node.python_code or "")[:400]
            entries.append((step_id, node.description, code_snippet))
        parallel = len(nodes_in_lvl) > 1
        stage_descs.append((lvl, parallel, entries))

    # Build steps_text with explicit parallel/sequential annotation
    steps_lines = []
    for lvl, parallel, entries in stage_descs:
        if parallel:
            steps_lines.append(f"[STAGE {lvl+1} — PARALLEL: these {len(entries)} steps run simultaneously]")
        else:
            steps_lines.append(f"[STAGE {lvl+1} — SEQUENTIAL]")
        for sid, desc, code in entries:
            steps_lines.append(f"  Step {sid}: {desc}\n  Code:\n{code}")
    steps_text = "\n\n".join(steps_lines)

    # Pre-compute SVG layout constants
    BOX_W, BOX_H, H_GAP, V_GAP = 160, 200, 40, 12
    # Stages: stage 0 = sources, stages 1..n_op_stages = ops, stage n_op_stages+1 = result
    n_stages = 1 + n_op_stages + 1
    n_src = max(len(source_tables_info), 1)
    max_parallel_ops = max((len(v) for v in level_groups.values()), default=1)
    max_parallel = max(n_src, max_parallel_ops, 1)
    SVG_W = n_stages * BOX_W + (n_stages - 1) * H_GAP + 2
    SVG_H = max_parallel * BOX_H + (max_parallel - 1) * V_GAP + 20

    # Pre-compute box (x, y) for each slot: (stage_idx, row_idx)
    def box_xy(stage: int, row: int, total_rows: int) -> tuple[int, int]:
        bx = stage * (BOX_W + H_GAP)
        # Center rows vertically in the SVG
        group_h = total_rows * BOX_H + (total_rows - 1) * V_GAP
        by = (SVG_H - group_h) // 2 + row * (BOX_H + V_GAP)
        return bx, by

    # Build explicit box coordinate table for the prompt
    box_coord_lines = []
    # Source boxes
    for i, tname in enumerate(list(source_tables_info.keys()) or ["(source)"]):
        bx, by = box_xy(0, i, n_src)
        box_coord_lines.append(f"  Source '{tname}': stage=0 row={i}  →  x={bx} y={by} width={BOX_W} height={BOX_H}")
    # Op boxes
    for lvl, parallel, entries in stage_descs:
        n_rows = len(entries)
        for row, (sid, desc, _) in enumerate(entries):
            bx, by = box_xy(lvl + 1, row, n_rows)
            ptag = " [PARALLEL]" if parallel else ""
            box_coord_lines.append(f"  Step {sid}{ptag}: stage={lvl+1} row={row}  →  x={bx} y={by} width={BOX_W} height={BOX_H}")
    # Result box
    res_name = list(target_tables_info.keys())[0] if target_tables_info else "result"
    rbx, rby = box_xy(n_stages - 1, 0, 1)
    box_coord_lines.append(f"  Result '{res_name}': stage={n_stages-1} row=0  →  x={rbx} y={rby} width={BOX_W} height={BOX_H}")

    prompt = (
        "You are a data visualization expert. Generate a complete SVG pipeline diagram.\n\n"
        f"Source Tables (name → columns):\n{dumps(source_tables_info, indent=2)}\n\n"
        f"Result Tables (name → columns):\n{dumps(target_tables_info, indent=2)}\n\n"
        f"Pipeline Structure (stages are SEQUENTIAL left-to-right; steps within the same stage are PARALLEL):\n{steps_text}\n\n"
        "=== SVG LAYOUT ===\n"
        f"SVG: width={SVG_W} height={SVG_H} xmlns='http://www.w3.org/2000/svg'\n"
        "Use these EXACT box coordinates (x, y, width, height) — do not change them:\n"
        + "\n".join(box_coord_lines)
        + "\n\nBox colors:\n"
        "  Source boxes:    fill=#e8f0fe  stroke=#93b4f7  title-color=#1e3a8a\n"
        "  Operation boxes: fill=#fef3c7  stroke=#fcd34d  title-color=#92400e\n"
        "  Result box:      fill=#d1fae5  stroke=#6ee7b7  title-color=#065f46\n"
        "  All boxes: rx=6, stroke-width=1.5\n\n"
        "Box content:\n"
        "  Title: x=box_x+(BOX_W/2), y=box_y+20, text-anchor=middle, font-size=11, font-weight=bold, font-family=system-ui\n"
        "  Column pills: start at y=box_y+36, 2 per row, row height=16\n"
        f"    Left pill:  x=box_x+6,  width=70  |  Right pill: x=box_x+82, width=70\n"
        "    Pill height=13, rx=3; pill text: font-size=8, font-family=monospace, centered, truncate to 10 chars\n"
        "    Show max 6 pills; add '+N more' text (font-size=8, fill=#9ca3af) if truncated\n"
        "  Operation box sections:\n"
        "    Top section — input_cols: pill fill=#e0e7ff, text=#3730a3\n"
        "    Dashed separator line: y=box_y+110, x1=box_x+6, x2=box_x+154, stroke=#fcd34d, stroke-dasharray=3,2\n"
        "    Bottom section — output_cols: pill fill=#fef9c3, text=#78350f, start at y=box_y+116\n\n"
        "Arrows (use marker id='arr'):\n"
        "  <defs><marker id='arr' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'>"
        "<path d='M0,0 L8,4 L0,8 Z' fill='#9ca3af'/></marker></defs>\n"
        "  Draw arrows based on data dependencies:\n"
        "  - Each source table box → the operation box(es) that use it (right edge of source to left edge of op)\n"
        "  - Each operation box → the next operation(s) that consume its output\n"
        "  - The final operation box(es) → the result box\n"
        "  - For parallel steps in the same stage: their inputs may come from different prior boxes\n"
        "  - Arrow: <line x1='...' y1='...' x2='...' y2='...' stroke='#9ca3af' stroke-width='1.5' marker-end='url(#arr)'/>\n"
        "  - Connect from right-center of source box (x=box_x+BOX_W, y=box_y+BOX_H/2) "
        "to left-center of target box (x=box_x, y=box_y+BOX_H/2)\n\n"
        "Return a JSON object with exactly two keys:\n"
        '1. "svg": the complete SVG element as a string (starting with <svg and ending with </svg>)\n'
        '2. "col_flow": { "sources": [{"table": "name", "cols": [...]}], "ops": ["step label", ...], "result_cols": [...] }\n\n'
        "Return ONLY valid JSON. No markdown fences."
    )

    try:
        response_text = "".join(
            chat_session.language_model_api.chat(
                [LLMMessage(role="user", content=prompt)],
                LLMOption(json_mode=True, max_new_tokens=4000),
            )
        )
        logger.info(f"[prov_charts] LLM response: {response_text[:300]}")
        charts = loads(response_text)
        return JSONResponse(
            content={
                "svg": charts.get("svg", ""),
                "col_flow": charts.get("col_flow", {}),
            }
        )
    except Exception as e:
        logger.error(f"[prov_charts] Error: {e}")
        return JSONResponse(content={"svg": "", "col_flow": {}})


# ---------------------------------------------------------------------------
# Per-operation diagram data: LLM classifies each pipeline step and produces
# a short action label, enabling client-side type-specific diagram rendering.
# ---------------------------------------------------------------------------
@app.get("/prov_col_flow/{user_id}/{chat_id}")
async def get_prov_col_flow(user_id: str, chat_id: str):
    """
    For each operation in the pipeline, ask the LLM to identify:
      - op_type: one of filter|join|aggregate|select|sort|rename|add_column|union|other
      - short_label: ≤60-char human-readable action (e.g. "Keep rows where age > 30")
      - input_tables: list of input table/dataset names consumed by this step
      - output_label: short name for the output (e.g. "filtered employees")
    Returns { "ops": [ { "step": 1, "op_type": "filter", "short_label": "...",
                         "input_tables": ["t1"], "output_label": "..." }, ... ] }
    """
    chat_session = session_manager.get_chat_session(user_id, chat_id)
    conductor = chat_session.conductor
    prov_graph = conductor.materializer.prov_graph

    if not conductor.state.is_T_materialized:
        return JSONResponse(content={"ops": []})

    ordered_nodes = prov_graph.topological_sort()
    mat_nodes = [
        n
        for n in ordered_nodes
        if getattr(n.source_retriever, "value", str(n.source_retriever)) == "Materializer"
    ]

    if not mat_nodes:
        return JSONResponse(content={"ops": []})

    source_table_names = [doc.doc_id for doc in conductor.retrieved_tables]
    result_name = (
        list(conductor.state.T.keys())[0] if conductor.state.T else "result"
    )

    # Build a numbered list of steps with their descriptions and code
    steps_lines = []
    for i, node in enumerate(mat_nodes, 1):
        code_snippet = (node.python_code or "")[:400]
        steps_lines.append(
            f"Step {i}: {node.description or '(no description)'}\nCode:\n{code_snippet}"
        )
    steps_text = "\n\n".join(steps_lines)

    prompt = (
        "You are a data pipeline analyst. For each pipeline step below, return:\n"
        '  "op_type": one of: filter | join | aggregate | select | sort | rename | add_column | union | other\n'
        '  "short_label": ≤60 characters — a plain-English description of the action '
        "(e.g. \"Keep rows where year = 2004\", \"Join on site_id\", \"Group by region, sum count\"). "
        "Do NOT repeat the full description — be concise.\n"
        '  "input_tables": list of inputs consumed by this step. Use:\n'
        "    - The exact source table name (from the source tables list) if this step reads directly from a source table\n"
        "    - The output_label of the PREVIOUS step if this step chains on that step's output (not the literal string 'previous step output')\n"
        '  "output_label": the Python variable name assigned in this step\'s code '
        "(e.g. if code does `filtered_df = ...`, use 'filtered_df'). "
        f"For the last step, always use '{result_name}'.\n\n"
        f"Source tables available: {source_table_names}\n\n"
        f"Pipeline steps:\n{steps_text}\n\n"
        "Return a JSON object with one key 'ops', an array with one entry per step in order:\n"
        '{ "ops": [ { "step": 1, "op_type": "filter", "short_label": "...", '
        '"input_tables": ["table_name"], "output_label": "filtered_df" }, ... ] }\n\n'
        "Return ONLY valid JSON. No markdown fences."
    )

    try:
        response_text = "".join(
            chat_session.language_model_api.chat(
                [LLMMessage(role="user", content=prompt)],
                LLMOption(json_mode=True, max_new_tokens=1500),
            )
        )
        logger.info(f"[prov_col_flow] LLM response: {response_text[:300]}")
        data = loads(response_text)
        return JSONResponse(content={"ops": data.get("ops", [])})
    except Exception as e:
        logger.error(f"[prov_col_flow] Error: {e}")
        return JSONResponse(content={"ops": []})


# ---------------------------------------------------------------------------
# Script annotation: LLM divides a Python script into labeled blocks with
# natural-language descriptions, triggered on demand from the UI.
# ---------------------------------------------------------------------------
@app.post("/annotate_script/{user_id}/{chat_id}")
async def annotate_script(user_id: str, chat_id: str, request: Request):
    """
    Accept a Python script and ask the LLM to divide it into logical blocks,
    each with a short label and a one-sentence natural-language description.
    Returns { "blocks": [{ "code": "...", "label": "...", "desc": "..." }] }.
    """
    body = await request.json()
    script = body.get("script", "").strip()
    if not script:
        return JSONResponse(content={"blocks": []})

    chat_session = session_manager.get_chat_session(user_id, chat_id)

    prompt = (
        "You are explaining a data pipeline to someone who has never written SQL or code.\n\n"
        "Break the following Python script into as many small, atomic steps as needed — "
        "each step should do exactly one thing (e.g. one filter, one join, one aggregation, one rename). "
        "Do not merge multiple operations into one block just to keep the count low.\n\n"
        "For each step return:\n"
        '  "code": the exact lines of code for that step\n'
        '  "label": a 3–5 word plain-English title (no jargon), e.g. "Keep only 2021 records"\n'
        '  "desc": 1–3 sentences of plain English that a non-technical person can understand. '
        "Always be explicit: name every table and column involved in the step using their actual names from the code. "
        "Say what data we start with, what we do to it, and what we end up with. "
        "Avoid words like 'filter', 'query', 'dataframe', 'SQL', 'merge', 'join' — "
        "instead say things like 'we only keep rows where ...', 'we attach information from ... by matching ...', "
        "'we add up all the values in the ... column for each ...'.\n"
        '  "columns": a list of every column name referenced or produced in this step (exact names as they appear in the code)\n'
        '  "tables": a list of every table/variable name read or written in this step (exact names as they appear in the code)\n\n'
        "Return ONLY a valid JSON object with a single key \"blocks\" whose value is an array.\n\n"
        f"Script:\n{script}"
    )

    response_text = ""
    try:
        response_text = "".join(
            chat_session.language_model_api.chat(
                [LLMMessage(role="user", content=prompt)],
                LLMOption(json_mode=True),
            )
        )
        logger.info(f"[annotate_script] raw response: {response_text[:500]}")
        # Strip markdown code fences if present
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        data = loads(cleaned)
        blocks = data.get("blocks", [])
        logger.info(f"[annotate_script] parsed {len(blocks)} blocks")
        chat_session.conductor.annotation_cache[script] = blocks
        return JSONResponse(content={"blocks": blocks})
    except Exception as e:
        logger.error(f"[annotate_script] Error: {e}, response: {response_text!r}")
        return JSONResponse(content={"blocks": [], "error": str(e)})


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
