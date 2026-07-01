import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument, Table
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import LLMOption
from pneuma_seeker.shared.schemas.language_model.role import Role
from pneuma_seeker.shared.str_processor import clean_column_table_name

_INDEX_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "indices", "pneuma"
)


def _attr_index_dir(dataset: str) -> str:
    return os.path.join(_INDEX_BASE, f"attribute-index-{dataset}")


def _fulltext_corpus_path(dataset: str) -> str:
    return os.path.join(_INDEX_BASE, f"fulltext-index-{dataset}", "corpus.jsonl")


def index_attribute(
    dataset: str,
    language_model_api: LanguageModelAPI,
    overwrite: bool = False,
) -> None:
    out_dir = _attr_index_dir(dataset)
    nodes_path = os.path.join(out_dir, "attr_nodes.json")
    embs_path = os.path.join(out_dir, "attr_embeddings.npy")

    if os.path.exists(nodes_path) and os.path.exists(embs_path) and not overwrite:
        print(
            f"[ATTRIBUTE INDEX] Index for dataset {dataset} already exists. Skipping."
        )
        return

    corpus_path = _fulltext_corpus_path(dataset)
    if not os.path.exists(corpus_path):
        print(f"[ATTRIBUTE INDEX] corpus.jsonl not found at {corpus_path}. Skipping.")
        return

    nodes: list[dict] = []
    descriptions: list[str] = []
    empty_desc_count = 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Processing {dataset}"):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            table_key = entry["metadata"]["table"]
            if "_SEP_contents_SEP_schema-" not in table_key:
                continue

            table_raw = table_key.split("_SEP_")[0]
            table_name = clean_column_table_name(Path(table_raw).stem)

            for pair in entry["text"].split(" | "):
                sep_idx = pair.find(": ")
                if sep_idx == -1:
                    continue
                raw_col = pair[:sep_idx].strip()
                desc = pair[sep_idx + 2 :].strip()
                col_name = clean_column_table_name(raw_col)
                if not desc:
                    # Fall back to column name as description to avoid empty-string embed errors
                    empty_desc_count += 1
                    desc = raw_col if raw_col else col_name
                nodes.append({"table": table_name, "col": col_name, "desc": desc})
                descriptions.append(desc)

    if empty_desc_count:
        print(
            f"[ATTRIBUTE INDEX] {empty_desc_count} columns had empty descriptions"
            f" for dataset {dataset} — using column name as fallback."
        )

    if not nodes:
        print(
            f"[ATTRIBUTE INDEX] No schema entries found for dataset {dataset}."
            f" Falling back to column names extracted from row entries."
        )
        nodes, descriptions = _extract_cols_from_rows(corpus_path)
        if not nodes:
            print(
                f"[ATTRIBUTE INDEX] No row entries found either. Skipping dataset {dataset}."
            )
            return

    print(
        f"[ATTRIBUTE INDEX] Embedding {len(nodes)} attributes for dataset {dataset}..."
    )
    embeddings = language_model_api.encode(descriptions).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    with open(nodes_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False)
    np.save(embs_path, embeddings)
    print(f"[ATTRIBUTE INDEX] Saved to {out_dir}")


def retrieve_attribute(
    queries: list[str],
    tables: list[Table],
    dataset: str,
    language_model_api: LanguageModelAPI,
    alpha: float = 0.85,
    sim_threshold: float = 0.6,
    attr_threshold: float = -1.0,
    use_llm_threshold: bool = False,
    verbose: bool = True,
) -> list[Table]:
    """
    Tables whose every column falls below the threshold are removed from the result
    (they are not just column-filtered — they are dropped entirely).
    """
    index_dir = _attr_index_dir(dataset)
    nodes_path = os.path.join(index_dir, "attr_nodes.json")
    embs_path = os.path.join(index_dir, "attr_embeddings.npy")

    if not os.path.exists(nodes_path) or not os.path.exists(embs_path):
        if verbose:
            print(
                f"[ATTRIBUTE RETRIEVE] Index missing for dataset {dataset}. Skipping compaction."
            )
        return tables

    with open(nodes_path, "r", encoding="utf-8") as f:
        all_nodes: list[dict] = json.load(f)
    all_embeddings: np.ndarray = np.load(embs_path)

    table_ids = {t.doc_id for t in tables}
    mask = np.array([n["table"] in table_ids for n in all_nodes])
    if not mask.any():
        return tables

    sub_nodes = [n for n, m in zip(all_nodes, mask) if m]
    sub_embs = all_embeddings[mask]  # (M, D)
    M = len(sub_nodes)

    # Resolve auto threshold (attr_threshold < 0 → 0.75 × uniform baseline)
    uniform_baseline = 1.0 / M
    if attr_threshold < 0:
        attr_threshold = 0.75 * uniform_baseline

    # Personalization vector: max-pool cosine similarity across queries
    query_embs = language_model_api.encode(queries).astype(np.float32)  # (Q, D)
    sub_norm = sub_embs / (np.linalg.norm(sub_embs, axis=1, keepdims=True) + 1e-9)
    q_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-9)
    sim_qattr = q_norm @ sub_norm.T  # (Q, M)
    v = sim_qattr.max(axis=0)  # (M,) — max over queries
    v = np.clip(v, 0, None)
    v_sum = v.sum()
    if v_sum > 0:
        v /= v_sum
    else:
        v = np.ones(M, dtype=np.float32) / M

    # Build row-stochastic adjacency matrix
    sim_attr = sub_norm @ sub_norm.T  # (M, M)
    A = np.where(sim_attr >= sim_threshold, sim_attr, 0.0).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    row_sums = A.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    np.fill_diagonal(
        A, np.where(zero_rows, 1.0, np.diag(A))
    )  # self-loop for isolated nodes
    row_sums = A.sum(axis=1, keepdims=True)
    A /= row_sums  # row-stochastic

    # PPR power iteration: r = alpha * A^T @ r + (1 - alpha) * v
    r = v.copy()
    iters = 100
    for i in range(iters):
        r_new = alpha * (A.T @ r) + (1 - alpha) * v
        if np.abs(r_new - r).sum() < 1e-6:
            iters = i + 1
            r = r_new
            break
        r = r_new

    if verbose:
        print(
            f"[ATTRIBUTE RETRIEVE] PPR converged after {iters} iterations. "
            f"uniform baseline={uniform_baseline:.4f}, threshold={attr_threshold:.4f}"
        )
        print(f"[ATTRIBUTE RETRIEVE] PPR scores per table:")
        for tid in sorted({n["table"] for n in sub_nodes}):
            tbl_mask_v = np.array([n["table"] == tid for n in sub_nodes])
            tbl_indices_v = np.where(tbl_mask_v)[0]
            scored = sorted(
                [(sub_nodes[j]["col"], r[j]) for j in tbl_indices_v],
                key=lambda x: -x[1],
            )
            col_strs = "  ".join(f"{col}={score:.4f}" for col, score in scored)
            print(f"  [{tid}]  {col_strs}")

    # Select columns per table and build surviving table list
    surviving: list[Table] = []
    for table in tables:
        tbl_mask = np.array([n["table"] == table.doc_id for n in sub_nodes])
        if not tbl_mask.any():
            # Table has no entries in the attribute index — keep it unchanged
            surviving.append(table)
            continue

        tbl_indices = np.where(tbl_mask)[0]
        tbl_nodes = [sub_nodes[j] for j in tbl_indices]
        tbl_scores = r[tbl_indices]

        if use_llm_threshold:
            selected_cols = _llm_binary_search(
                tbl_nodes, tbl_scores, queries, language_model_api
            )
            if not selected_cols:
                if verbose:
                    print(
                        f"[ATTRIBUTE RETRIEVE] Dropping table '{table.doc_id}' — LLM found no relevant columns."
                    )
                continue
        else:
            keep = tbl_scores > attr_threshold
            if not keep.any():
                if verbose:
                    print(
                        f"[ATTRIBUTE RETRIEVE] Dropping table '{table.doc_id}' — all columns below threshold."
                    )
                continue
            selected_cols = {n["col"] for n, k in zip(tbl_nodes, keep) if k}

        # Filter DataFrame columns (keep only those present in the table)
        existing_cols = list(table.content.columns)
        filtered_cols = [c for c in existing_cols if c in selected_cols]
        if not filtered_cols:
            # Attribute index col names don't match DataFrame cols — keep table unchanged
            surviving.append(table)
            continue
        table.content = table.content[filtered_cols]

        # Rebuild column_types metadata to stay in sync
        if "column_types" in table.metadata:
            try:
                col_types: dict = json.loads(table.metadata["column_types"])
                filtered_types = {
                    k: v for k, v in col_types.items() if k in selected_cols
                }
                table.metadata["column_types"] = json.dumps(
                    filtered_types, ensure_ascii=False
                )
            except (json.JSONDecodeError, TypeError):
                pass

        surviving.append(table)

    return surviving


def retrieve_attribute_cosine_only(
    queries: list[str],
    tables: list[Table],
    dataset: str,
    language_model_api: LanguageModelAPI,
    cos_threshold: float = 0.5,
    verbose: bool = True,
) -> list[Table]:
    """
    Cosine-only baseline: keep columns whose mean cosine similarity across all
    queries exceeds cos_threshold. Tables where every column is below the
    threshold are dropped entirely (same table-dropping behaviour as PPR).
    """
    index_dir = _attr_index_dir(dataset)
    nodes_path = os.path.join(index_dir, "attr_nodes.json")
    embs_path = os.path.join(index_dir, "attr_embeddings.npy")

    if not os.path.exists(nodes_path) or not os.path.exists(embs_path):
        if verbose:
            print(
                f"[COSINE RETRIEVE] Index missing for dataset {dataset}. Skipping compaction."
            )
        return tables

    with open(nodes_path, "r", encoding="utf-8") as f:
        all_nodes: list[dict] = json.load(f)
    all_embeddings: np.ndarray = np.load(embs_path)

    table_ids = {t.doc_id for t in tables}
    mask = np.array([n["table"] in table_ids for n in all_nodes])
    if not mask.any():
        return tables

    sub_nodes = [n for n, m in zip(all_nodes, mask) if m]
    sub_embs = all_embeddings[mask]  # (M, D)

    # Mean cosine similarity across all queries (equal weight = 1/Q each)
    query_embs = language_model_api.encode(queries).astype(np.float32)  # (Q, D)
    sub_norm = sub_embs / (np.linalg.norm(sub_embs, axis=1, keepdims=True) + 1e-9)
    q_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-9)
    scores = (q_norm @ sub_norm.T).mean(axis=0)  # (M,) — uniform average over queries

    if verbose:
        print(
            f"[COSINE RETRIEVE] Scores per table (threshold={cos_threshold}):"
        )
        for tid in sorted({n["table"] for n in sub_nodes}):
            tbl_mask_v = np.array([n["table"] == tid for n in sub_nodes])
            tbl_indices_v = np.where(tbl_mask_v)[0]
            scored = sorted(
                [(sub_nodes[j]["col"], scores[j]) for j in tbl_indices_v],
                key=lambda x: -x[1],
            )
            col_strs = "  ".join(f"{col}={score:.4f}" for col, score in scored)
            print(f"  [{tid}]  {col_strs}")

    surviving: list[Table] = []
    for table in tables:
        tbl_mask = np.array([n["table"] == table.doc_id for n in sub_nodes])
        if not tbl_mask.any():
            surviving.append(table)
            continue

        tbl_indices = np.where(tbl_mask)[0]
        tbl_nodes = [sub_nodes[j] for j in tbl_indices]
        tbl_scores = scores[tbl_indices]

        keep = tbl_scores > cos_threshold
        if not keep.any():
            if verbose:
                print(
                    f"[COSINE RETRIEVE] Dropping table '{table.doc_id}' — all columns below threshold."
                )
            continue
        selected_cols = {n["col"] for n, k in zip(tbl_nodes, keep) if k}

        existing_cols = list(table.content.columns)
        filtered_cols = [c for c in existing_cols if c in selected_cols]
        if not filtered_cols:
            surviving.append(table)
            continue
        table.content = table.content[filtered_cols]

        if "column_types" in table.metadata:
            try:
                col_types: dict = json.loads(table.metadata["column_types"])
                filtered_types = {
                    k: v for k, v in col_types.items() if k in selected_cols
                }
                table.metadata["column_types"] = json.dumps(
                    filtered_types, ensure_ascii=False
                )
            except (json.JSONDecodeError, TypeError):
                pass

        surviving.append(table)

    return surviving


def _extract_cols_from_rows(corpus_path: str) -> tuple[list[dict], list[str]]:
    """Fallback for corpora with no schema entries: extract column names from the first row per table."""
    nodes: list[dict] = []
    descriptions: list[str] = []
    seen_tables: set[str] = set()

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            table_key = entry["metadata"]["table"]
            if "_SEP_contents_SEP_row-" not in table_key:
                continue

            table_raw = table_key.split("_SEP_")[0]
            table_name = clean_column_table_name(Path(table_raw).stem)
            if table_name in seen_tables:
                continue
            seen_tables.add(table_name)

            # Take the first row in the cell (rows separated by " || ")
            first_row = entry["text"].split(" || ")[0]
            for pair in first_row.split(" | "):
                sep_idx = pair.find(": ")
                if sep_idx == -1:
                    continue
                raw_col = pair[:sep_idx].strip()
                col_name = clean_column_table_name(raw_col)
                desc = raw_col if raw_col else col_name
                nodes.append({"table": table_name, "col": col_name, "desc": desc})
                descriptions.append(desc)

    return nodes, descriptions


def _llm_binary_search(
    nodes: list[dict],
    scores: np.ndarray,
    queries: list[str],
    language_model_api: LanguageModelAPI,
) -> set[str]:
    order = np.argsort(-scores)
    sorted_nodes = [nodes[i] for i in order]
    query_str = "; ".join(queries)

    def is_relevant(node: dict) -> bool:
        prompt = (
            f"Is column '{node['col']}' (description: '{node['desc']}') "
            f"relevant to answering: '{query_str}'? Reply yes or no."
        )
        messages = [LLMMessage(role=Role.USER.value, content=prompt)]
        resp = "".join(language_model_api.chat(messages, LLMOption(max_new_tokens=1)))
        return resp.strip().lower().startswith("y")

    lo, hi = 0, len(sorted_nodes) - 1
    cutoff = len(sorted_nodes)
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_relevant(sorted_nodes[mid]):
            cutoff = mid + 1
            lo = mid + 1
        else:
            hi = mid - 1

    return (
        {n["col"] for n in sorted_nodes[:cutoff]}
        if cutoff > 0
        else {n["col"] for n in sorted_nodes}
    )
