import gc
import math
import os
import re
import time
from collections import defaultdict
from enum import Enum
from math import ceil
from pathlib import Path
from typing import Optional, cast

import bm25s
import chromadb_deterministic as chromadb
import pandas as pd
import Stemmer
from bm25s.tokenization import convert_tokenized_to_string_list
from chromadb_deterministic.api import ClientAPI
from chromadb_deterministic.api.models.Collection import Collection
from scipy.spatial.distance import cosine
from tiktoken import encoding_for_model
from torch import cuda
from tqdm import tqdm
import json

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.ir_system.retriever.interface import (
    AbstractRetriever,
)
from pneuma_seeker.services.language_model.abstract_model import AbstractModel
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.parser import parse_json
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
    TableContext,
    Text,
)
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import (
    EmbeddingModelOption,
    LLMOption,
)
from pneuma_seeker.shared.schemas.language_model.role import Role
from pneuma_seeker.shared.str_processor import clean_column_table_name


class PneumaRetriever(AbstractRetriever):
    """Represents a tabular data retriever."""

    def __init__(
        self,
        user_id: str,
        chat_id: str,
        config: Config,
        db_api: DBAPI,
        language_model_api: LanguageModelAPI,
    ):
        super().__init__(user_id, chat_id, config, db_api, language_model_api)
        self.hybrid_retriever = HybridRetriever(
            self.language_model_api.llm,
            RerankingMode.NONE,  # Alternative: RerankingMode.LLM
        )
        self.stemmer = Stemmer.Stemmer("english")
        self.index_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "indices", "pneuma"
        )

    @property
    def retriever_type(self) -> RetrieverType:
        """
        Defines the type of the retriever.
        """
        return RetrieverType.PNEUMA_RETRIEVER

    def load(self):
        """
        Currently, we do not need to load the retriever's models.
        """
        pass

    def retrieve(
        self,
        query: str,
        k: int,
        sample_only: bool,
        sample_size: int | None = None,
    ) -> list[AbstractDocument]:
        """
        Retrieves a list of documents given a query.
        """
        try:
            retrieval_results: list[AbstractDocument] = []
            increased_k = k * 5
            self.db_api.link_dataset_tables(
                self.user_id, self.chat_id, self.config.DATA_SOURCES[0]
            )

            client = chromadb.PersistentClient(
                os.path.join(
                    self.index_path, f"vector-index-{self.config.DATA_SOURCES[0]}"
                )
            )
            collection = client.get_collection("benchmark")
            retriever = bm25s.BM25.load(
                os.path.join(
                    self.index_path, f"fulltext-index-{self.config.DATA_SOURCES[0]}"
                ),
                load_corpus=True,
            )

            dictionary_id_bm25 = dict()
            if retriever.corpus is not None:
                if len(retriever.corpus) < increased_k:
                    print(
                        f"Reducing increased_k from {increased_k} to {len(retriever.corpus)}"
                    )
                    increased_k = len(retriever.corpus)
                dictionary_id_bm25 = {
                    datum["metadata"]["table"]: datum_idx
                    for datum_idx, datum in enumerate(retriever.corpus)
                }

            question_embedding = self.language_model_api.embed_model.encode([query])[
                0
            ].tolist()
            query_tokens = bm25s.tokenize(
                query, stemmer=self.stemmer, show_progress=False
            )

            results, scores = retriever.retrieve(
                query_tokens, k=increased_k, show_progress=False
            )
            bm25_res = (results, scores)
            vec_res = collection.query(
                query_embeddings=[question_embedding], n_results=increased_k
            )
            all_nodes = self.hybrid_retriever.retrieve(
                retriever,
                collection,
                bm25_res,
                vec_res,
                increased_k,
                query,
                0.5,
                query_tokens,
                question_embedding,
                dictionary_id_bm25,
            )

            seen_tables: list[str] = []
            final_rank: list[tuple[str, float]] = []
            table_keywords: dict[str, set[str]] = {}
            if self.config.TABLE_RETRIEVE_ENABLE_ENTITIES_RELEVANCE_BOOSTER:
                messages = [
                    LLMMessage(
                        role=Role.SYSTEM.value,
                        content=self.__get_entity_extraction_sys_prompt(),
                    ),
                    LLMMessage(role=Role.USER.value, content=query),
                ]
                entities = parse_json(
                    "".join(
                        self.language_model_api.chat(
                            messages, LLMOption(json_mode=True)
                        )
                    )
                )
                print(f"Extracted entities for relevance boosting: {entities}")
                if "entities" in entities and isinstance(entities["entities"], list):
                    keywords = entities["entities"]
                    if len(keywords) > 0:
                        final_rank, table_keywords = self.__keyword_relevance_by_table(
                            keywords
                        )

            for table, _, _ in all_nodes[:k]:
                table_raw = table.split("_SEP_")[0]
                table_name = clean_column_table_name(Path(table_raw).stem)

                if table_name not in seen_tables:
                    seen_tables.append(table_name)
                else:
                    continue

                query_table = f"""
                SELECT * FROM {self.config.DATA_SOURCES[0]}."{table_name}"
                """
                if sample_only:
                    if sample_size is None or sample_size <= 0:
                        sample_size = 5
                    query_table += f" LIMIT {sample_size}"
                actual_table = self.db_api.execute_query(
                    self.user_id, self.chat_id, query_table
                )

                # Fetch authoritative column types from DuckDB (not pandas sample dtypes).
                duckdb_col_types: dict[str, str] = {}
                try:
                    col_types_df = self.db_api.execute_query(
                        self.user_id,
                        self.chat_id,
                        """
                        SELECT column_name, data_type
                        FROM information_schema.columns
                                                WHERE table_name = ?
                                                    AND (table_schema = ? OR table_catalog = ?)
                        ORDER BY ordinal_position
                        """.strip(),
                                                (table_name, self.config.DATA_SOURCES[0], self.config.DATA_SOURCES[0]),
                    )
                    duckdb_col_types = {
                        clean_column_table_name(str(col)): str(dtype)
                        for col, dtype in zip(
                            col_types_df["column_name"].tolist(),
                            col_types_df["data_type"].tolist(),
                        )
                    }
                except Exception:
                    duckdb_col_types = {}
                table_metadata: dict[str, str] = {
                    "description": self.db_api.get_table_description(
                        self.config.DATA_SOURCES[0], table_name
                    ),
                    "dataset_name": self.config.DATA_SOURCES[0],
                }
                if duckdb_col_types:
                    table_metadata["column_types"] = json.dumps(
                        duckdb_col_types, ensure_ascii=False
                    )

                actual_table.rename(columns=clean_column_table_name, inplace=True)
                retrieval_results.append(
                    Table(
                        doc_id=table_name,
                        retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                        content=actual_table,
                        metadata=table_metadata,
                        path=f'{self.config.DATA_SOURCES[0]}."{table_name}"',
                    )
                )

            if (
                self.config.TABLE_RETRIEVE_ENABLE_ENTITIES_RELEVANCE_BOOSTER
                and len(final_rank) > 0
            ):  # Future-TODO: Improve scoring mechanism
                for i in final_rank:
                    table_id = i[0]
                    if table_id in seen_tables:
                        # Include the keyword existence info
                        for doc in retrieval_results:
                            if doc.doc_id == table_id:
                                doc.metadata["keywords_existence"] = ", ".join(
                                    sorted(table_keywords.get(table_id, []))
                                )
                        continue
                    if len(retrieval_results) >= k:
                        break

                    seen_tables.append(table_id)
                    table_description = self.db_api.get_table_description(
                        self.config.DATA_SOURCES[0], table_id
                    )

                    booster_query = (
                        f'SELECT * FROM {self.config.DATA_SOURCES[0]}."{table_id}"'
                    )
                    if sample_only:
                        if sample_size is None or sample_size <= 0:
                            sample_size = 5
                        booster_query += f" LIMIT {sample_size}"
                    booster_table = self.db_api.execute_query(
                        self.user_id,
                        self.chat_id,
                        booster_query,
                    )

                    duckdb_col_types: dict[str, str] = {}
                    try:
                        col_types_df = self.db_api.execute_query(
                            self.user_id,
                            self.chat_id,
                            """
                            SELECT column_name, data_type
                            FROM information_schema.columns
                                                        WHERE table_name = ?
                                                            AND (table_schema = ? OR table_catalog = ?)
                            ORDER BY ordinal_position
                            """.strip(),
                                                        (table_id, self.config.DATA_SOURCES[0], self.config.DATA_SOURCES[0]),
                        )
                        duckdb_col_types = {
                            clean_column_table_name(str(col)): str(dtype)
                            for col, dtype in zip(
                                col_types_df["column_name"].tolist(),
                                col_types_df["data_type"].tolist(),
                            )
                        }
                    except Exception:
                        duckdb_col_types = {}

                    booster_metadata: dict[str, str] = {
                        "dataset_name": self.config.DATA_SOURCES[0],
                        "description": table_description,
                        "keywords_existence": ", ".join(
                            sorted(table_keywords.get(table_id, []))
                        ),
                    }
                    if duckdb_col_types:
                        booster_metadata["column_types"] = json.dumps(
                            duckdb_col_types, ensure_ascii=False
                        )
                    retrieval_results.append(
                        Table(
                            doc_id=table_id,
                            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                            content=booster_table,
                            metadata=booster_metadata,
                            path=f'{self.config.DATA_SOURCES[0]}."{table_id}"',
                        )
                    )

            return retrieval_results
        except Exception as e:
            print(f"Error during retrieval: {e}")
            raise e

    def __get_entity_extraction_sys_prompt(self) -> str:
        return f"""You are an information extraction system.

Your task is to analyze a natural-language query and extract **explicitly mentioned, concrete entities** that are suitable for direct lookup in a structured dataset.

**Extraction rules:**

* Only extract entities that are **specific, named, and canonical**, such as identifiers, symbols, or codes that would typically appear verbatim in a database.
* Do **not** extract general concepts or categories.
* If the query does **not** contain any extractable entities under these rules, return an empty list ({{"entities": []}}).

**Output requirements:**

* Output **only** a JSON object.
* The JSON object must contain a single key `"entities"` whose value is a list of strings.
* Each string must exactly match how the entity appears in the query.
* Preserve original casing and punctuation.
* Do not include duplicates.
* Do not include any explanation, comments, formatting, or additional text outside the JSON object.

**Output format:**
{{"entities": [...]}}"""

    def __keyword_relevance_by_table(
        self,
        keywords: list[str],
    ) -> tuple[list[tuple[str, float]], dict[str, set[str]]]:
        """
        Returns:
        1) dict[table_name, relevance_score in [0, 1]]
        2) dict[table_name, set[keyword]]
        """

        # ------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------
        WEIGHTS = {
            "table_hits": 3.0,
            "column_hits": 2.0,
            "data_hits": 1.0,
        }

        table_columns = self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f"""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE data_type IN ('VARCHAR', 'TEXT')
            AND table_catalog = '{self.config.DATA_SOURCES[0]}'
            """,
        )

        # Precompile keyword regexes
        keyword_regexes = {
            kw: self.__keyword_to_single_char_regex(kw)
            for kw in keywords
            if self.__keyword_to_single_char_regex(kw)
        }

        # ------------------------------------------------------------
        # Data structures
        # ------------------------------------------------------------
        # table -> keyword -> hits
        per_table_keyword_hits: dict[str, dict[str, dict[str, int]]] = {}

        # table -> keywords (original keywords, not regex)
        table_keyword_hits: dict[str, set[str]] = {}

        def _ensure(table: str, regex: str) -> None:
            per_table_keyword_hits.setdefault(table, {})
            per_table_keyword_hits[table].setdefault(
                regex,
                {"data_hits": 0, "column_hits": 0, "table_hits": 0},
            )

        # ------------------------------------------------------------
        # 1. Column-name + table-name hits
        # ------------------------------------------------------------
        for _, row in table_columns.iterrows():
            table = row["table_name"]
            column = row["column_name"]

            for keyword, regex in keyword_regexes.items():
                _ensure(table, regex)

                if re.search(regex, column):
                    per_table_keyword_hits[table][regex]["column_hits"] += 1
                    table_keyword_hits.setdefault(table, set()).add(keyword)

                if re.search(regex, table):
                    per_table_keyword_hits[table][regex]["table_hits"] += 1
                    table_keyword_hits.setdefault(table, set()).add(keyword)

        # ------------------------------------------------------------
        # 2. Data hits (DuckDB-side)
        # ------------------------------------------------------------
        for _, row in table_columns.iterrows():
            table = row["table_name"]
            column = row["column_name"]
            fq_table = f"{self.__quote_ident(self.config.DATA_SOURCES[0])}.{self.__quote_ident(table)}"

            for keyword, regex in keyword_regexes.items():
                _ensure(table, regex)

                query = f"""
                    SELECT
                        COALESCE(
                            SUM(
                                ARRAY_LENGTH(
                                    REGEXP_EXTRACT_ALL("{column}", '{regex}')
                                )
                            ),
                            0
                        ) AS cnt
                    FROM {fq_table}
                    WHERE REGEXP_MATCHES("{column}", '{regex}')
                """

                cnt = cast(
                    int,
                    self.db_api.execute_query(self.user_id, self.chat_id, query).iat[
                        0, 0
                    ],
                )

                if cnt > 0:
                    per_table_keyword_hits[table][regex]["data_hits"] += cnt
                    table_keyword_hits.setdefault(table, set()).add(keyword)

        # ------------------------------------------------------------
        # 3. Per-keyword scoring (weighted + log dampening)
        # ------------------------------------------------------------
        # keyword -> table -> score
        keyword_table_scores: dict[str, dict[str, float]] = {}

        for table, kw_map in per_table_keyword_hits.items():
            for regex, hits in kw_map.items():
                raw_score = (
                    WEIGHTS["table_hits"] * hits["table_hits"]
                    + WEIGHTS["column_hits"] * hits["column_hits"]
                    + WEIGHTS["data_hits"] * hits["data_hits"]
                )

                if raw_score > 0:
                    tf = math.log(1.0 + raw_score)
                    keyword_table_scores.setdefault(regex, {})[table] = tf

        # ------------------------------------------------------------
        # 4. Per-keyword normalization across tables
        # ------------------------------------------------------------
        normalized_scores: dict[str, dict[str, float]] = {}

        for regex, table_scores in keyword_table_scores.items():
            max_tf = max(table_scores.values(), default=0.0)
            if max_tf == 0:
                continue

            normalized_scores[regex] = {
                table: tf / max_tf for table, tf in table_scores.items()
            }

        # ------------------------------------------------------------
        # 5. Aggregate per table: mean × coverage
        # ------------------------------------------------------------
        final_scores: dict[str, float] = {}
        num_keywords = len(keyword_regexes)

        all_tables = set(per_table_keyword_hits.keys())

        for table in all_tables:
            values = []
            covered = 0

            for regex in keyword_regexes.values():
                v = normalized_scores.get(regex, {}).get(table, 0.0)
                values.append(v)
                if v > 0:
                    covered += 1

            if num_keywords == 0:
                final_scores[clean_column_table_name(table)] = 0.0
                continue

            mean_score = sum(values) / num_keywords
            coverage = covered / num_keywords

            final_scores[clean_column_table_name(table)] = mean_score * coverage

        final_rank = sorted(final_scores.items(), key=lambda x: (-x[1], x[0]))
        return final_rank, table_keyword_hits

    def __quote_ident(self, x: str) -> str:
        return '"' + x.replace('"', '""') + '"'

    def __keyword_to_single_char_regex(self, keyword: str) -> str:
        """
        Convert a keyword into a case-insensitive regex that
        allows exactly one arbitrary character between segments.
        """
        parts = re.split(r"[^A-Za-z0-9]+", keyword)
        parts = [re.escape(p) for p in parts if p]

        if not parts:
            return ""

        core = ".{1}".join(parts)

        return rf"(?i)(^|[^A-Za-z0-9_-]){core}($|[^A-Za-z0-9_-])"

    def index(self, documents: list[AbstractDocument]):
        """
        Indexes a list of documents to the retriever. Assume the documents are
        from a certain dataset only.
        """
        if len(documents) > 0:
            dataset = documents[0].metadata["dataset_name"]
            table_context = [i for i in documents if isinstance(i, TableContext)]
            tables = [i for i in documents if isinstance(i, Table)]

            schema_summaries: list[Text] = self.__get_schema_summaries(
                tables, table_context
            )
            sample_rows: list[Text] = self.__get_sample_rows(tables)

            schema_summaries = self.__split_schema_summaries(schema_summaries)
            sample_rows = self.__merge_sample_rows(sample_rows)
            table_context = self.__merge_table_context(table_context)

            print(f"[VECTOR INDEX] Indexing dataset: {dataset}")
            start = time.time()
            client = chromadb.PersistentClient(
                os.path.join(self.index_path, f"vector-index-{dataset}")
            )
            self.__indexing_vector(client, schema_summaries, sample_rows, table_context)
            end = time.time()
            print(f"[VECTOR INDEX] Indexing time: {end-start} seconds")

            print(f"[FULL-TEXT INDEX] Indexing dataset: {dataset}")
            start = time.time()
            stemmer = Stemmer.Stemmer("english")
            self.__indexing_full_text(
                stemmer,
                schema_summaries,
                sample_rows,
                table_context,
                dataset,
            )
            end = time.time()
            print(f"[FULL-TEXT INDEX] Indexing time: {end-start} seconds")

    def index_with_existing_schema_summaries(
        self,
        documents: list[AbstractDocument],
        existing_schema_summaries: list[Text] | None = None,
    ):
        """
        Indexes a list of documents to the retriever. Assume the documents are
        from a certain dataset only.
        """
        if len(documents) > 0:
            dataset = documents[0].metadata["dataset_name"]
            table_context = [i for i in documents if isinstance(i, TableContext)]
            tables = [i for i in documents if isinstance(i, Table)]

            if existing_schema_summaries is not None:
                schema_summaries = existing_schema_summaries
            else:
                schema_summaries: list[Text] = self.__get_schema_summaries(
                    tables, table_context
                )
            sample_rows: list[Text] = self.__get_sample_rows(tables)

            schema_summaries = self.__split_schema_summaries(schema_summaries)
            sample_rows = self.__merge_sample_rows(sample_rows)
            table_context = self.__merge_table_context(table_context)

            print(f"[VECTOR INDEX] Indexing dataset: {dataset}")
            start = time.time()
            client = chromadb.PersistentClient(
                os.path.join(self.index_path, f"vector-index-{dataset}")
            )
            self.__indexing_vector(client, schema_summaries, sample_rows, table_context)
            end = time.time()
            print(f"[VECTOR INDEX] Indexing time: {end-start} seconds")

            print(f"[FULL-TEXT INDEX] Indexing dataset: {dataset}")
            start = time.time()
            stemmer = Stemmer.Stemmer("english")
            self.__indexing_full_text(
                stemmer,
                schema_summaries,
                sample_rows,
                table_context,
                dataset,
            )
            end = time.time()
            print(f"[FULL-TEXT INDEX] Indexing time: {end-start} seconds")

    def __indexing_vector(
        self,
        client: ClientAPI,
        schema_summaries: list[Text],
        sample_rows: list[Text],
        contexts: list[Text] = [],
        collection_name="benchmark",
        reindex=False,
    ):
        if not reindex:
            try:
                collection = client.get_collection(collection_name)
                return collection
            except:
                pass
        try:
            client.delete_collection(collection_name)
        except:
            pass
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:random_seed": 42,
                "hnsw:M": 48,
            },
        )

        documents = []
        ids = []

        if sample_rows is not None:
            tables = sorted({content.metadata["table_name"] for content in sample_rows})
        else:
            tables = sorted(
                {content.metadata["table_name"] for content in schema_summaries}
            )
        for table in tables:
            if schema_summaries is not None:
                table_schema_summaries = [
                    content
                    for content in schema_summaries
                    if content.metadata["table_name"] == table
                ]

                for content_idx, schema_summary in enumerate(table_schema_summaries):
                    documents.append(schema_summary.content)
                    ids.append(f"{table}_SEP_contents_SEP_schema-{content_idx}")

            if sample_rows is not None:
                table_sample_rows = [
                    content
                    for content in sample_rows
                    if content.metadata["table_name"] == table
                ]

                for content_idx, sample_row in enumerate(table_sample_rows):
                    documents.append(sample_row.content)
                    ids.append(f"{table}_SEP_contents_SEP_row-{content_idx}")

            if len(contexts) > 0:
                table_contexts = [
                    context
                    for context in contexts
                    if context.metadata["table_name"] == table
                ]
                for context_idx, context in enumerate(table_contexts):
                    documents.append(context.content)
                    ids.append(f"{table}_SEP_contexts-{context_idx}")

        for i in tqdm(range(0, len(documents), 30000)):
            embeddings = self.language_model_api.encode(
                documents[i : i + 30000], EmbeddingModelOption(batch_size=100)
            )

            collection.add(
                embeddings=[embed.tolist() for embed in embeddings],
                documents=documents[i : i + 30000],
                ids=ids[i : i + 30000],
            )
        return collection

    def __indexing_full_text(
        self,
        stemmer,
        schema_summaries: list[Text],
        sample_rows: list[Text],
        contexts: list[Text],
        dataset: str,
    ):
        corpus_json = []

        if sample_rows is not None:
            tables = sorted({content.metadata["table_name"] for content in sample_rows})
        else:
            tables = sorted(
                {content.metadata["table_name"] for content in schema_summaries}
            )

        for table in tables:
            if schema_summaries is not None:
                table_schema_summaries = [
                    content
                    for content in schema_summaries
                    if content.metadata["table_name"] == table
                ]

                for content_idx, schema_summary in enumerate(table_schema_summaries):
                    corpus_json.append(
                        {
                            "text": schema_summary.content,
                            "metadata": {
                                "table": f"{table}_SEP_contents_SEP_schema-{content_idx}"
                            },
                        }
                    )

            if sample_rows is not None:
                table_sample_rows = [
                    content
                    for content in sample_rows
                    if content.metadata["table_name"] == table
                ]

                for content_idx, sample_row in enumerate(table_sample_rows):
                    corpus_json.append(
                        {
                            "text": sample_row.content,
                            "metadata": {
                                "table": f"{table}_SEP_contents_SEP_row-{content_idx}"
                            },
                        }
                    )

            if contexts is not None:
                table_contexts = [
                    context
                    for context in contexts
                    if context.metadata["table_name"] == table
                ]
                for context_idx, context in enumerate(table_contexts):
                    corpus_json.append(
                        {
                            "text": context.content,
                            "metadata": {
                                "table": f"{table}_SEP_contexts-{context_idx}"
                            },
                        }
                    )

        corpus_text = [doc["text"] for doc in corpus_json]
        corpus_tokens = bm25s.tokenize(
            corpus_text, stopwords="en", stemmer=stemmer, show_progress=False
        )

        retriever = bm25s.BM25(corpus=corpus_json)
        retriever.index(corpus_tokens, show_progress=True)
        retriever.save(os.path.join(self.index_path, f"fulltext-index-{dataset}"))

    def __get_schema_summaries(
        self, tables: list[Table], table_context: list[TableContext]
    ) -> list[Text]:
        summaries: list[Text] = []
        conversations, conv_tables, conv_cols = self.__parse_tables(
            tables, table_context
        )
        # Still need adjustments; we set the value to 20 for now.
        # optimal_batch_size = self.__get_optimal_batch_size(conversations)
        optimal_batch_size = 20
        sorted_indices = self.__get_special_indices(conversations, optimal_batch_size)

        conversations = [conversations[i] for i in sorted_indices]
        conv_tables = [conv_tables[i] for i in sorted_indices]
        conv_cols = [conv_cols[i] for i in sorted_indices]

        if len(conversations) > 0:
            outputs: list[str] = []
            max_batch_size = optimal_batch_size
            same_batch_size_counter = 0
            for i in tqdm(range(0, len(conversations), max_batch_size)):
                llm_output = self.language_model_api.batch_chat(
                    conversations[i : i + max_batch_size],
                    LLMOption(max_new_tokens=800, batch_size=optimal_batch_size),
                )
                outputs += llm_output[0]

                if llm_output[1] == optimal_batch_size:
                    same_batch_size_counter += 1
                    if same_batch_size_counter % 10 == 0:
                        optimal_batch_size = min(optimal_batch_size + 2, max_batch_size)
                else:
                    optimal_batch_size = llm_output[1]
                    same_batch_size_counter = 0

            col_narrations: dict[str, list[str]] = defaultdict(list)
            for output_idx, output in enumerate(outputs):
                col_narrations[conv_tables[output_idx]] += [
                    f"{conv_cols[output_idx]}: {output}"
                ]

            # Sample code to load created narrations
            # path = os.path.dirname(os.path.abspath(__file__))
            # col_narrations_path = os.path.join(path, "pneuma_col_narrations.txt")
            # with open(col_narrations_path, 'r') as file:
            #     content = file.read()
            #     col_narrations = ast.literal_eval(content)

            for table in tables:
                summaries.append(
                    Text(
                        doc_id=f"{table.metadata['table_name']}_schema_summary",
                        retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                        content=" | ".join(
                            col_narrations[table.metadata["table_name"]]
                        ),
                        metadata={"table_name": table.metadata["table_name"]},
                    )
                )
        summaries = sorted(summaries, key=lambda x: x.metadata["table_name"])
        return summaries

    def __get_special_indices(self, texts: list[list[LLMMessage]], batch_size: int):
        # Step 1: Sort the conversations (indices) in decreasing order
        sorted_indices = sorted(
            range(len(texts)), key=lambda x: len(texts[x][0]["content"]), reverse=True
        )

        # Step 2: Interleave the indices (longest, shortest, second longest, second shortest, ...)
        final_indices: list[int] = []
        i, j = 0, len(sorted_indices) - 1

        while i <= j:
            if i == j:
                final_indices.append(sorted_indices[i])
                break

            final_indices.append(sorted_indices[i])
            i += 1

            for _ in range(batch_size - 1):
                if i <= j:
                    final_indices.append(sorted_indices[j])
                    j -= 1
                else:
                    break
        return final_indices

    def __is_fit_in_memory(
        self, conversations: list[list[LLMMessage]], batch_size: int
    ):
        special_indices = self.__get_special_indices(conversations, batch_size)
        adjusted_conversations = [conversations[i] for i in special_indices]

        conv_low_idx = len(adjusted_conversations) // 2 - batch_size // 2
        conv_high_idx = conv_low_idx + batch_size

        output = self.language_model_api.batch_chat(
            adjusted_conversations[conv_low_idx:conv_high_idx],
            LLMOption(max_new_tokens=1, batch_size=batch_size),
        )

        cuda.empty_cache()
        gc.collect()

        if output[0] == "":
            del output
            return False
        else:
            del output
            return True

    def __get_optimal_batch_size(self, conversations):
        print("Looking for an optimal batch size")
        max_batch_size = 50  # Change to a higher value if you have more capacity to explore batch size
        min_batch_size = 1
        while min_batch_size < max_batch_size:
            mid_batch_size = (min_batch_size + max_batch_size) // 2
            print(f"Current mid batch size: {mid_batch_size}")
            if self.__is_fit_in_memory(conversations, mid_batch_size):
                min_batch_size = mid_batch_size + 1
            else:
                max_batch_size = mid_batch_size - 1
        optimal_batch_size = min_batch_size
        print(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size

    def __parse_tables(self, tables: list[Table], table_context: list[TableContext]):
        conversations: list[list[LLMMessage]] = []
        conv_tables: list[str] = []
        conv_cols: list[str] = []

        table_names = [i.metadata["table_name"] for i in tables]
        for table in tqdm(table_names):
            try:
                df = pd.read_csv(table, nrows=0)
            except pd.errors.EmptyDataError:
                continue

            table_desc = None
            relevant_table_context = [
                i
                for i in table_context
                if i.metadata["table_name"] == table
                and i.metadata["type"] == "description"
            ]
            if len(relevant_table_context) > 0:
                table_desc = relevant_table_context[0].content

            cols = df.columns
            for col in cols:
                prompt = self.__get_col_description_prompt(
                    table, " | ".join(cols), col, table_desc
                )
                conversations.append(
                    [LLMMessage(role=Role.SYSTEM.value, content=prompt)]
                )
                conv_tables.append(table)
                conv_cols.append(col)
        return conversations, conv_tables, conv_cols

    def __get_col_description_prompt(
        self,
        table_name: str,
        columns: str,
        column: str,
        table_description: Optional[str] = None,
    ):
        if table_description is not None:
            return f"""A table with the name {table_name}, which represents ```{table_description}```, has the following columns:
/*
{columns}
*/
Describe very briefly what the ```{column}``` column represents. Consider the table name as well if relevant to contextualize the description. If not possible, simply state "No description.\""""
        else:
            return f"""A table with the name {table_name} has the following columns:
/*
{columns}
*/
Describe very briefly what the ```{column}``` column represents. Consider the table name as well to contextualize the description. If not possible, simply state "No description.\""""

    def __get_sample_rows(self, tables: list[Table]) -> list[Text]:
        sample_rows: list[Text] = []
        for table_idx, table in enumerate(tqdm(tables)):
            try:
                df = pd.read_csv(
                    table.metadata["table_name"], on_bad_lines="skip", nrows=100
                )
            except pd.errors.EmptyDataError:
                continue
            sample_size = ceil(min(len(df), 5))

            selected_df = df.sample(n=sample_size, random_state=table_idx).reset_index(
                drop=True
            )
            for _, row in selected_df.iterrows():
                formatted_row = " | ".join(
                    [f"{col}: {val}" for col, val in row.items()]
                )
                sample_rows.append(
                    Text(
                        doc_id=f"{table.doc_id}_sample_row",
                        retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                        content=formatted_row,
                        metadata={"table_name": table.metadata["table_name"]},
                    )
                )
        return sample_rows

    def __split_schema_summaries(self, schema_summaries: list[Text]) -> list[Text]:
        """
        Split schema summaries to fit the constraint of the embedding model.
        """
        processed_schema_summaries: list[Text] = []
        unique_tables = sorted(
            set([summary.metadata["table_name"] for summary in schema_summaries])
        )
        for table in tqdm(unique_tables):
            table_schema_summary = [
                summary.content
                for summary in schema_summaries
                if summary.metadata["table_name"] == table
            ][0]
            column_summaries = table_schema_summary.split(" | ")
            col_idx = 0
            while col_idx < len(column_summaries):
                processed_summary = column_summaries[col_idx]

                while (col_idx + 1) < len(column_summaries):
                    temp = processed_summary + " | " + column_summaries[col_idx + 1]
                    if self.__estimate_tokens(temp) < self.config.EMBEDDING_MAX_TOKENS:
                        processed_summary = temp
                        col_idx += 1
                    else:
                        break

                col_idx += 1
                processed_schema_summaries.append(
                    Text(
                        doc_id=f"{table}_schema_summaries_{col_idx}",
                        retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                        content=processed_summary,
                        metadata={"table_name": table},
                    )
                )
        print(f"Num of schema summaries (BEFORE): {len(schema_summaries)}")
        print(f"Num of schema summaries (AFTER): {len(processed_schema_summaries)}")
        return processed_schema_summaries

    def __merge_sample_rows(self, sample_rows: list[Text]) -> list[Text]:
        unique_tables = sorted(set([row.metadata["table_name"] for row in sample_rows]))
        processed_sample_rows: list[Text] = []
        for table in tqdm(unique_tables):
            table_rows = [
                row for row in sample_rows if row.metadata["table_name"] == table
            ]

            rows_idx = 0
            while rows_idx < len(table_rows):
                processed_sample_row = table_rows[rows_idx].content

                while (rows_idx + 1) < len(table_rows):
                    temp = (
                        processed_sample_row + " || " + table_rows[rows_idx + 1].content
                    )
                    if self.__estimate_tokens(temp) < self.config.EMBEDDING_MAX_TOKENS:
                        processed_sample_row = temp
                        rows_idx += 1
                    else:
                        break

                rows_idx += 1
                processed_sample_rows.append(
                    Text(
                        doc_id=f"{table}_sample_row_{rows_idx}",
                        retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                        content=processed_sample_row,
                        metadata={"table_name": table},
                    )
                )

        print(f"Num of rows summaries (BEFORE): {len(sample_rows)}")
        print(f"Num of rows summaries (AFTER): {len(processed_sample_rows)}")
        return processed_sample_rows

    def __merge_table_context(self, table_context: list[TableContext]) -> list[Text]:
        unique_tables = sorted(
            set([context.metadata["table_name"] for context in table_context])
        )
        processed_table_context: list[Text] = []
        for table in tqdm(unique_tables):
            table_contexts = [
                context
                for context in table_context
                if context.metadata["table_name"] == table
            ]
            context_idx = 0
            while context_idx < len(table_contexts):
                processed_context = table_contexts[context_idx].content
                while (context_idx + 1) < len(table_contexts):
                    temp = (
                        processed_context
                        + " || "
                        + table_contexts[context_idx + 1].content
                    )
                    if self.__estimate_tokens(temp) < self.config.EMBEDDING_MAX_TOKENS:
                        processed_context = temp
                        context_idx += 1
                    else:
                        break

                context_idx += 1
                processed_table_context.append(
                    Text(
                        doc_id=f"{table}_context_{context_idx}",
                        retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                        content=processed_context,
                        metadata={"table_name": table},
                    )
                )
        print(f"Num of context summaries (BEFORE): {len(table_context)}")
        print(f"Num of context summaries (AFTER): {len(processed_table_context)}")
        return processed_table_context

    def __estimate_tokens(self, text: str):
        """Estimates the number of tokens in a given text."""
        try:
            if len(text) == 0:
                return 0
            enc = encoding_for_model("o4-mini")
            return len(enc.encode(text))
        except Exception:
            return len(text.split(" "))


class RerankingMode(Enum):
    NONE = 0
    LLM = 1


class HybridRetriever:

    def __init__(self, reranker: AbstractModel, reranking_mode: RerankingMode) -> None:
        self.reranker = reranker
        self.reranking_mode = reranking_mode

    def _process_nodes_bm25(
        self,
        items,
        all_ids: list,
        dictionary_id_bm25,
        bm25_retriever: bm25s.BM25,
        query_tokens,
    ):
        if bm25_retriever.corpus is None:
            raise ValueError(
                "BM25 retriever corpus is not loaded. Please ensure the corpus is loaded before processing nodes."
            )
        results = [node for node in items[0][0]]
        scores = [node for node in items[1][0]]

        extra_results = [
            bm25_retriever.corpus[dictionary_id_bm25[one_id]] for one_id in all_ids
        ]
        extra_scores = [
            bm25_retriever.get_scores(
                convert_tokenized_to_string_list(query_tokens)[0]
            )[dictionary_id_bm25[one_id]]
            for one_id in all_ids
        ]

        results.extend(extra_results)
        scores.extend(extra_scores)

        max_score = max(scores)
        min_score = min(scores)
        processed_nodes = {
            node["metadata"]["table"]: (
                (
                    1
                    if min_score == max_score
                    else (scores[i] - min_score) / (max_score - min_score)
                ),
                node["text"],
            )
            for i, node in enumerate(results)
        }
        return processed_nodes

    def _process_nodes_vec(
        self, items, missing_ids, collection: Collection, question_embedding
    ):
        extra_information = collection.get_fast(
            ids=missing_ids, limit=len(missing_ids), include=["documents", "embeddings"]  # type: ignore
        )
        items["ids"][0].extend(extra_information["ids"])
        items["documents"][0].extend(extra_information["documents"])
        items["distances"][0].extend(
            cosine(question_embedding, extra_information["embeddings"][i])  # type: ignore
            for i in range(len(missing_ids))
        )

        scores: list[float] = [1 - dist for dist in items["distances"][0]]
        documents: list[str] = items["documents"][0]
        ids: list[str] = items["ids"][0]

        max_score = max(scores)
        min_score = min(scores)
        processed_nodes = {
            ids[idx]: (
                (
                    1
                    if min_score == max_score
                    else (scores[idx] - min_score) / (max_score - min_score)
                ),
                documents[idx],
            )
            for idx in range(len(scores))
        }
        return processed_nodes

    def _llm_rerank(self, nodes: list[tuple[str, float, str]], question: str):
        # Each node is of the form (name, score, doc)
        node_tables = [node[0] for node in nodes]

        relevance_prompts = [
            [
                LLMMessage(
                    role=Role.USER.value,
                    content=self._get_relevance_prompt(
                        node[2],
                        (
                            "content"
                            if node[0].split("_SEP_")[1].startswith("contents")
                            else "context"
                        ),
                        question,
                    ),
                )
            ]
            for node in nodes
        ]

        arguments = self.reranker.batch_chat(
            relevance_prompts,
            LLMOption(
                max_new_tokens=2,
                batch_size=2,
            ),
        )[0]

        tables_relevance = {
            node_tables[arg_idx]: argument.lower().startswith("yes")
            for arg_idx, argument in enumerate(arguments)
        }

        new_nodes = [
            (table_name, score, doc)
            for table_name, score, doc in nodes
            if tables_relevance[table_name]
        ] + [
            (table_name, score, doc)
            for table_name, score, doc in nodes
            if not tables_relevance[table_name]
        ]
        return new_nodes

    def _get_relevance_prompt(self, desc: str, desc_type: str, question: str):
        if desc_type == "content":
            return f"""Given a table with the following columns:
*/
{desc}
*/
and this question:
/*
{question}
*/
Is the table relevant to answer the question? Begin your answer with yes/no."""
        else:  # Must be context
            return f"""Given this context describing a table:
*/
{desc}
*/
and this question:
/*
{question}
*/
Is the table relevant to answer the question? Begin your answer with yes/no."""

    def retrieve(
        self,
        bm25_retriever: bm25s.BM25,
        vec_retriever,
        bm25_res,
        vec_res,
        k: int,
        question: str,
        alpha=0.5,
        query_tokens=None,
        question_embedding=None,
        dictionary_id_bm25=None,
    ):
        vec_ids = {vec_id for vec_id in vec_res["ids"][0]}
        bm25_ids = {node["metadata"]["table"] for node in bm25_res[0][0]}
        processed_nodes_bm25 = self._process_nodes_bm25(
            bm25_res,
            list(vec_ids - bm25_ids),
            dictionary_id_bm25,
            bm25_retriever,
            query_tokens,
        )
        processed_nodes_vec = self._process_nodes_vec(
            vec_res, list(bm25_ids - vec_ids), vec_retriever, question_embedding
        )

        all_nodes: list[tuple[str, float, str]] = []
        for node_id in sorted(vec_ids | bm25_ids):
            bm25_score_doc = processed_nodes_bm25.get(node_id)
            vec_score_doc = processed_nodes_vec.get(node_id)
            combined_score = alpha * bm25_score_doc[0] + (1 - alpha) * vec_score_doc[0]  # type: ignore
            if bm25_score_doc[1] is None:  # type: ignore
                doc = vec_score_doc[1]  # type: ignore
            else:
                doc = bm25_score_doc[1]  # type: ignore
            all_nodes.append((node_id, combined_score, doc))

        sorted_nodes = sorted(all_nodes, key=lambda node: (-node[1], node[0]))[:k]
        if self.reranking_mode == RerankingMode.LLM:
            sorted_nodes = self._llm_rerank(sorted_nodes, question)
        return sorted_nodes
