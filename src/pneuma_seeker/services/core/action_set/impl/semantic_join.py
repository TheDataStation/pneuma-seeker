from enum import Enum
from typing import Any

import numpy as np
from pandas import DataFrame
from pyxdameraulevenshtein import damerau_levenshtein_distance
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.services.core.action_set.interfaces import Applicable
from pneuma_seeker.shared.parser import augmented_literal_eval
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.role import Role


class SyntacticSimMetric(Enum):
    NONE = None
    EDIT_DIST = "Edit Distance"
    JACCARD_QGRAM = "Jaccard QGram"


class SemanticJoin(Action, Applicable):
    def get_name(self) -> str:
        return ActionNames.SEMANTIC_JOIN.value

    def get_description(self) -> str:
        return f"""**{ActionNames.SEMANTIC_JOIN.value}**
    - Joins two tables (internal, external, or intermediate) by computing semantic similarity between specified columns.
    - Similarity uses a weighted combination of embedding cosine similarity and normalized Damerau-Levenshtein edit similarity.
    - Produces a new joined table containing matched rows and a similarity_score column.
    - Use case: when the user explicitly asks for it, when two tables contain related entities that do not match exactly by key or text (e.g., "Intl Business Machines" vs. "IBM"), or when there are no potential join paths.
        Even if both tables share a key column (e.g., "product_id"), the user may prefer semantic matching — for instance, comparing product descriptions between catalogs from different years to detect essentially identical products that were renumbered but now sold at different prices.
    - Args: {{
        "left_table_id": "<ID of left table (must exist in retrieved or intermediate tables)>",
        "right_table_id": "<ID of right table (must exist in retrieved or intermediate tables)>",
        "relevant_left_cols": ["<list of columns from left table used for semantic comparison>"],
        "relevant_right_cols": ["<list of columns from right table used for semantic comparison>"],
        "joined_table_id": "<ID to store the resulting joined table>"
    }}
    - Example: {{
        "left_table_id": "companies_2024",
        "right_table_id": "clients_2024",
        "relevant_left_cols": ["company_name", "headquarters_city"],
        "relevant_right_cols": ["client_name", "hq_location"],
        "joined_table_id": "company_client_matches"
    }}"""

    def get_input_schema(self) -> dict[str, str]:
        return {
            "left_table_id": "String table ID for the left table (backed by DB).",
            "right_table_id": "String table ID for the right table (backed by DB).",
            "relevant_left_cols": "List of column names from the left table for semantic comparison.",
            "relevant_right_cols": "List of column names from the right table for semantic comparison.",
            "joined_table_id": "String table ID to store the joined results.",
            "alpha": "Float (0 to 1) weighting cosine vs syntactic similarity (default=0.5).",
            "top_k": "Integer number of best matches to keep per left row (default=3).",
            "delimiter": "String delimiter used when concatenating text (default=' [SEP] ').",
            "embed_batch_size": "Integer batch size for embedding calls (default=30).",
            "syntactic_sim_metric": "Syntactic similarity metric to use: NONE, EDIT_DIST, or JACCARD_QGRAM (default=EDIT_DIST).",
            "use_llm": "Boolean indicating whether to use LLM filtering for matches (default=False).",
        }

    def get_notes(self) -> str:
        return """
        This action performs a semantic join between two DB-backed tables based on specified columns.
        It computes semantic similarity using embeddings and optionally syntactic similarity metrics,
        and materializes results into a target table, returning sample rows.
        """

    def apply(self, input: dict[str, Any]) -> DataFrame:
        left_table_id = input.get("left_table_id")
        right_table_id = input.get("right_table_id")
        relevant_left_cols = input.get("relevant_left_cols")
        relevant_right_cols = input.get("relevant_right_cols")
        joined_table_id = input.get("joined_table_id")
        alpha: float = input.get("alpha", self.config.SEMANTIC_JOIN_ALPHA)
        top_k: int = input.get("top_k", self.config.SEMANTIC_JOIN_TOP_K)
        delimiter: str = input.get("delimiter", self.config.SEMANTIC_JOIN_DELIMITER)
        embed_batch_size: int = input.get(
            "embed_batch_size", self.config.SEMANTIC_JOIN_BATCH_SIZE
        )
        syntactic_sim_metric: SyntacticSimMetric = input.get(
            "syntactic_sim_metric", SyntacticSimMetric.EDIT_DIST
        )
        use_llm: bool = input.get("use_llm", False)

        if not isinstance(left_table_id, str):
            raise ValueError("left_table_id must be a string.")
        if not isinstance(right_table_id, str):
            raise ValueError("right_table_id must be a string.")
        if not isinstance(joined_table_id, str):
            raise ValueError("joined_table_id must be a string.")
        if not isinstance(relevant_left_cols, list) or not all(
            isinstance(c, str) for c in relevant_left_cols
        ):
            raise ValueError("relevant_left_cols must be a list of strings.")
        if not isinstance(relevant_right_cols, list) or not all(
            isinstance(c, str) for c in relevant_right_cols
        ):
            raise ValueError("relevant_right_cols must be a list of strings.")
        if len(relevant_left_cols) == 0 or len(relevant_right_cols) == 0:
            raise ValueError(
                "relevant_left_cols and relevant_right_cols cannot be empty."
            )
        return self.join(
            left_table_id,
            right_table_id,
            relevant_left_cols,
            relevant_right_cols,
            joined_table_id,
            alpha,
            top_k,
            delimiter,
            embed_batch_size,
            syntactic_sim_metric,
            use_llm,
        )

    def join(
        self,
        left_table_id: str,
        right_table_id: str,
        relevant_left_cols: list[str],
        relevant_right_cols: list[str],
        joined_table_id: str,
        alpha: float = 0.5,
        top_k: int = 3,
        delimiter: str = " [SEP] ",
        embed_batch_size: int = 30,
        syntactic_sim_metric: SyntacticSimMetric = SyntacticSimMetric.EDIT_DIST,
        use_llm: bool = False,
    ) -> DataFrame:
        """
        Join rows from left_table_id and right_table_id using semantic similarity.

        Parameters:
            relevant_left_cols / relevant_right_cols: columns to use for semantic comparison
            alpha: weight for cosine vs edit similarity (0 to 1)
            top_k: number of best matches to keep for each row in left_df
            delimiter: used when concatenating text
            embed_batch_size: outer batching size for embeddings (progress via tqdm).
                          Set to None or <=0 to disable outer batching.

        Returns:
            DataFrame of joined rows with similarity_score column.
        """

        left_table_ref = self.__resolve_table_ref(left_table_id)
        right_table_ref = self.__resolve_table_ref(right_table_id)
        left_all_cols = self.__get_table_columns(left_table_ref)
        right_all_cols = self.__get_table_columns(right_table_ref)

        if not set(relevant_left_cols) <= set(left_all_cols):
            raise ValueError(
                "relevant_left_cols is not a subset of left_table columns."
            )
        if not set(relevant_right_cols) <= set(right_all_cols):
            raise ValueError(
                "relevant_right_cols is not a subset of right_table columns."
            )

        row_batch_size = max(1, int(self.config.SEMANTIC_JOIN_BATCH_SIZE))

        left_select = ", ".join(f'l."{col}" AS "left_{col}"' for col in left_all_cols)
        right_select = ", ".join(
            f'r."{col}" AS "right_{col}"' for col in right_all_cols
        )

        temp_table = f"{joined_table_id}__tmp_semantic_join"
        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f"""
            CREATE OR REPLACE TABLE "{temp_table}" AS
            SELECT {left_select}, {right_select}, CAST(NULL AS DOUBLE) AS similarity_score
            FROM {left_table_ref} AS l
            JOIN {right_table_ref} AS r ON FALSE;
            """,
        )

        offset = 0
        while True:
            left_df = self.db_api.execute_query(
                self.user_id,
                self.chat_id,
                f"""
                SELECT rowid, {", ".join(f'"{c}"' for c in relevant_left_cols)}
                FROM {left_table_ref}
                ORDER BY rowid
                LIMIT {row_batch_size}
                OFFSET {offset};
                """,
            )

            if left_df.empty:
                break

            left_values = self.__concat_relevant_values(
                left_df[relevant_left_cols], relevant_left_cols, delimiter
            )
            left_emb = self.__embed_texts(
                left_values,
                "Embedding left (concat)",
                embed_batch_size,
            )

            top_scores = np.full((len(left_df), top_k), -np.inf, dtype=np.float32)
            top_right_rowids = np.full((len(left_df), top_k), -1, dtype=np.int64)

            right_offset = 0
            while True:
                right_df = self.db_api.execute_query(
                    self.user_id,
                    self.chat_id,
                    f"""
                    SELECT rowid, {", ".join(f'"{c}"' for c in relevant_right_cols)}
                    FROM {right_table_ref}
                    ORDER BY rowid
                    LIMIT {row_batch_size}
                    OFFSET {right_offset};
                    """,
                )

                if right_df.empty:
                    break

                right_values = self.__concat_relevant_values(
                    right_df[relevant_right_cols], relevant_right_cols, delimiter
                )
                right_emb = self.__embed_texts(
                    right_values,
                    "Embedding right (concat)",
                    embed_batch_size,
                )

                cos_mat = self.__pairwise_cosine_sim_matrix(left_emb, right_emb)

                if syntactic_sim_metric == SyntacticSimMetric.NONE:
                    score_mat = cos_mat
                elif syntactic_sim_metric == SyntacticSimMetric.EDIT_DIST:
                    edit_mat = self.__pairwise_edit_sim_matrix(
                        left_values, right_values, desc="Edit similarity (concat)"
                    )
                    score_mat = alpha * cos_mat + (1.0 - alpha) * edit_mat
                else:
                    jaccard_qgram_mat = self.__pairwise_jaccard_qgram_matrix(
                        left_values, right_values
                    )
                    score_mat = alpha * cos_mat + (1.0 - alpha) * jaccard_qgram_mat

                right_rowids = right_df["rowid"].to_numpy(dtype=np.int64)
                for li in range(len(left_df)):
                    combined_scores = np.concatenate([top_scores[li], score_mat[li]])
                    combined_rowids = np.concatenate(
                        [top_right_rowids[li], right_rowids]
                    )
                    top_idx = np.argsort(-combined_scores)[:top_k]
                    top_scores[li] = combined_scores[top_idx]
                    top_right_rowids[li] = combined_rowids[top_idx]

                right_offset += row_batch_size

            pairs: list[tuple[int, int, float]] = []
            left_rowids = left_df["rowid"].to_numpy(dtype=np.int64)

            if use_llm:
                left_full_rows = self.__fetch_rows_by_rowid(left_table_ref, left_rowids)
                right_needed = np.unique(top_right_rowids[top_right_rowids >= 0])
                right_full_rows = self.__fetch_rows_by_rowid(
                    right_table_ref, right_needed
                )
                left_rows_by_id = {
                    int(row.rowid): row for _, row in left_full_rows.iterrows()
                }
                right_rows_by_id = {
                    int(row.rowid): row for _, row in right_full_rows.iterrows()
                }

                for li, left_rowid in enumerate(left_rowids):
                    valid_mask = top_right_rowids[li] >= 0
                    candidate_rowids = top_right_rowids[li][valid_mask].tolist()
                    if not candidate_rowids:
                        continue
                    lrow = left_rows_by_id.get(int(left_rowid))
                    if lrow is None:
                        continue
                    candidate_rrows = [
                        right_rows_by_id.get(int(rid)) for rid in candidate_rowids
                    ]
                    candidate_rrows = [
                        row for row in candidate_rrows if row is not None
                    ]
                    if not candidate_rrows:
                        continue
                    mask = self.__llm_filter_pairs(lrow, candidate_rrows)
                    for keep, rid, score in zip(
                        mask,
                        candidate_rowids,
                        top_scores[li][valid_mask],
                    ):
                        if keep == 1:
                            pairs.append((int(left_rowid), int(rid), float(score)))
            else:
                for li, left_rowid in enumerate(left_rowids):
                    valid_mask = top_right_rowids[li] >= 0
                    candidate_rowids = top_right_rowids[li][valid_mask]
                    candidate_scores = top_scores[li][valid_mask]
                    for rid, score in zip(candidate_rowids, candidate_scores):
                        pairs.append((int(left_rowid), int(rid), float(score)))

            if pairs:
                values_sql = ", ".join(
                    f"({l_id}, {r_id}, {repr(score)})" for l_id, r_id, score in pairs
                )
                self.db_api.execute_query(
                    self.user_id,
                    self.chat_id,
                    f"""
                    INSERT INTO "{temp_table}"
                    SELECT {left_select}, {right_select}, v.score AS similarity_score
                    FROM {left_table_ref} AS l
                    JOIN {right_table_ref} AS r
                    JOIN (VALUES {values_sql}) AS v(left_rowid, right_rowid, score)
                    ON l.rowid = v.left_rowid AND r.rowid = v.right_rowid
                    ORDER BY l.rowid;
                    """,
                )

            offset += row_batch_size

        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f'DROP TABLE IF EXISTS "{joined_table_id}";',
        )
        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f'ALTER TABLE "{temp_table}" RENAME TO "{joined_table_id}";',
        )

        return self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f'SELECT * FROM "{joined_table_id}" LIMIT 5;',
        )

    def __resolve_table_ref(self, table_id: str) -> str:
        db_tables = self.db_api.execute_query(
            self.user_id, self.chat_id, "SHOW TABLES;"
        )
        if "name" in db_tables.columns and table_id in db_tables["name"].tolist():
            return f'"{table_id}"'
        self.db_api.link_dataset_tables(
            self.user_id, self.chat_id, self.config.DATA_SOURCES[0]
        )
        return f'{self.config.DATA_SOURCES[0]}."{table_id}"'

    def __get_table_columns(self, table_ref: str) -> list[str]:
        df = self.db_api.execute_query(
            self.user_id, self.chat_id, f"SELECT * FROM {table_ref} LIMIT 0;"
        )
        return list(df.columns)

    def __fetch_rows_by_rowid(self, table_ref: str, rowids: np.ndarray) -> DataFrame:
        if rowids.size == 0:
            return DataFrame(columns=["rowid"])
        rowid_list = ", ".join(str(int(rid)) for rid in rowids)
        return self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f"SELECT rowid, * FROM {table_ref} WHERE rowid IN ({rowid_list});",
        )

    def __concat_relevant_values(
        self, df: DataFrame, relevant_cols: list[str], delimiter: str
    ) -> list[str]:
        """Builds per-row concatenated values of the relevant columns once."""
        texts: list[str] = []
        for _, row in df.iterrows():
            texts.append(
                self.__concat_for_embedding(
                    relevant_cols, row.to_dict(), delimiter=delimiter # type: ignore
                )
            )
        return texts

    def __concat_for_embedding(
        self,
        fields: list[str],
        row: dict[str, str | float | int],
        delimiter: str,
    ) -> str:
        """Concatenates values as 'col: value' chunks to preserve structure."""
        chunks: list[str] = []
        for col in fields:
            val = row.get(col, "")
            sval = str(val).strip()
            chunks.append(f"{col}: {sval}")
        return delimiter.join(chunks)

    def __embed_texts(
        self,
        texts: list[str],
        desc="Embedding",
        embed_batch_size: int = 256,
    ) -> np.ndarray:
        """
        Embeds texts with optional outer-level batching.

        Returns a (N, D) ndarray.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if embed_batch_size is None or embed_batch_size <= 0:
            return self.language_model_api.encode(texts)

        chunks: list[np.ndarray] = []
        n = len(texts)
        num_chunks = (n + embed_batch_size - 1) // embed_batch_size
        for i in tqdm(range(0, n, embed_batch_size), total=num_chunks, desc=desc):
            chunk = texts[i : i + embed_batch_size]
            chunks.append(self.language_model_api.encode(chunk))
        return np.vstack(chunks)

    def __pairwise_cosine_sim_matrix(
        self, Left: np.ndarray, Right: np.ndarray
    ) -> np.ndarray:
        """
        Computes pairwise cosine similarity matrix between two sets of embeddings.

        Left: (L, D), Right: (R, D) -> returns (L, R), clipped to [0, 1] (negatives -> 0).
        """
        if Left.size == 0 or Right.size == 0:
            return np.zeros((Left.shape[0], Right.shape[0]), dtype=np.float32)

        # L2-normalize rows
        def _safe_row_norm(X: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0  # avoid div-by-zero
            return X / norms

        L = _safe_row_norm(Left.astype(np.float32, copy=False))
        R = _safe_row_norm(Right.astype(np.float32, copy=False))
        S = L @ R.T  # cosine similarity in [-1, 1]
        np.clip(S, 0.0, 1.0, out=S)
        return S

    def __pairwise_edit_sim_matrix(
        self,
        left_texts: list[str],
        right_texts: list[str],
        desc: str = "Edit similarity",
    ) -> np.ndarray:
        """
        Computes pairwise normalized Damerau-Levenshtein similarity matrix (L x R).
        """
        L = len(left_texts)
        R = len(right_texts)
        M = np.zeros((L, R), dtype=np.float32)
        for i in tqdm(range(L), desc=desc, total=L):
            a = left_texts[i]
            row_vals = []
            for b in right_texts:
                row_vals.append(self.__normalized_damerau_levenshtein(a, b))
            M[i, :] = row_vals
        return M

    def __normalized_damerau_levenshtein(self, a: str, b: str) -> float:
        """Normalize edit distance to a similarity score in [0,1]."""
        if not a and not b:
            return 1.0
        max_len = max(len(a), len(b))
        d = float(damerau_levenshtein_distance(a, b))
        return max(0.0, min(1.0, 1.0 - (d / max_len)))

    def __pairwise_jaccard_qgram_matrix(
        self,
        left_texts: list[str],
        right_texts: list[str],
        q: int = 3,
        pad: bool = False,
        dtype=np.float32,
    ):
        """
        Compute the pairwise Jaccard similarity matrix between two lists of strings
        using character q-grams.

        Args:
            left_texts: List of strings (rows of the similarity matrix).
            right_texts: List of strings (columns of the similarity matrix).
            q: Length of character n-grams (default=3).
            pad: Whether to pad strings with start/end markers before extracting q-grams.
            dtype: Data type of the returned similarity matrix.

        Returns:
            A (len(left_texts), len(right_texts)) NumPy array of Jaccard similarities.
        """

        # Helper: optionally pad text so prefixes and suffixes contribute q-grams
        def maybe_pad(text: str) -> str:
            if pad:
                return ("^" * (q - 1)) + text + ("$" * (q - 1))
            return text

        # Preprocess texts
        left_texts = [maybe_pad(s) for s in left_texts]
        right_texts = [maybe_pad(s) for s in right_texts]

        # Build q-gram vocabulary across both sets
        vectorizer = CountVectorizer(
            analyzer="char",  # extract character-level features
            ngram_range=(q, q),  # fixed q-gram size
            binary=True,  # treat q-grams as sets (presence/absence) instead of count
        )
        all_texts = left_texts + right_texts
        all_vectors = vectorizer.fit_transform(all_texts)

        # Split back into left and right subsets
        left_matrix = all_vectors[: len(left_texts), :]  # type: ignore # shape (L, vocab_size)
        right_matrix = all_vectors[len(left_texts) :, :]  # type: ignore # shape (R, vocab_size)

        # Intersection counts: |A ∩ B| for each pair (via sparse dot product)
        intersections = (left_matrix @ right_matrix.T).toarray().astype(np.float32)  # type: ignore # shape (L, R)

        # Set sizes: |A| and |B| for each string
        left_sizes = np.array(left_matrix.sum(axis=1)).ravel()  # shape (L,)
        right_sizes = np.array(right_matrix.sum(axis=1)).ravel()  # shape (R,)

        # Broadcast to compute unions: |A ∪ B| = |A| + |B| - |A ∩ B|
        unions = left_sizes[:, None] + right_sizes[None, :] - intersections

        # Jaccard index: |A ∩ B| / |A ∪ B| (avoid division by zero)
        similarities = np.divide(
            intersections, unions, out=np.zeros_like(intersections), where=unions > 0
        )

        return similarities.astype(dtype)

    def __llm_filter_pairs(self, left_row, right_rows) -> list[int]:
        """
        Calls LLM to classify which right_rows are valid matches for left_row.
        Returns a Python list of 0/1 of length len(right_rows).
        """
        prompt = f"""You are given one reference item from the LEFT table and several candidate items from the RIGHT table.  
Decide which RIGHT items refer to the same or very closely equivalent entity as the LEFT item.

Output your answer as a Python list of integers without any extra explanations, one per RIGHT item, where:  
- 1 means the RIGHT item matches/is equivalent to the LEFT item.  
- 0 means it does not match.  

LEFT item:
{left_row.to_dict()}

RIGHT candidates:
{[r.to_dict() for r in right_rows]}"""

        response = "".join(
            self.language_model_api.chat(
                [LLMMessage(role=Role.SYSTEM.value, content=prompt)]
            )
        )
        return augmented_literal_eval(response)
