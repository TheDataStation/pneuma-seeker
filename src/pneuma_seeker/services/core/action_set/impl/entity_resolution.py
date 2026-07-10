from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import extractOne
from rapidfuzz.utils import default_process
from tqdm import tqdm

from pneuma_seeker.services.core.action_set.interfaces import Action, Applicable
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType

_UNRESOLVED_SENTINEL = "__unresolved__"
_BATCH_SIZE = 1000


class EntityResolution(Action, Applicable):
    """Vocabulary harmonization: maps every unique dirty string in a column to a canonical form."""

    action_name = ActionNames.ENTITY_RESOLUTION
    agents = frozenset({AgentType.MATERIALIZER})
    flag = None
    order = 13
    show_in_prompt = True

    def get_description(self, agent: AgentType | None = None) -> str:
        return f"""**{ActionNames.ENTITY_RESOLUTION.value}**
    - Harmonizes noisy, non-standardized text values in a single column by producing a two-column mapping table: `original_value` → `canonical_value`.
    - The mapping table has a strict schema: `original_value VARCHAR PRIMARY KEY, canonical_value VARCHAR`.
    - Every unique non-null value in `source_table_id.[target_column]` appears exactly once in `original_value`, so a downstream `JOIN ON original_value` is always lossless.
    - Two resolution modes, chosen and thresholded by server configuration (not caller-controllable):
      - **Unsupervised** (no `canonical_entities`): clusters similar strings automatically using JaroWinkler similarity. The shortest string in each cluster becomes the canonical representative.
      - **Supervised** (`canonical_entities` provided): maps each value to the best-matching entity from the provided seed list if similarity ≥ the configured threshold; otherwise maps to `"{_UNRESOLVED_SENTINEL}"`. Matching strategy:
        - `jarowinkler`: syntactic similarity only.
        - `embedding`: semantic similarity via text embeddings (cosine similarity).
        - `hybrid`: candidate must pass **both** JaroWinkler AND embedding thresholds; winner chosen by embedding similarity. Best for noisy real-world names where pure string matching fails.
    - Args: {{
        "source_table_id": "<table to harmonize — bare ID for intermediate/external tables; dataset-qualified form (e.g. 'dataset.\"table_id\"') for retrieved tables>",
        "target_column": "<column name containing noisy text>",
        "output_mapping_table_id": "<ID to register the resulting mapping table under>",
        "canonical_entities": ["<optional seed list of known canonical names>"]
    }}
    - Example usage: after calling this action with `output_mapping_table_id = "merchant_map"`, enrich your source table via:
      `SELECT src.*, m.canonical_value AS canonical_merchant FROM source_table src JOIN merchant_map m ON src.merchant_name = m.original_value`"""

    def apply(self, input: dict[str, Any]) -> DataFrame:
        source_table_id: str = input["source_table_id"]
        target_column: str = input["target_column"]
        output_mapping_table_id: str = input["output_mapping_table_id"]
        canonical_entities: list[str] | None = input.get("canonical_entities")
        mode: str = input.get("mode", self.config.ENTITY_RESOLUTION_MODE).lower()
        threshold_raw = input.get("threshold", None)

        # Pick sensible per-mode defaults when no threshold is supplied
        if threshold_raw is None:
            if mode == "hybrid":
                threshold_input: Any = {
                    "jarowinkler": self.config.ENTITY_RESOLUTION_JW_THRESHOLD,
                    "embedding": self.config.ENTITY_RESOLUTION_EMBEDDING_THRESHOLD,
                }
            elif mode == "embedding":
                threshold_input = self.config.ENTITY_RESOLUTION_EMBEDDING_THRESHOLD
            else:
                threshold_input = self.config.ENTITY_RESOLUTION_JW_THRESHOLD
        else:
            threshold_input = threshold_raw

        # Schema-qualified refs like proc_spend."tbl" must not be re-quoted;
        # bare workspace table names like "merchant_map" need quoting.
        source_ref = (
            source_table_id if "." in source_table_id else f'"{source_table_id}"'
        )

        count_df = self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f'SELECT COUNT(DISTINCT "{target_column}") AS cnt FROM {source_ref} '
            f'WHERE "{target_column}" IS NOT NULL;',
        )
        total = int(count_df.iloc[0]["cnt"])

        mapping: dict[str, str | None] = {}
        representatives: list[str] = []

        # Pre-compute canonical embeddings once outside the batch loop
        canonical_embeddings: np.ndarray | None = None
        if canonical_entities and mode in ("embedding", "hybrid"):
            canonical_embeddings = self._encode_normalized(canonical_entities)

        with tqdm(total=total, desc=f"Resolving '{target_column}'") as pbar:
            for offset in range(0, total, _BATCH_SIZE):
                batch_df = self.db_api.execute_query(
                    self.user_id,
                    self.chat_id,
                    f'SELECT DISTINCT "{target_column}" FROM {source_ref} '
                    f'WHERE "{target_column}" IS NOT NULL '
                    f'ORDER BY LENGTH("{target_column}"), "{target_column}" '
                    f"LIMIT {_BATCH_SIZE} OFFSET {offset};",
                )
                batch: list[str] = [str(v) for v in batch_df[target_column].tolist()]
                if not batch:
                    break

                if canonical_entities:
                    batch_mapping = self._resolve_supervised(
                        batch,
                        canonical_entities,
                        threshold_input,
                        mode,
                        canonical_embeddings,
                    )
                else:
                    jw_threshold = (
                        float(threshold_input)
                        if isinstance(threshold_input, (int, float))
                        else self.config.ENTITY_RESOLUTION_JW_THRESHOLD
                    )
                    batch_mapping, representatives = self._resolve_unsupervised_batch(
                        batch, jw_threshold, representatives
                    )

                mapping.update(batch_mapping)
                pbar.update(len(batch))

        if not mapping:
            self.db_api.execute_query(
                self.user_id,
                self.chat_id,
                f'CREATE OR REPLACE TABLE "{output_mapping_table_id}" '
                f"(original_value VARCHAR PRIMARY KEY, canonical_value VARCHAR);",
            )
        else:
            mapping_df = pd.DataFrame(
                [
                    {"original_value": orig, "canonical_value": canon}
                    for orig, canon in mapping.items()
                ]
            )
            tmp_name = f"__er_tmp_{output_mapping_table_id}__"
            self.db_api.register_temporary_df(
                self.user_id, self.chat_id, mapping_df, tmp_name
            )
            self.db_api.execute_query(
                self.user_id,
                self.chat_id,
                f'CREATE OR REPLACE TABLE "{output_mapping_table_id}" AS '
                f'SELECT original_value, canonical_value FROM "{tmp_name}";',
            )

        return self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f'SELECT * FROM "{output_mapping_table_id}" LIMIT 5;',
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_normalized(self, texts: list[str]) -> np.ndarray:
        """Encode texts and L2-normalize for cosine similarity via dot product."""
        vectors = np.asarray(self.language_model_api.encode(texts), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return vectors / norms

    # ------------------------------------------------------------------
    # Resolution modes
    # ------------------------------------------------------------------

    def _resolve_unsupervised_batch(
        self,
        batch: list[str],
        threshold: float,
        representatives: list[str],
    ) -> tuple[dict[str, str | None], list[str]]:
        """Process one length-sorted batch; mutates and returns the shared representatives list."""
        mapping: dict[str, str | None] = {}
        for name in tqdm(batch, desc="Unsupervised mapping"):
            if not name.strip():
                mapping[name] = name
                continue
            match = extractOne(
                name,
                representatives,
                scorer=JaroWinkler.similarity,
                processor=default_process,
                score_cutoff=threshold,
            )
            if match:
                mapping[name] = match[0]
            else:
                representatives.append(name)
                mapping[name] = name
        return mapping, representatives

    def _resolve_supervised(
        self,
        batch: list[str],
        canonical_entities: list[str],
        threshold_input: Any,
        mode: str,
        canonical_embeddings: np.ndarray | None,
    ) -> dict[str, str | None]:
        mapping: dict[str, str | None] = {}

        if isinstance(threshold_input, dict):
            t_jw = float(
                threshold_input.get(
                    "jarowinkler", self.config.ENTITY_RESOLUTION_JW_THRESHOLD
                )
            )
            t_emb = float(
                threshold_input.get(
                    "embedding", self.config.ENTITY_RESOLUTION_EMBEDDING_THRESHOLD
                )
            )
        else:
            t_jw = float(threshold_input)
            t_emb = float(threshold_input)

        if mode == "jarowinkler":
            for name in batch:
                match = extractOne(
                    name,
                    canonical_entities,
                    scorer=JaroWinkler.similarity,
                    processor=default_process,
                    score_cutoff=t_jw,
                )
                mapping[name] = match[0] if match else _UNRESOLVED_SENTINEL
            return mapping

        # embedding or hybrid: compute batch embeddings and cosine similarities
        batch_embeddings = self._encode_normalized(batch)
        cosine_sims = np.dot(batch_embeddings, canonical_embeddings.T)  # type: ignore[union-attr]

        for idx, name in enumerate(batch):
            if mode == "embedding":
                best_idx = int(np.argmax(cosine_sims[idx]))
                mapping[name] = (
                    canonical_entities[best_idx]
                    if cosine_sims[idx][best_idx] >= t_emb
                    else _UNRESOLVED_SENTINEL
                )
            elif mode == "hybrid":
                jw_scores = np.array(
                    [
                        JaroWinkler.similarity(name, entity, processor=default_process)
                        for entity in canonical_entities
                    ]
                )
                valid_mask = (jw_scores >= t_jw) & (cosine_sims[idx] >= t_emb)
                if np.any(valid_mask):
                    passing_indices = np.where(valid_mask)[0]
                    best_idx = int(
                        passing_indices[np.argmax(cosine_sims[idx][passing_indices])]
                    )
                    mapping[name] = canonical_entities[best_idx]
                else:
                    mapping[name] = _UNRESOLVED_SENTINEL

        return mapping
