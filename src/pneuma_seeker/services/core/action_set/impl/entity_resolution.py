from typing import Any

import pandas as pd
from pandas import DataFrame
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import extractOne
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
    - Two resolution modes:
      - **Unsupervised** (no `canonical_entities`): clusters similar strings automatically using JaroWinkler similarity. The shortest string in each cluster becomes the canonical representative.
      - **Supervised** (`canonical_entities` provided): maps each value to the best-matching entity from the provided seed list if similarity ≥ threshold; otherwise maps to `"{_UNRESOLVED_SENTINEL}"`.
    - Args: {{
        "source_table_id": "<table whose column needs harmonization>",
        "target_column": "<column name containing noisy text>",
        "output_mapping_table_id": "<ID to register the resulting mapping table under>",
        "canonical_entities": ["<optional seed list of known canonical names>"],
        "threshold": <optional float, default from config ({ActionNames.ENTITY_RESOLUTION.value}_threshold)>
    }}
    - Example usage: after calling this action with `output_mapping_table_id = "merchant_map"`, enrich your source table via:
      `SELECT src.*, m.canonical_value AS canonical_merchant FROM source_table src JOIN merchant_map m ON src.merchant_name = m.original_value`"""

    def apply(self, input: dict[str, Any]) -> DataFrame:
        source_table_id: str = input["source_table_id"]
        target_column: str = input["target_column"]
        output_mapping_table_id: str = input["output_mapping_table_id"]
        canonical_entities: list[str] | None = input.get("canonical_entities")
        threshold: float = float(
            input.get("threshold", self.config.ENTITY_RESOLUTION_THRESHOLD)
        )

        count_df = self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f'SELECT COUNT(DISTINCT "{target_column}") AS cnt FROM "{source_table_id}" '
            f'WHERE "{target_column}" IS NOT NULL;',
        )
        total = int(count_df.iloc[0]["cnt"])

        mapping: dict[str, str | None] = {}
        # Maintained across batches so later unsupervised batches can match against
        # representatives discovered in earlier ones.
        representatives: list[str] = []

        # Batches arrive pre-sorted by (LENGTH, value): shortest strings first, which
        # ensures the unsupervised algorithm always promotes the cleanest representative.
        with tqdm(total=total, desc=f"Resolving '{target_column}'") as pbar:
            for offset in range(0, total, _BATCH_SIZE):
                batch_df = self.db_api.execute_query(
                    self.user_id,
                    self.chat_id,
                    f'SELECT DISTINCT "{target_column}" FROM "{source_table_id}" '
                    f'WHERE "{target_column}" IS NOT NULL '
                    f'ORDER BY LENGTH("{target_column}"), "{target_column}" '
                    f"LIMIT {_BATCH_SIZE} OFFSET {offset};",
                )
                batch: list[str] = [str(v) for v in batch_df[target_column].tolist()]
                if not batch:
                    break

                if canonical_entities:
                    batch_mapping = self._resolve_supervised(
                        batch, canonical_entities, threshold
                    )
                else:
                    batch_mapping, representatives = self._resolve_unsupervised_batch(
                        batch, threshold, representatives
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
        threshold: float,
    ) -> dict[str, str | None]:
        """Map each value to the best canonical entity above threshold, else sentinel."""
        mapping: dict[str, str | None] = {}
        for name in tqdm(batch, desc="Supervised mapping"):
            match = extractOne(
                name,
                canonical_entities,
                scorer=JaroWinkler.similarity,
                score_cutoff=threshold,
            )
            mapping[name] = match[0] if match else _UNRESOLVED_SENTINEL
        return mapping
