from itertools import combinations

import pandas as pd
from pyxdameraulevenshtein import damerau_levenshtein_distance

from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.action_set.interfaces import Action


class JoinPathExtraction(Action):
    """Extracts join paths between tables based on common columns."""

    def get_name(self) -> str:
        return ActionNames.JOIN_PATH_EXTRACTION.value

    def get_description(self) -> str:
        return "Extracts join paths between tables based on common columns."

    def get_input_schema(self) -> dict[str, str]:
        return {
            "tables": "A dictionary mapping table IDs to their corresponding pandas DataFrames.",
        }
    
    def get_notes(self) -> str:
        return "The join paths are discovered based on column name similarity and value overlap."

    def discover_join_paths(self, tables: dict[str, pd.DataFrame]):
        """
        tables: dict of table_name -> DataFrame
        k: number of top join candidates
        alpha: weight for name similarity
        """

        # ---- profile columns once ----
        profiles = {}
        for tname, df in tables.items():
            profiles[tname] = {}
            for col in df.columns:
                col_data = df[col]
                if isinstance(col_data, pd.DataFrame):
                    continue
                prof = self.__profile_column(col_data)
                if prof:
                    profiles[tname][col] = prof

        candidates = []

        # ---- compare table pairs ----
        for t1, t2 in combinations(tables.keys(), 2):
            for c1, p1 in profiles[t1].items():
                for c2, p2 in profiles[t2].items():

                    # prune constants
                    if p1["cardinality"] <= 1 or p2["cardinality"] <= 1:
                        continue

                    ns = self.__name_similarity(c1.lower(), c2.lower())
                    vo = self.__value_overlap(p1, p2)

                    # cheap early reject
                    if ns < 0.2 and vo == 0:
                        continue

                    score = (
                        self.config.JOIN_PATH_EXTRACTION_ALPHA * ns
                        + (1 - self.config.JOIN_PATH_EXTRACTION_ALPHA) * vo
                    )

                    if score > 0:
                        candidates.append(
                            {"score": score, "t1": t1, "c1": c1, "t2": t2, "c2": c2}
                        )

        # ---- select top-k ----
        candidates.sort(key=lambda x: x["score"], reverse=True)
        topk = candidates[: self.config.JOIN_PATH_EXTRACTION_TOP_K]

        # ---- pretty output ----
        lines = [
            f"Top-{self.config.JOIN_PATH_EXTRACTION_TOP_K} potential join paths (sorted from more likely to less likely):"
        ]
        for c in topk:
            lines.append(
                f"- {c['t1']} (columns: [{c['c1']}]) "
                f"<-> {c['t2']} (columns: [{c['c2']}]) "
                f"[score={c['score']:.3f}]"
            )
        
        self.logger.info("[JOIN PATH EXTRACTION] Discovered join paths:\n" + "\n".join(lines))
        return "\n".join(lines)

    def __value_overlap(self, col_a, col_b) -> float:
        A = col_a["values"]
        B = col_b["values"]
        if not A or not B:
            return 0.0
        return len(A & B) / min(len(A), len(B))

    def __name_similarity(self, a: str, b: str) -> float:
        d = damerau_levenshtein_distance(a, b)
        return max(0.0, 1.0 - d / max(len(a), len(b)))

    def __profile_column(self, series: pd.Series, sample_size: int = 500):
        s = series.dropna()
        if s.empty:
            return None

        s = s.astype(str)
        sample = s.sample(min(len(s), sample_size), random_state=42)
        values = set(sample)

        return {
            "values": values,
            "cardinality": s.nunique(),
            "dtype": str(series.dtype),
        }
