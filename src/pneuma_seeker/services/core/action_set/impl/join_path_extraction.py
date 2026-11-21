import re
import time
from itertools import combinations

import pandas as pd
from pyxdameraulevenshtein import damerau_levenshtein_distance

from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.services.core.ir_system.join_profiles import JoinProfiles, containment, key_sql
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType


# Ids made of a word prefix and a number, such as purchaseid_186, ID-7 or user 12 (not dates).
_PREFIXED = re.compile(r"^[A-Za-z][A-Za-z_\-:#/ ]*\d+$")
_MOSTLY = 0.8  # share of a column's values that must fit a normalization
_SUBSTANTIAL = 0.3  # overlap (one way) that makes a join path with a unique side likely
_VERIFY_MAX_ROWS = 200_000

# name -> how a suggestion describes it
NORMALIZATIONS = {
    "case": "ignoring case and surrounding spaces",
    "number": "as numbers",
    "digits": "on their trailing digits",
}


def _cardinality(left_unique: bool, right_unique: bool) -> str:
    return {(True, True): "1:1", (False, True): "N:1", (True, False): "1:N"}.get((left_unique, right_unique), "N:M")


def _normalize(values: pd.Series, how: str | None) -> pd.Series:
    """Returns string values normalized as NORMALIZATIONS[how] describes (as is for None)."""
    if how is None:
        return values
    if how == "case":
        return values.str.strip().str.lower()
    if how == "number":
        numbers = pd.to_numeric(values, errors="coerce").dropna()
        return numbers.map(lambda x: str(int(x)) if float(x).is_integer() else repr(float(x)))
    # Trailing digits of prefixed ids (purchaseid_007 -> 7); plain numbers (7, 7.0) as integers.
    numbers = pd.to_numeric(values, errors="coerce")
    whole = numbers.dropna().map(lambda x: str(int(x)) if float(x).is_integer() else None).dropna()
    digits = values[numbers.isna()].str.extract(r"(\d+)$", expand=False).dropna()
    return pd.concat([whole, digits.map(lambda d: str(int(d)))])


class JoinPathExtraction(Action):
    """Internal tool — not exposed to LLM agents; used by ActionSet directly."""

    action_name = ActionNames.JOIN_PATH_EXTRACTION
    agents: frozenset = frozenset()  # no agent may invoke this via planning
    flag = None
    order = 99
    show_in_prompt = False

    def get_description(self, agent: AgentType | None = None) -> str:
        return "Extracts join paths between tables based on common columns."

    def get_input_schema(self) -> dict[str, str]:
        return {
            "tables": "A dictionary mapping table IDs to their corresponding pandas DataFrames.",
        }

    def get_notes(self) -> str:
        return "The join paths are discovered based on column name similarity and value overlap."

    def discover_join_paths(self, tables: dict[str, pd.DataFrame], sources: dict[str, str] | None = None):
        """
        Returns the top-k likely join paths between the tables (table id -> rows), scored
        by column-name similarity and value overlap.

        When every table is a source table whose dataset (`sources`: table id -> dataset)
        has current join profiles (see ir_system.join_profiles), overlap is the share of
        one column's distinct values that occur in the other, estimated from the
        profiles without reading rows. Otherwise it is computed on the given rows: with
        JOIN_PATH_EXTRACTION_NORMALIZE also after a normalization, and with
        JOIN_PATH_EXTRACTION_VERIFY each path shows how many of those rows match.
        """
        start = time.monotonic()
        stored = self.__stored_profiles(tables, sources or {})
        if stored is not None:
            topk, mode = self.__from_profiles(stored), "from join profiles"
        else:
            topk, mode = self.__from_rows(tables), "from the retrieved rows"

        # ---- pretty output ----
        lines = [
            f"Top-{self.config.JOIN_PATH_EXTRACTION_TOP_K} potential join paths (sorted from more likely to less likely):"
        ]
        def name(t: str, others: int) -> str:
            return f"{t} (and {others} other tables with its columns)" if others else t

        for c in topk:
            line = (
                f"- {name(c['t1'], c.get('others', (0, 0))[0])} (columns: [{c['c1']}]) "
                f"<-> {name(c['t2'], c.get('others', (0, 0))[1])} (columns: [{c['c2']}]) "
                f"[score={c['score']:.3f}]"
            )
            if c["how"] is not None and c["vo"] > 0:
                left, right = (
                    key_sql(c["how"], f'{t}."{col}"', prefixed)
                    for t, col, prefixed in ((c["t1"], c["c1"], c["prefixed"][0]), (c["t2"], c["c2"], c["prefixed"][1]))
                )
                line += f"; values match {NORMALIZATIONS[c['how']]}: {left} = {right}"
            if "check" in c:
                if c["vo"] == 0:
                    line += "; similar names only, no shared values"
                line += "; " + c["check"]["text"]
            lines.append(line)

        self.logger.info(
            f"[JOIN PATH EXTRACTION] Discovered join paths {mode} ({time.monotonic() - start:.2f}s):\n"
            + "\n".join(lines)
        )
        return "\n".join(lines)

    def __stored_profiles(self, tables: dict[str, pd.DataFrame], sources: dict[str, str]) -> dict | None:
        """Returns {table id: its stored join profile} if every table has a current one, else None."""
        if not self.config.JOIN_PATH_EXTRACTION_PROFILES or not tables:
            return None
        store, loaded, out = JoinProfiles(self.config, self.db_api), {}, {}
        for t in tables:
            dataset = sources.get(t)
            if not dataset:
                return None
            if dataset not in loaded:
                try:
                    loaded[dataset] = store.load(dataset)
                except Exception as exc:
                    loaded[dataset] = None
                    self.logger.info(f"[JOIN PATH EXTRACTION] Could not load the join profiles of {dataset}: {exc}")
            if loaded[dataset] is None or t not in loaded[dataset]:
                self.logger.info(
                    f"[JOIN PATH EXTRACTION] No current join profiles for {dataset}; comparing the retrieved rows. "
                    f"Build them with: python -m pneuma_seeker.services.core.ir_system.join_profiles {dataset}"
                )
                return None
            out[t] = loaded[dataset][t]
        return out

    def __from_profiles(self, stored: dict) -> list[dict]:
        """
        Returns the top-k join paths scored from the tables' join profiles: paths with a
        unique side (as a key is) whose values overlap substantially first, then those
        whose values overlap at all, each by score. Trailing
        digits are compared only between columns of one kind (prefixed ids with prefixed
        ids, integers with integers), since small numbers occur in any range of ids.

        Tables with the same columns (partitions of one table, e.g. one file per period)
        are compared once, through the first of them, since they are unioned, not joined.
        """
        families: dict[tuple, list[str]] = {}
        for t in sorted(stored):
            signature = tuple(sorted((c, p["kind"]) for c, p in stored[t]["columns"].items()))
            families.setdefault(signature, []).append(t)
        others = {members[0]: len(members) - 1 for members in families.values()}
        stored = {t: stored[t] for t in others}

        def pct(share: float | None, exact: bool) -> str:
            if share is None:
                return "an unknown share (too few values in the sketches)"
            return f"{share:.0%}" if exact else f"about {share:.0%}"

        candidates = []
        for t1, t2 in combinations(stored.keys(), 2):
            for c1, p1 in stored[t1]["columns"].items():
                for c2, p2 in stored[t2]["columns"].items():
                    ns = self.__name_similarity(c1.lower(), c2.lower())
                    best = None
                    for how in sorted(p1["keys"].keys() & p2["keys"].keys()):
                        if how == "digits" and p1["kind"] != p2["kind"]:
                            continue
                        k1, k2 = p1["keys"][how], p2["keys"][how]
                        if k1["distinct"] <= 1 or k2["distinct"] <= 1:
                            continue  # constants
                        (a, a_exact), (b, b_exact) = containment(k1, k2), containment(k2, k1)
                        vo = max(a or 0.0, b or 0.0)
                        if best is None or vo > best["vo"]:
                            best = {"vo": vo, "how": how, "a": (a, a_exact), "b": (b, b_exact), "k1": k1, "k2": k2}
                    if best is None or (ns < 0.2 and best["vo"] == 0):
                        continue
                    k1, k2 = best["k1"], best["k2"]
                    unique = (k1["distinct"] == k1["nonnull"], k2["distinct"] == k2["nonnull"])
                    candidates.append({
                        "score": self.config.JOIN_PATH_EXTRACTION_ALPHA * ns
                        + (1 - self.config.JOIN_PATH_EXTRACTION_ALPHA) * best["vo"],
                        "t1": t1, "c1": c1, "t2": t2, "c2": c2, "how": best["how"], "vo": best["vo"],
                        "others": (others[t1], others[t2]),
                        "prefixed": (p1["kind"] == "prefixed", p2["kind"] == "prefixed"),
                        "check": {
                            "text": (
                                f"{pct(*best['a'])} of {t1}.{c1} distinct values occur in {t2}.{c2}, "
                                f"{pct(*best['b'])} the other way; {_cardinality(*unique)}"
                            ),
                            "keyed": unique[0] or unique[1],
                            "coverage": best["vo"],
                        },
                    })
        candidates.sort(
            key=lambda c: (
                c["check"]["keyed"] and c["vo"] >= _SUBSTANTIAL,
                c["check"]["keyed"] and c["vo"] > 0,
                c["score"],
            ),
            reverse=True,
        )
        return candidates[: self.config.JOIN_PATH_EXTRACTION_TOP_K]

    def __from_rows(self, tables: dict[str, pd.DataFrame]) -> list[dict]:
        """Returns the top-k join paths scored from the given rows (the retrieved tables' few rows)."""
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
                    vo, how = self.__value_overlap(p1, p2)

                    # cheap early reject
                    if ns < 0.2 and vo == 0:
                        continue

                    score = (
                        self.config.JOIN_PATH_EXTRACTION_ALPHA * ns
                        + (1 - self.config.JOIN_PATH_EXTRACTION_ALPHA) * vo
                    )

                    if score > 0:
                        candidates.append(
                            {"score": score, "t1": t1, "c1": c1, "t2": t2, "c2": c2,
                             "how": how, "vo": vo,
                             "prefixed": (p1.get("prefixed_share", 0) >= _MOSTLY, p2.get("prefixed_share", 0) >= _MOSTLY)}
                        )

        # ---- select top-k ----
        candidates.sort(key=lambda x: x["score"], reverse=True)
        k = self.config.JOIN_PATH_EXTRACTION_TOP_K
        if self.config.JOIN_PATH_EXTRACTION_VERIFY:
            # Check a wider pool on the rows, then rank: paths whose rows match with a
            # unique side (as a key is) first, then by the share of rows that match,
            # then by score.
            pool = candidates[: 3 * k]
            for c in pool:
                c["check"] = self.__verify(tables, c)
            pool.sort(
                key=lambda c: (
                    c["check"]["keyed"] and c["check"]["coverage"] > 0,
                    c["check"]["coverage"],
                    c["score"],
                ),
                reverse=True,
            )
            topk = pool[:k]
        else:
            topk = candidates[:k]
        return topk

    def __value_overlap(self, col_a, col_b) -> tuple[float, str | None]:
        """
        Returns the best overlap of the two columns' sampled values and the
        normalization that gives it (None: as is, which wins ties).
        """
        best, best_how = self.__overlap(col_a["values"], col_b["values"]), None
        for how, (a, b) in self.__normalized_pairs(col_a, col_b).items():
            overlap = self.__overlap(a, b)
            if overlap > best:
                best, best_how = overlap, how
        return best, best_how

    @staticmethod
    def __overlap(A: set, B: set) -> float:
        if not A or not B:
            return 0.0
        return len(A & B) / min(len(A), len(B))

    def __normalized_pairs(self, col_a, col_b) -> dict[str, tuple[set, set]]:
        """Returns {normalization: (values of a, values of b)} for those that apply to both columns."""
        if not self.config.JOIN_PATH_EXTRACTION_NORMALIZE:
            return {}
        pairs = {}
        if col_a["text"] and col_b["text"]:
            pairs["case"] = (col_a["case"], col_b["case"])
        if col_a["numeric_share"] >= _MOSTLY and col_b["numeric_share"] >= _MOSTLY:
            pairs["number"] = (col_a["number"], col_b["number"])
        # Prefixed ids (purchaseid_7) match prefixed ids (bookid_7), or plain integer ids
        # when those are (nearly) unique, as ids are.
        a_prefixed, b_prefixed = col_a["prefixed_share"] >= _MOSTLY, col_b["prefixed_share"] >= _MOSTLY
        a_id, b_id = col_a["integer_id"], col_b["integer_id"]
        if (a_prefixed and (b_prefixed or b_id)) or (b_prefixed and a_id):
            pairs["digits"] = (col_a["digits"], col_b["digits"])
        return pairs

    def __verify(self, tables: dict[str, pd.DataFrame], c: dict) -> dict:
        """
        Returns {"text": how many rows of each table match the other and the path's
        cardinality, "keyed": whether a side is unique, "coverage": the larger share}.
        """
        try:
            left = self.__key_values(tables[c["t1"]][c["c1"]], c["how"])
            right = self.__key_values(tables[c["t2"]][c["c2"]], c["how"])
        except Exception as exc:
            return {"text": f"not checked ({exc})", "keyed": False, "coverage": 0.0}
        if left.empty or right.empty:
            return {"text": "no rows to check", "keyed": False, "coverage": 0.0}
        left_share = left.isin(set(right)).mean()
        right_share = right.isin(set(left)).mean()
        left_unique, right_unique = left.is_unique, right.is_unique
        cardinality = _cardinality(left_unique, right_unique)
        sampled = " (on a sample)" if max(len(left), len(right)) >= _VERIFY_MAX_ROWS else ""
        return {
            "text": (
                f"{left_share:.0%} of {c['t1']} rows and {right_share:.0%} of {c['t2']} rows "
                f"have a match; {cardinality}{sampled}"
            ),
            "keyed": left_unique or right_unique,
            "coverage": max(left_share, right_share),
        }

    @staticmethod
    def __key_values(series: pd.Series, how: str | None) -> pd.Series:
        s = series.dropna()
        if len(s) > _VERIFY_MAX_ROWS:
            s = s.sample(_VERIFY_MAX_ROWS, random_state=42)
        return _normalize(s.astype(str), how)

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

        profile = {
            "values": values,
            "cardinality": s.nunique(),
            "dtype": str(series.dtype),
        }
        if self.config.JOIN_PATH_EXTRACTION_NORMALIZE:
            numbers = pd.to_numeric(sample, errors="coerce")
            profile.update({
                "text": not pd.api.types.is_numeric_dtype(series),
                "case": set(_normalize(sample, "case")),
                "numeric_share": numbers.notna().mean(),
                "number": set(_normalize(sample, "number")),
                "prefixed_share": sample.str.match(_PREFIXED).mean(),
                "digits": set(_normalize(sample, "digits")),
                "integer_id": bool(
                    numbers.notna().mean() >= _MOSTLY
                    and (numbers.dropna() % 1 == 0).all()
                    and profile["cardinality"] >= 0.9 * len(s)
                ),
            })
        return profile
