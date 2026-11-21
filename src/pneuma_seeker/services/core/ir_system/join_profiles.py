"""
Column profiles for join discovery: built offline per dataset, compared online.

For each column of a dataset's tables, a profile holds the kind of values it has,
and for each way join discovery compares values (see join_path_extraction's
NORMALIZATIONS) the column's non-null and distinct counts and a bottom-k sketch:
the SKETCH_SIZE smallest hashes of its distinct normalized values. From two
sketches, join discovery estimates what share of one column's values occur in the
other without reading any rows (exactly when both columns have at most
SKETCH_SIZE distinct values). Only columns that can be keys get a sketch (ids,
integers, short text); decimals, long text and the like are skipped. Building tells
each column's kind from the table's first rows, then reads the table once, into a
DuckDB copy of the columns that get a sketch, and computes from the copy, within
BUILD_MEMORY_LIMIT (beyond it, DuckDB spills to disk).

Profiles are stored as RETRIEVAL_INDEX_ROOT/join_profiles/<dataset>.json with the
dataset's fingerprint and are not used once the dataset changes. They are built
when the benchmarks are registered and when a dataset's index is rebuilt, or with:

    python -m pneuma_seeker.services.core.ir_system.join_profiles [DATASET ...] [--force] [--verbose]
"""

import argparse
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pandas import DataFrame
from pandas.api.types import is_list_like
from tqdm import tqdm

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger

VERSION = "jp-v2"  # bump to rebuild every profile after a format change
SKETCH_SIZE = 1024
CLASSIFY_ROWS = 10_000  # first rows read to tell a column's kind
BUILD_MEMORY_LIMIT = "2GB"  # DuckDB's memory while building; the rest spills to disk
LONG_TEXT = 100  # average characters above which text is not a join key
MOSTLY = 0.8  # share of a column's values that must fit a kind
MIN_SUPPORT = 5  # sketch values needed to estimate a containment
_HASH_SPACE = 2**64
_PREFIXED = r"[A-Za-z][A-Za-z_\-:#/ ]*\d+"  # ids like purchaseid_186, ID-7, NM###1
_INTEGER_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT")
_FRACTION_TYPES = ("FLOAT", "DOUBLE", "REAL", "DECIMAL")
_TEXT_TYPES = ("VARCHAR", "DATE", "TIME", "TIMESTAMP", "UUID")

_cache: dict[str, tuple[float, dict]] = {}


def key_sql(how: str, col: str, prefixed: bool) -> str:
    """Returns the DuckDB expression giving column `col`'s values normalized as `how` (case, number, digits)."""
    if how == "case":
        return f"lower(trim(CAST({col} AS VARCHAR)))"
    if how == "number":
        return f"TRY_CAST({col} AS DOUBLE)"
    if prefixed:
        return f"TRY_CAST(regexp_extract(CAST({col} AS VARCHAR), '(\\d+)$', 1) AS BIGINT)"
    return f"TRY_CAST({col} AS BIGINT)"


def _keys(kind: str) -> list[str]:
    """Returns the normalizations a column of this kind is compared under (none for "other")."""
    return {"integer": ["digits"], "prefixed": ["digits", "case"], "text": ["case"]}.get(kind, [])


def _type_kind(column_type: str) -> str:
    """Returns a column's kind from its type alone, or "?fraction" / "?text" when its values decide."""
    t = column_type.upper()
    if t.startswith(_INTEGER_TYPES):
        return "integer"
    if t.startswith(_FRACTION_TYPES):
        return "?fraction"
    if t.startswith(_TEXT_TYPES):
        return "?text"
    return "other"  # booleans, blobs, lists, structs


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def describe(run: Callable[[str], DataFrame], relation: str) -> dict[str, str]:
    """Returns {column: type} of a table."""
    described = run(f"DESCRIBE SELECT * FROM {relation}")
    return {str(r["column_name"]): str(r["column_type"]) for _, r in described.iterrows()}


def profile_relation(
    run: Callable[[str], DataFrame],
    relation: str,
    types: dict[str, str] | None = None,
    on_column: Callable[[str], None] | None = None,
) -> dict:
    """
    Returns {"rows": row count, "columns": {column: profile}} for a table, where `run`
    executes SQL and `relation` names the table in it. A profile is {"kind": integer,
    prefixed, text or other, "keys": {normalization: {"nonnull", "distinct", "hashes"}}},
    its hashes the SKETCH_SIZE smallest of the distinct normalized values; "other"
    columns get no keys. `on_column(column)` is called as each column is done.
    """
    types = types if types is not None else describe(run, relation)
    candidates = [c for c, t in types.items() if _type_kind(t) != "other"]
    kinds = {c: "other" for c in types if c not in candidates}
    if candidates:
        first_rows = f"(SELECT {', '.join(_quote(c) for c in candidates)} FROM {relation} LIMIT {CLASSIFY_ROWS})"
        kinds.update(_classify(run, first_rows, {c: types[c] for c in candidates}))
    keyed = [c for c in candidates if _keys(kinds[c])]
    for col in types:
        if col not in keyed and on_column:
            on_column(col)
    columns: dict[str, dict] = {c: {"kind": kinds[c], "keys": {}} for c in types}
    if not keyed:
        return {"rows": int(run(f"SELECT count(*) AS n FROM {relation}").iloc[0]["n"]), "columns": columns}

    # One read of the source: the columns that get a sketch, into a DuckDB table the rest reads.
    copy = _quote(f"_join_profile_{uuid.uuid4().hex}")
    run(f"CREATE OR REPLACE TEMP TABLE {copy} AS SELECT {', '.join(_quote(c) for c in keyed)} FROM {relation}")
    try:
        rows = int(run(f"SELECT count(*) AS n FROM {copy}").iloc[0]["n"])
        for col in keyed:
            for how in _keys(kinds[col]):
                columns[col]["keys"][how] = _sketch(run, copy, key_sql(how, _quote(col), kinds[col] == "prefixed"))
            if on_column:
                on_column(col)
    finally:
        run(f"DROP TABLE IF EXISTS {copy}")
    return {"rows": rows, "columns": columns}


def _sketch(run: Callable[[str], DataFrame], table: str, sql: str) -> dict:
    """Returns {"nonnull", "distinct", "hashes"} of the values `sql` gives in `table`."""
    counts = run(f"SELECT count({sql}) AS c, count(DISTINCT {sql}) AS d FROM {table}").iloc[0]
    nonnull, distinct = int(counts["c"]), int(counts["d"])
    h = f"hash(CAST({sql} AS VARCHAR))"
    # Keep hashes below a cutoff expected to leave about 2 * SKETCH_SIZE of them.
    cutoff = "" if distinct <= SKETCH_SIZE else f" FILTER (WHERE {h} < {int(_HASH_SPACE * 2 * SKETCH_SIZE / distinct)})"
    got = run(f"SELECT list(DISTINCT {h}){cutoff} AS h FROM {table} WHERE {sql} IS NOT NULL").iloc[0]["h"]
    hashes = sorted(int(x) for x in (got if is_list_like(got) else []))[:SKETCH_SIZE]  # NULL when no rows
    return {"nonnull": nonnull, "distinct": distinct, "hashes": hashes}


def _classify(run: Callable[[str], DataFrame], relation: str, types: dict[str, str]) -> dict[str, str]:
    """
    Returns each column's kind from the rows `relation` gives: decimals that are all
    whole numbers count as integers and other decimals as "other"; text is told apart
    by its values.
    """
    kinds = {col: _type_kind(t) for col, t in types.items()}
    probes = []
    for col, t in types.items():
        q = _quote(col)
        if kinds[col] == "?fraction":
            probes.append((col, f"avg(CAST({q} = floor({q}) AS INTEGER))"))
        elif kinds[col] == "?text" and t.upper().startswith("VARCHAR"):
            probes += [
                (col, f"avg(CAST(regexp_full_match({q}, '\\d+') AS INTEGER))"),
                (col, f"avg(CAST(regexp_full_match({q}, '{_PREFIXED}') AS INTEGER))"),
                (col, f"avg(length({q}))"),
            ]
    values: dict[str, list] = {}
    if probes:
        row = run(
            "SELECT " + ", ".join(f"{sql} AS p{i}" for i, (_, sql) in enumerate(probes))
            + f" FROM {relation} AS first_rows"
        ).iloc[0]
        for i, (col, _) in enumerate(probes):
            values.setdefault(col, []).append(row[f"p{i}"])
    for col, kind in kinds.items():
        got = [0.0 if v is None or v != v else float(v) for v in values.get(col, [])]
        if kind == "?fraction":
            kinds[col] = "integer" if got and got[0] >= 1.0 else "other"
        elif kind == "?text":
            if len(got) == 3 and got[0] >= MOSTLY:
                kinds[col] = "integer"
            elif len(got) == 3 and got[1] >= MOSTLY:
                kinds[col] = "prefixed"
            elif len(got) == 3 and got[2] > LONG_TEXT:
                kinds[col] = "other"
            else:
                kinds[col] = "text"
    return kinds


def containment(a: dict, b: dict) -> tuple[float | None, bool]:
    """
    Returns (estimated share of a's distinct values that occur in b, whether it is
    exact) from their sketches; (None, False) when the sketches overlap too little
    to tell, as for a small column against a much larger one.
    """
    ha, hb = a["hashes"], b["hashes"]
    if not ha or not hb:
        return 0.0, True
    a_full, b_full = a["distinct"] <= len(ha), b["distinct"] <= len(hb)
    # Below this hash both sketches hold every value their column has.
    t = min(ha[-1] if not a_full else _HASH_SPACE, hb[-1] if not b_full else _HASH_SPACE)
    below = [h for h in ha if h <= t]
    if a_full and b_full:
        return len(set(ha) & set(hb)) / len(ha), True
    if len(below) < MIN_SUPPORT:
        return None, False
    return len(set(below) & set(hb)) / len(below), False


class ColumnBar:
    """
    Shows a dataset's profiling as a bar of its columns done, under an outer bar; use
    it as JoinProfiles.build's `progress`, then close it.
    """

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name, self.bar = dataset_name, None

    def __call__(self, done: int, total: int, what: str) -> None:
        if self.bar is None:
            self.bar = tqdm(total=total, desc=f"  {self.dataset_name}", unit="column", leave=False, position=1)
        self.bar.n = done
        self.bar.set_postfix_str(what, refresh=True)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


class JoinProfiles:
    """Builds, stores and loads the join profiles of datasets."""

    def __init__(self, config: Config, db_api: DBAPI) -> None:
        self.config, self.db_api = config, db_api
        self.root = Path(config.RETRIEVAL_INDEX_ROOT) / "join_profiles"

    def _path(self, dataset_name: str) -> Path:
        return self.root / f"{dataset_name}.json"

    def _fingerprint(self, dataset_name: str) -> str:
        return f"{VERSION}|{SKETCH_SIZE}|{self.db_api.dataset_fingerprint(dataset_name)}"

    def load(self, dataset_name: str) -> dict | None:
        """Returns {table: {"rows", "columns"}} of a dataset, or None if missing or stale."""
        path = self._path(dataset_name)
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        cached = _cache.get(str(path))
        if cached is None or cached[0] != mtime:
            try:
                cached = (mtime, json.loads(path.read_text()))
            except ValueError:
                return None
            _cache[str(path)] = cached
        stored = cached[1]
        return stored["tables"] if stored.get("fingerprint") == self._fingerprint(dataset_name) else None

    def status(self, dataset_name: str) -> str:
        """Returns "current", "stale" or "missing"."""
        if self.load(dataset_name) is not None:
            return "current"
        return "stale" if self._path(dataset_name).exists() else "missing"

    def build(
        self, dataset_name: str, force: bool = False, progress: Callable[[int, int, str], None] | None = None
    ) -> bool:
        """
        Builds the dataset's profiles unless they are current (or `force`); returns whether
        it built them. `progress(done, total, what)` counts the dataset's columns done out
        of all of them, `what` naming the table being read or the column just done.
        """
        if not force and self.load(dataset_name) is not None:
            return False
        user_id, chat_id = "indexing_service", f"join-profiles-{dataset_name}"
        self.db_api.link_dataset_tables(user_id, chat_id, dataset_name)
        run = lambda sql: self.db_api.execute_query(user_id, chat_id, sql)  # noqa: E731
        run(f"SET memory_limit = '{BUILD_MEMORY_LIMIT}'")
        run("SET preserve_insertion_order = false")  # lets large copies stream with less memory
        relations = {name: f"{dataset_name}.{_quote(name)}" for name in self.db_api.list_dataset_tables(dataset_name)}
        types = {name: describe(run, relation) for name, relation in relations.items()}
        total, done = sum(len(t) for t in types.values()), 0
        tables = {}
        for name, relation in relations.items():
            if progress:
                progress(done, total, f"{name} (reading)")

            def column_done(col: str, name: str = name) -> None:
                nonlocal done
                done += 1
                if progress:
                    progress(done, total, f"{name}.{col}")

            tables[name] = profile_relation(run, relation, types[name], column_done)
        self.root.mkdir(parents=True, exist_ok=True)
        staging = self._path(dataset_name).with_suffix(".json.tmp")
        staging.write_text(json.dumps({
            "fingerprint": self._fingerprint(dataset_name),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tables": tables,
        }))
        os.replace(staging, self._path(dataset_name))
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", help="registered datasets (default: all)")
    parser.add_argument("--force", action="store_true", help="rebuild even if current")
    parser.add_argument("--verbose", action="store_true", help="also show the services' logs")
    args = parser.parse_args()
    if not args.verbose:
        logging.disable(logging.INFO)  # the bar and the summary show the progress
    config = Config()
    db_api = DBAPI(config, setup_logger("Join Profiles"))
    names = args.datasets or [s.name for s in db_api.pneuma_db.dataset_manager.registry.list_sources()]
    profiles = JoinProfiles(config, db_api)
    built = []
    start = time.monotonic()
    with tqdm(names, desc="join profiles", unit="dataset", position=0) as bar:
        for name in bar:
            bar.set_postfix_str(name)
            columns = ColumnBar(name)
            try:
                if profiles.build(name, force=args.force, progress=columns):
                    built.append(name)
            finally:
                columns.close()
    print(f"join profiles: {len(built)} built, {len(names) - len(built)} current ({time.monotonic() - start:.0f}s)")


if __name__ == "__main__":
    main()
