"""
Table discovery with M1 (TABLE_DISCOVERY_METHOD = "m1"), ported from Agentic
Catalog: BM25 over each table's R2 representation (name, schema, sample rows;
tables with identical columns form one candidate), then M1 judges the ranking.
If M1 fails, the BM25 top k is returned. An accepted family is returned as all
its tables, since provenance needs each one as a retrieved table.

The index is built per dataset on first use, persisted under
RETRIEVAL_INDEX_ROOT/discovery/<dataset>, and rebuilt when the dataset changes.
"""

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.ir_system.m1 import M1, TABLE_INSTRUCTIONS, seeded_rng
from pneuma_seeker.services.core.ir_system.retriever.interface import AbstractRetriever
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import formatted_log, setup_logger
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument, RetrieverType, Table

SAMPLE_ROWS = 5
_MAX_VALUE_CHARS = 40
_K1, _B = 1.5, 0.75
_INDEX_VERSION = "r2-v1"  # bump to rebuild every index after a format change

_log = setup_logger()

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def cell(value: Any) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    return text if len(text) <= _MAX_VALUE_CHARS else text[: _MAX_VALUE_CHARS - 1] + "…"


def table_text(table: dict, max_columns: int, header: bool = True) -> str:
    columns = table["columns"][:max_columns]
    more = len(table["columns"]) - len(columns)
    lines = [f"Table: {table['name']} (dataset {table['dataset_name']})"] if header else []
    lines.append("Schema: " + " | ".join(c["name"] for c in columns) + (f" | … (+{more} columns)" if more else ""))
    lines.append("Sample rows:")
    lines += [" | ".join(cell(v) for v in row[: len(columns)]) for row in table.get("sample_rows", [])]
    return "\n".join(lines)


def schema_key(columns: list[dict]) -> str:
    return hashlib.sha256("|".join(f"{c['name']}:{c.get('type', '')}" for c in columns).encode()).hexdigest()


def build_candidates(dataset_name: str, tables: list[dict], max_columns: int, max_names: int) -> list[dict]:
    """Builds one candidate per table, or per family of tables with identical columns."""
    groups: dict[str, list[dict]] = {}
    for table in tables:
        groups.setdefault(schema_key(table["columns"]), []).append(table)
    out = []
    for key, members in groups.items():
        first = members[0]
        if len(members) == 1:
            out.append({"id": f"{dataset_name}.{first['name']}", "content": table_text(first, max_columns),
                        "tables": [first["name"]]})
            continue
        names = [t["name"] for t in members]
        shown = ", ".join(names[:max_names]) + (f", … and {len(names) - max_names} more" if len(names) > max_names else "")
        content = (f"Family of {len(names)} tables with the same columns (dataset {dataset_name}): {shown}\n"
                   + table_text(first, max_columns, header=False))
        out.append({"id": f"{dataset_name}:family:{key[:16]}", "content": content, "tables": names})
    return out


class TermIndex:
    """Agentic Catalog's BM25, saved as an inverted index."""

    def __init__(self, vocabulary: dict[str, int], pointers, documents, frequencies, lengths) -> None:
        self._vocabulary = vocabulary
        self._pointers = pointers  # term t's postings are [pointers[t], pointers[t + 1])
        self._documents = documents
        self._frequencies = frequencies
        self._lengths = lengths
        self._size = len(lengths)
        self._average = sum(int(n) for n in lengths) / self._size if self._size else 0.0

    @classmethod
    def build(cls, texts: list[str]) -> "TermIndex":
        tokenized = [tokenize(t) for t in texts]
        postings: dict[str, list[tuple[int, int]]] = {}
        for doc, tokens in enumerate(tokenized):
            for term, count in Counter(tokens).items():
                postings.setdefault(term, []).append((doc, count))
        vocabulary = {term: i for i, term in enumerate(postings)}
        pointers = np.zeros(len(vocabulary) + 1, dtype=np.int64)
        documents, frequencies = [], []
        for term, i in vocabulary.items():
            for doc, count in postings[term]:
                documents.append(doc)
                frequencies.append(count)
            pointers[i + 1] = len(documents)
        return cls(vocabulary, pointers, np.asarray(documents, dtype=np.int64),
                   np.asarray(frequencies, dtype=np.int64), np.asarray([len(t) for t in tokenized], dtype=np.int64))

    def save(self, directory: Path) -> None:
        (directory / "vocabulary.json").write_text(json.dumps(self._vocabulary))
        np.savez(directory / "postings.npz", pointers=self._pointers, documents=self._documents,
                 frequencies=self._frequencies, lengths=self._lengths)

    @classmethod
    def load(cls, directory: Path) -> "TermIndex":
        arrays = np.load(directory / "postings.npz")
        return cls(json.loads((directory / "vocabulary.json").read_text()), arrays["pointers"],
                   arrays["documents"], arrays["frequencies"], arrays["lengths"])

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self._size, dtype=np.float64)
        n = self._size
        for term in tokenize(query):
            t = self._vocabulary.get(term)
            if t is None:
                continue
            lo, hi = int(self._pointers[t]), int(self._pointers[t + 1])
            docs, tf = self._documents[lo:hi], self._frequencies[lo:hi].astype(np.float64)
            df = hi - lo
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            out[docs] += idf * tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * self._lengths[docs] / self._average))
        return out


_process_lock = threading.Lock()
_dataset_locks: dict[str, threading.Lock] = {}
_cache: dict[str, tuple[str, dict, list[dict], TermIndex]] = {}


def _dataset_lock(key: str) -> threading.Lock:
    with _process_lock:
        return _dataset_locks.setdefault(key, threading.Lock())


class DiscoveryIndex:
    """<root>/discovery/<dataset>/CURRENT points at the current version directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / "discovery"

    def get(self, dataset_name: str, fingerprint: str, scan) -> tuple[list[dict], TermIndex]:
        """
        Returns the dataset's candidates and BM25, (re)built with `scan()` if missing or stale.
        """
        directory = self.root / dataset_name
        loaded = self._load(directory)
        if loaded is None or loaded[1].get("fingerprint") != fingerprint:
            with _dataset_lock(str(directory)):
                loaded = self._load(directory)
                if loaded is None or loaded[1].get("fingerprint") != fingerprint:
                    self._build(directory, dataset_name, fingerprint, scan)
                    loaded = self._load(directory)
        assert loaded is not None
        return loaded[2], loaded[3]

    def rebuild(self, dataset_name: str, fingerprint: str, scan) -> int:
        """Rebuilds the dataset's index even if it looks current; returns its candidate count."""
        directory = self.root / dataset_name
        with _dataset_lock(str(directory)):
            self._build(directory, dataset_name, fingerprint, scan, force=True)
            loaded = self._load(directory)
        assert loaded is not None
        return len(loaded[2])

    def status(self, dataset_name: str) -> dict | None:
        """Returns the current index's metadata (built_at, candidates), or None if never built."""
        loaded = self._load(self.root / dataset_name)
        return loaded[1] if loaded else None

    def _load(self, directory: Path):
        try:
            version = (directory / "CURRENT").read_text().strip()
        except FileNotFoundError:
            return None
        key = str(directory)
        cached = _cache.get(key)
        if cached and cached[0] == version:
            return cached
        path = directory / version
        try:
            meta = json.loads((path / "meta.json").read_text())
            candidates = json.loads((path / "candidates.json").read_text())
            index = TermIndex.load(path)
        except (FileNotFoundError, ValueError):
            return None
        _cache[key] = (version, meta, candidates, index)
        return _cache[key]

    def _build(self, directory: Path, dataset_name: str, fingerprint: str, scan, force: bool = False) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = self._load(directory)
            if current and current[1].get("fingerprint") == fingerprint and not force:
                return
            candidates = scan()
            version = uuid.uuid4().hex
            staging = directory / f".staging-{version}"
            staging.mkdir()
            TermIndex.build([c["content"] for c in candidates]).save(staging)
            (staging / "candidates.json").write_text(json.dumps(candidates))
            (staging / "meta.json").write_text(
                json.dumps({
                    "fingerprint": fingerprint,
                    "candidates": len(candidates),
                    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
            )
            os.replace(staging, directory / version)
            pointer = directory / f".CURRENT-{version}"
            pointer.write_text(version)
            os.replace(pointer, directory / "CURRENT")
            keep = {version, current[0] if current else None}
            for path in directory.iterdir():
                if path.is_dir() and path.name not in keep:
                    shutil.rmtree(path, ignore_errors=True)
            formatted_log(_log, "NewDiscovery", f"built index for {dataset_name}: {len(candidates)} candidates")


class NewDiscovery(AbstractRetriever):
    def __init__(
        self,
        user_id: str,
        chat_id: str,
        config: Config,
        db_api: DBAPI,
        language_model_api: LanguageModelAPI,
    ):
        super().__init__(user_id, chat_id, config, db_api, language_model_api)
        self.discovery_index = DiscoveryIndex(Path(config.RETRIEVAL_INDEX_ROOT))
        self.m1 = M1(language_model_api, _log)

    @property
    def retriever_type(self) -> RetrieverType:
        return RetrieverType.PNEUMA_RETRIEVER

    def load(self):
        pass

    def index(self, documents: list[AbstractDocument]):
        """Does nothing: the index is built from the dataset itself."""

    def retrieve(
        self,
        query: str,
        dataset_name: str,
        k: int,
        sample_only: bool,
        sample_size: int | None = None,
    ) -> list[AbstractDocument]:
        self.db_api.link_dataset_tables(self.user_id, self.chat_id, dataset_name)
        candidates, index = self._index(dataset_name)
        if not candidates:
            return []
        scores = index.scores(query)
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))

        def ranked():
            for i in order:
                yield {**candidates[i], "score": float(scores[i])}

        result = self.m1.select(
            query,
            ranked(),
            seeded_rng(self.chat_id, query, dataset_name),
            tau=self.config.TABLE_DISCOVERY_BATCH_TOKENS,
            window_batches=self.config.M1_WINDOW_BATCHES,
            empty_windows=self.config.M1_EMPTY_WINDOWS,
            budget=self.config.M1_TOKEN_BUDGET,
            instructions=TABLE_INSTRUCTIONS,
        )
        accepted = result.accepted
        if result.failed:
            formatted_log(
                _log, "NewDiscovery", f"classifier failed ({result.errors[0]}): using the BM25 top {k}"
            )
            accepted = [c for _, c in zip(range(k), ranked())]
        formatted_log(
            _log,
            "NewDiscovery",
            f"{dataset_name} {query!r}: read {result.read} of {len(candidates)} candidates "
            f"in {result.calls} call(s), returned {len(accepted)} "
            f"({'BM25 fallback' if result.failed else 'M1'})",
        )
        return [
            self._table_document(dataset_name, name, sample_only, sample_size)
            for c in accepted
            for name in c["tables"]
        ]

    def rebuild_index(self, dataset_name: str) -> int:
        """Rebuilds the dataset's index now (e.g. after a connected database changed); returns its candidate count."""
        return self.discovery_index.rebuild(dataset_name, self._fingerprint(dataset_name), lambda: self._scan(dataset_name))

    def index_status(self, dataset_name: str) -> dict | None:
        """Returns the dataset's index metadata (built_at, candidates), or None if never built."""
        return self.discovery_index.status(dataset_name)

    def _index(self, dataset_name: str) -> tuple[list[dict], TermIndex]:
        return self.discovery_index.get(
            dataset_name, self._fingerprint(dataset_name), lambda: self._scan(dataset_name)
        )

    def _fingerprint(self, dataset_name: str) -> str:
        return "|".join(
            [
                _INDEX_VERSION,
                self.db_api.dataset_fingerprint(dataset_name),
                str(self.config.TABLE_DISCOVERY_MAX_COLUMNS),
                str(self.config.TABLE_DISCOVERY_GROUP_MAX_NAMES),
            ]
        )

    def _scan(self, dataset_name: str) -> list[dict]:
        self.db_api.link_dataset_tables(self.user_id, self.chat_id, dataset_name)
        tables = []
        for name in self.db_api.list_dataset_tables(dataset_name):
            columns = self._column_types(dataset_name, name)
            rows = self.db_api.execute_query(
                self.user_id, self.chat_id, f'SELECT * FROM {dataset_name}."{name}" LIMIT {SAMPLE_ROWS}'
            )
            tables.append(
                {
                    "name": name,
                    "dataset_name": dataset_name,
                    "columns": [{"name": c, "type": t} for c, t in columns.items()]
                    or [{"name": str(c), "type": ""} for c in rows.columns],
                    "sample_rows": rows.astype(object).where(rows.notna(), None).values.tolist(),
                }
            )
        return build_candidates(
            dataset_name,
            tables,
            self.config.TABLE_DISCOVERY_MAX_COLUMNS,
            self.config.TABLE_DISCOVERY_GROUP_MAX_NAMES,
        )

    def _column_types(self, dataset_name: str, table_name: str) -> dict[str, str]:
        try:
            df = self.db_api.execute_query(
                self.user_id,
                self.chat_id,
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = ? AND (table_schema = ? OR table_catalog = ?)
                ORDER BY ordinal_position
                """.strip(),
                (table_name, dataset_name, dataset_name),
            )
        except Exception:
            return {}
        return {str(c): str(t) for c, t in zip(df["column_name"], df["data_type"])}

    def _table_document(
        self,
        dataset_name: str,
        table_name: str,
        sample_only: bool,
        sample_size: int | None,
    ) -> Table:
        query = f'SELECT * FROM {dataset_name}."{table_name}"'
        if sample_only:
            query += f" LIMIT {sample_size if sample_size and sample_size > 0 else 5}"
        content = self.db_api.execute_query(self.user_id, self.chat_id, query)
        column_types = self._column_types(dataset_name, table_name)
        metadata: dict[str, str] = {
            "description": self.db_api.get_table_description(dataset_name, table_name),
            "dataset_name": dataset_name,
        }
        if column_types:
            metadata["column_types"] = json.dumps(column_types, ensure_ascii=False)
        column_descriptions = {
            c: d
            for c in column_types
            if (d := self.db_api.get_column_description(dataset_name, table_name, c))
        }
        if column_descriptions:
            metadata["column_descriptions"] = json.dumps(column_descriptions, ensure_ascii=False)
        return Table(
            doc_id=table_name,
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=content,
            metadata=metadata,
            path=f'{dataset_name}."{table_name}"',
        )
