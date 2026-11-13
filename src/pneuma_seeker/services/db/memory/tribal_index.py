"""
BM25 over all tribal-knowledge entries, persisted under
RETRIEVAL_INDEX_ROOT/tribal_knowledge and rebuilt before answering whenever the
entries' fingerprint changes. Visibility is decided by the caller.
"""

import fcntl
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import cast

import bm25s
import numpy as np
import Stemmer

_STEMMER = Stemmer.Stemmer("english")


class TribalKnowledgeIndex:
    def __init__(self, root: Path, memory_manager) -> None:
        self.directory = Path(root) / "tribal_knowledge"
        self.memory_manager = memory_manager
        self._lock = threading.Lock()
        self._loaded: tuple[str, dict, bm25s.BM25 | None, list[str]] | None = None

    def scores(self, query: str) -> dict[str, float]:
        """Returns every entry's BM25 score for `query`, by memory_id."""
        fingerprint = self.memory_manager.fingerprint("tribal_knowledge")
        loaded = self._load()
        if loaded is None or loaded[1].get("fingerprint") != fingerprint:
            with self._lock:
                loaded = self._load()
                if loaded is None or loaded[1].get("fingerprint") != fingerprint:
                    self._build(fingerprint)
                    loaded = self._load()
        assert loaded is not None
        _, _, retriever, ids = loaded
        if retriever is None or not query.strip():
            return {memory_id: 0.0 for memory_id in ids}
        tokenized = bm25s.tokenize(query, stemmer=_STEMMER, show_progress=False, return_ids=False)
        tokens = cast(list[list[str]], tokenized)[0]
        values = np.asarray(retriever.get_scores(tokens), dtype=np.float64)
        return {memory_id: float(v) for memory_id, v in zip(ids, values)}

    def _load(self):
        try:
            version = (self.directory / "CURRENT").read_text().strip()
        except FileNotFoundError:
            return None
        if self._loaded and self._loaded[0] == version:
            return self._loaded
        path = self.directory / version
        try:
            meta = json.loads((path / "meta.json").read_text())
            ids = json.loads((path / "ids.json").read_text())
            retriever = bm25s.BM25.load(str(path)) if ids else None
        except (FileNotFoundError, ValueError):
            return None
        self._loaded = (version, meta, retriever, ids)
        return self._loaded

    def _build(self, fingerprint: list[str]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with open(self.directory / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = self._load()
            if current and current[1].get("fingerprint") == fingerprint:
                return
            rows = self.memory_manager.corpus("tribal_knowledge")
            version = uuid.uuid4().hex
            staging = self.directory / f".staging-{version}"
            staging.mkdir()
            ids = [memory_id for memory_id, _ in rows]
            if rows:
                retriever = bm25s.BM25()
                retriever.index(
                    bm25s.tokenize([text for _, text in rows], stopwords="en", stemmer=_STEMMER, show_progress=False),
                    show_progress=False,
                )
                retriever.save(str(staging))
            (staging / "ids.json").write_text(json.dumps(ids))
            (staging / "meta.json").write_text(json.dumps({"fingerprint": fingerprint, "rows": len(rows)}))
            os.replace(staging, self.directory / version)
            pointer = self.directory / f".CURRENT-{version}"
            pointer.write_text(version)
            os.replace(pointer, self.directory / "CURRENT")
            keep = {version, current[0] if current else None}
            for path in self.directory.iterdir():
                if path.is_dir() and path.name not in keep:
                    shutil.rmtree(path, ignore_errors=True)
