from dataclasses import dataclass, field
from typing import Any, Iterable

from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument

MULTI_DATASET_MODE = "multi_dataset"


@dataclass(frozen=True)
class DatasetContext:
    """Which datasets a turn may draw tables from.

    In single mode (the default) every retrieval targets `default`, exactly as
    before multi-dataset mode existed: a `datasets` argument from the LLM is
    ignored. In multi mode, `Table Retrieve`/`Table Enumeration` name the
    datasets they search, and each must be in `accessible` — the datasets the
    user may access, decided by the router, never by the LLM.
    """

    default: str = ""
    accessible: tuple[str, ...] = field(default_factory=tuple)
    multi: bool = False

    @classmethod
    def single(cls, dataset_name: str) -> "DatasetContext":
        return cls(default=dataset_name, accessible=(dataset_name,), multi=False)

    def resolve(self, requested: Any) -> list[str]:
        """Returns the datasets a retrieval should search.

        Raises ValueError (surfaced to the agent as an action error) when the
        request is malformed, names an inaccessible dataset, or names none while
        the chat has no default dataset.
        """
        if not self.multi:
            return [self.default]
        if requested is None or requested == []:
            if not self.default or self.default not in self.accessible:
                raise ValueError(
                    "No dataset to search: pass `datasets` (one or more of "
                    f"{list(self.accessible)}). Call List Datasets to see what each contains."
                )
            return [self.default]
        if isinstance(requested, str):
            requested = [requested]
        if not isinstance(requested, list) or not all(
            isinstance(d, str) and d.strip() for d in requested
        ):
            raise ValueError("`datasets` must be a list of dataset names.")
        names = list(dict.fromkeys(d.strip() for d in requested))
        inaccessible = [d for d in names if d not in self.accessible]
        if inaccessible:
            raise ValueError(
                f"Dataset(s) {inaccessible} are not accessible. Accessible datasets: "
                f"{list(self.accessible)}."
            )
        return names

    def datasets_of(self, docs: Iterable[AbstractDocument]) -> list[str]:
        """Distinct datasets the given documents come from (in order of first appearance)."""
        names = (dataset_of(doc) for doc in docs)
        return list(dict.fromkeys(n for n in names if n))

    def describe(self) -> str:
        """State-prompt line; empty in single mode so prompts stay unchanged."""
        if not self.multi:
            return ""
        default = f" (default: {self.default})" if self.default else ""
        return f"Accessible datasets: {', '.join(self.accessible)}{default}"


def dataset_of(doc: AbstractDocument) -> str:
    return str(doc.metadata.get("dataset_name") or "")


def merge_retrieved(
    existing: list[AbstractDocument],
    new: list[AbstractDocument],
    searched: list[str],
) -> tuple[list[AbstractDocument], list[str]]:
    """Merges a multi-dataset retrieval into the tables already retrieved.

    A retrieval replaces previously retrieved tables from the datasets it
    searched; tables from other datasets stay. Table IDs must stay unique
    (prompts, persistence and the Materializer key tables by name), so when a
    new table shares its name with a kept table from another dataset, the new
    one wins. Returns (merged tables, notes describing every replacement).
    """
    searched_set = set(searched)
    kept = [doc for doc in existing if dataset_of(doc) not in searched_set]
    notes: list[str] = []
    by_id: dict[str, AbstractDocument] = {}
    for doc in new:
        if doc.doc_id in by_id:
            if dataset_of(by_id[doc.doc_id]) != dataset_of(doc):
                notes.append(
                    f"`{doc.doc_id}` exists in both `{dataset_of(by_id[doc.doc_id])}` and "
                    f"`{dataset_of(doc)}`; kept the one from `{dataset_of(by_id[doc.doc_id])}`."
                )
            continue
        by_id[doc.doc_id] = doc
    merged: list[AbstractDocument] = []
    for doc in kept:
        if doc.doc_id in by_id:
            if dataset_of(doc) != dataset_of(by_id[doc.doc_id]):
                notes.append(
                    f"`{doc.doc_id}` from `{dataset_of(doc)}` was replaced by the one from "
                    f"`{dataset_of(by_id[doc.doc_id])}` (tables must have unique names)."
                )
            continue
        merged.append(doc)
    return merged + list(by_id.values()), notes


def in_scope(
    docs: list[AbstractDocument], accessible: Iterable[str]
) -> list[AbstractDocument]:
    """Drops tables from datasets outside `accessible` (tables with no dataset stay)."""
    allowed = set(accessible)
    return [doc for doc in docs if not dataset_of(doc) or dataset_of(doc) in allowed]
