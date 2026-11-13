from dataclasses import dataclass, field
from typing import Any, Iterable

from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument

MULTI_DATASET_MODE = "multi_dataset"


@dataclass(frozen=True)
class DatasetContext:
    """
    Represents the datasets a turn may draw tables from. In single mode (the default),
    every retrieval targets `default` and a `datasets` argument is ignored. In multi
    mode, retrievals name the datasets they search, each of which must be in
    `accessible` (decided by the router, never by the LLM).
    """

    default: str = ""
    accessible: tuple[str, ...] = field(default_factory=tuple)
    multi: bool = False

    @classmethod
    def single(cls, dataset_name: str) -> "DatasetContext":
        return cls(default=dataset_name, accessible=(dataset_name,), multi=False)

    def resolve(self, requested: Any) -> list[str]:
        """
        Returns the datasets a retrieval should search. Raises ValueError (an action
        error for the agent) when the request is malformed, names an inaccessible
        dataset, or names none while the chat has no default dataset.
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
        """Returns the distinct datasets the documents come from, in order of first appearance."""
        names = (dataset_of(doc) for doc in docs)
        return list(dict.fromkeys(n for n in names if n))

    def describe(self) -> str:
        """
        Describes the scope for the state prompt; empty in single mode so prompts stay unchanged.
        """
        if not self.multi:
            return ""
        default = f" (default: {self.default})" if self.default else ""
        return f"Accessible datasets: {', '.join(self.accessible)}{default}"


def dataset_of(doc: AbstractDocument) -> str:
    return str(doc.metadata.get("dataset_name") or "")


def table_key(doc: AbstractDocument) -> str:
    """Returns "<dataset>.<table>" (or just the table, for a table with no dataset)."""
    dataset = dataset_of(doc)
    return f"{dataset}.{doc.doc_id}" if dataset else doc.doc_id


def merge_retrieved(
    existing: list[AbstractDocument], new: list[AbstractDocument]
) -> tuple[list[AbstractDocument], list[str]]:
    """
    Adds newly retrieved tables to those already retrieved (append-only). Table
    IDs must stay unique, since prompts, persistence and the Materializer key
    tables by name: a new table replaces an existing one of the same name.
    Returns (merged tables, notes on same-named tables from different datasets).
    """
    notes: list[str] = []
    incoming: dict[str, AbstractDocument] = {}
    for doc in new:
        first = incoming.get(doc.doc_id)
        if first is None:
            incoming[doc.doc_id] = doc
        elif dataset_of(first) != dataset_of(doc):
            notes.append(
                f"`{doc.doc_id}` exists in both `{dataset_of(first)}` and "
                f"`{dataset_of(doc)}`; kept the one from `{dataset_of(first)}`."
            )
    merged: list[AbstractDocument] = []
    for doc in existing:
        replacement = incoming.pop(doc.doc_id, None)
        if replacement is None:
            merged.append(doc)
            continue
        if dataset_of(doc) != dataset_of(replacement):
            notes.append(
                f"`{doc.doc_id}` from `{dataset_of(doc)}` was replaced by the one from "
                f"`{dataset_of(replacement)}` (tables must have unique names)."
            )
        merged.append(replacement)
    return merged + list(incoming.values()), notes


def in_scope(
    docs: list[AbstractDocument], accessible: Iterable[str]
) -> list[AbstractDocument]:
    """Drops tables from datasets outside `accessible` (tables with no dataset stay)."""
    allowed = set(accessible)
    return [doc for doc in docs if not dataset_of(doc) or dataset_of(doc) in allowed]
