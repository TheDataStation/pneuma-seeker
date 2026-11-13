"""
How retrieved tables are shown to the agents. Tables accumulate across
retrievals; a table is in the spotlight (schema and sample rows) if it was
returned by the latest Table Retrieve or a column source of T uses it, and is
otherwise listed on one line. Tables with identical columns are shown once.
"""

from fnmatch import fnmatchcase

from pneuma_seeker.services.core.dataset_context import dataset_of, table_key
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument

_MAX_COMPACT_COLUMNS = 30


def _patterns(entry: str) -> list[str]:
    """
    Returns the table patterns a source entry may name: "<dataset>.<table>.<column>",
    "<table>.<column>" or "<dataset>.<table>", with `*` allowed.
    """
    entry = entry.strip()
    head = entry.rsplit(".", 1)[0] if "." in entry else entry
    return [entry, head]


def matches(entry: str, doc: AbstractDocument) -> bool:
    names = (table_key(doc), doc.doc_id)
    return any(fnmatchcase(name, p) for p in _patterns(entry) for name in names)


def source_tables(
    column_sources: dict[str, dict[str, list[str]]], docs: list[AbstractDocument]
) -> tuple[set[str], list[str]]:
    """Returns the tables (keys) T's column sources use, and the entries matching no table."""
    used: set[str] = set()
    unmatched: list[str] = []
    for columns in column_sources.values():
        for entries in columns.values():
            for entry in entries:
                hits = {table_key(d) for d in docs if matches(entry, d)}
                used |= hits
                if not hits and entry not in unmatched:
                    unmatched.append(entry)
    return used, unmatched


def _signature(doc: AbstractDocument) -> tuple:
    content = getattr(doc, "content", None)
    columns = tuple(str(c) for c in getattr(content, "columns", []))
    return dataset_of(doc), columns


def _found_by(keys: list[str], found_by: dict[str, list[str]]) -> str:
    queries: list[str] = []
    for key in keys:
        queries += [q for q in found_by.get(key, []) if q not in queries]
    return f" (found by: {'; '.join(queries)})" if queries else ""


def render_retrieved_tables(
    docs: list[AbstractDocument],
    spotlight: set[str] | None = None,
    found_by: dict[str, list[str]] | None = None,
) -> str:
    """
    Renders the retrieved tables; those in `spotlight` (all, if None) are shown with
    sample rows.
    """
    found_by = found_by or {}
    groups: dict[tuple, list[AbstractDocument]] = {}
    for doc in docs:
        signature = _signature(doc)
        # Only real column lists group tables; a table with no known columns stands alone.
        groups.setdefault(signature if signature[1] else (id(doc),), []).append(doc)

    lines: list[str] = []
    for members in groups.values():
        keys = [table_key(d) for d in members]
        first = members[0]
        others = f"\n  Also {len(members) - 1} more table(s) with the same columns: {', '.join(keys[1:])}" if len(members) > 1 else ""
        if spotlight is None or any(k in spotlight for k in keys):
            lines.append(f"- ```{first}```{others}{_found_by(keys, found_by)}")
            continue
        columns = [str(c) for c in getattr(first.content, "columns", [])]
        shown = ", ".join(columns[:_MAX_COMPACT_COLUMNS]) + (", …" if len(columns) > _MAX_COMPACT_COLUMNS else "")
        description = first.metadata.get("description") or ""
        name = keys[0] if len(keys) == 1 else f"{len(keys)} tables with the same columns ({', '.join(keys)})"
        lines.append(
            f"- {name}{f': {description}' if description else ''}; columns: {shown}"
            f"{_found_by(keys, found_by)} [no sample rows shown]"
        )
    return "\n".join(lines)
