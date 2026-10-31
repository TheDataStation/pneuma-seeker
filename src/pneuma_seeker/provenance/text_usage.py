"""Detects whether a step used a non-tabular document (e.g., a web search result).

The Materializer only sees web results as text in its context, so a step "uses"
one by transcribing values from it into its code (e.g., a hard-coded mapping in
SQL/Python) or into its arguments (e.g., an LLM instruction). Usage is detected
by looking for the step's literal values in the document's text.
"""

import re

# Single- and double-quoted literals are scanned separately so that literals
# nested in another string (e.g., SQL literals inside a Python string) are found.
_STRING_LITERAL_RES = (re.compile(r"'([^'\n]*)'"), re.compile(r'"([^"\n]*)"'))
_NUMBER_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])")
_WHITESPACE_RE = re.compile(r"\s+")

# Distinct values that must be found in the document before the step is
# considered to use it; a single coincidental match (e.g., a common word or a
# year) is not enough.
MIN_TRANSCRIBED_VALUES = 2


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def _normalize_number(number: str) -> str:
    number = number.replace(",", "")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def _is_informative_number(number: str) -> bool:
    digits = number.replace(".", "").lstrip("0")
    # Ignore small numbers and round numbers (0, 1, 10, 100, 1000, ...): they
    # are typical of code (limits, factors) rather than transcribed data.
    return len(digits) >= 3 and digits.strip("0") not in ("", "1")


def find_transcribed_values(
    step_text: str, document_text: str, identifiers: set[str] | None = None
) -> list[str]:
    """Returns the literal values of `step_text` (string and numeric literals)
    that also appear in `document_text`.

    Args:
        step_text: Code or arguments of a step.
        document_text: Content of the non-tabular document.
        identifiers: Names (tables, columns) to ignore, since they appear in
            code as quoted identifiers rather than as values.
    """
    if not step_text or not document_text:
        return []
    ignored = {_normalize_text(i) for i in identifiers or set()}
    doc_text = _normalize_text(document_text)
    doc_numbers = {_normalize_number(n) for n in _NUMBER_RE.findall(document_text)}

    found: list[str] = []
    literals = [m for regex in _STRING_LITERAL_RES for m in regex.findall(step_text)]
    for literal in literals:
        value = _normalize_text(literal)
        if (
            len(value) >= 4
            and any(c.isalpha() for c in value)
            and value not in ignored
            and value not in found
            and re.search(rf"(?<!\w){re.escape(value)}(?!\w)", doc_text)
        ):
            found.append(value)
    # Numbers are matched as whole numbers (commas and trailing zeros ignored).
    for number in _NUMBER_RE.findall(step_text):
        value = _normalize_number(number)
        if _is_informative_number(value) and value in doc_numbers and value not in found:
            found.append(value)
    return found


def uses_document(
    step_text: str, document_text: str, identifiers: set[str] | None = None
) -> list[str]:
    """Values transcribed from the document if the step uses it, else []."""
    found = find_transcribed_values(step_text, document_text, identifiers)
    return found if len(found) >= MIN_TRANSCRIBED_VALUES else []
