# src/pneuma_seeker/shared/parser.py
import json
import re

from ast import literal_eval


def parse_json(json_string: str) -> dict:
    """
    Parses a JSON string, removing any surrounding code block markers
    if present. Raises ValueError if the input is not a string or if the
    JSON is invalid.
    """
    if not isinstance(json_string, str):
        raise ValueError("Input must be a string")
    json_string = json_string.strip()
    json_string = re.sub(r"^```(?:json)?\s*", "", json_string)
    json_string = re.sub(r"\s*```$", "", json_string)
    json_string = json_string.strip()

    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(json_string)
    except json.JSONDecodeError:
        # Be tolerant of accidental prefix/suffix text by extracting the first JSON
        # object/array-looking region.
        match = re.search(r"[\[{]", json_string)
        if not match:
            raise ValueError("Invalid JSON string")
        try:
            obj, _end = decoder.raw_decode(json_string[match.start() :])
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON string") from exc

    if not isinstance(obj, dict):
        raise ValueError("Invalid JSON string")
    return obj


def parse_sql(sql_string: str) -> str:
    """
    Parses SQL string
    """
    if sql_string.startswith("```"):
        sql_string = sql_string[3:]
    if sql_string.endswith("```"):
        sql_string = sql_string[:-3]
    if sql_string.startswith("sql"):
        sql_string = sql_string[3:]
    return sql_string


def parse_code(code: str):
    """
    Parses python code
    """
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    if code.startswith("python"):
        code = code[6:]
    return code


def augmented_literal_eval(text: str):
    """
    Evaluates a Python literal expression from a string, removing any surrounding
    code block markers if present.
    """
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    if text.startswith("python"):
        text = text[6:]
    return literal_eval(text)
