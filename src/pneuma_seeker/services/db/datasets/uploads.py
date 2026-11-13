"""
Uploads turn CSV, TSV, Parquet and Excel (.xlsx) files into tables of a local
dataset, stored as <root>/<dataset>/<table>/<table>.parquet with cleaned column
names. Every file is converted in a staging folder first and then moved into
place under a per-dataset lock, so a failed upload changes nothing and readers
never see a half-written table.
"""

import fcntl
import shutil
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import duckdb

from pneuma_seeker.services.db.datasets.connectors import quote_ident, sql_literal
from pneuma_seeker.services.db.datasets.connectors.base import clean_names
from pneuma_seeker.services.db.datasets.registry import NAME_RE, NAME_RULE

FORMATS = {".csv": "csv", ".tsv": "tsv", ".parquet": "parquet", ".parq": "parquet", ".xlsx": "excel"}
ACCEPTED = "CSV, TSV, Parquet and Excel (.xlsx)"

# HTTP status of each refusal kind.
STATUS = {"invalid": 400, "not_found": 404, "table_exists": 409, "too_large": 413, "insufficient_storage": 507}


class UploadRefused(Exception):
    """Raised for an upload or deletion that can't be done; `kind` is a key of STATUS."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass
class UploadedTable:
    name: str
    rows: int
    columns: list[str]
    replaced: bool


@dataclass
class UploadResult:
    dataset_name: str
    created_dataset: bool
    tables: list[UploadedTable] = field(default_factory=list)


def file_format(file_name: str) -> str:
    """Returns the format of an uploaded file from its extension. Raises UploadRefused."""
    suffix = Path(file_name).suffix.lower()
    if suffix == ".xls":
        raise UploadRefused("invalid", "Old Excel files (.xls) can't be uploaded: save the file as .xlsx first.")
    if suffix not in FORMATS:
        raise UploadRefused("invalid", f"{file_name!r} can't be uploaded: only {ACCEPTED} files are accepted.")
    return FORMATS[suffix]


def table_name_for(name: str) -> str:
    """Returns a table name from a file stem or sheet name."""
    cleaned = clean_names([name])[0]
    if not cleaned[0].isalpha():
        cleaned = f"t_{cleaned}"
    return cleaned[:63]


def staging_dir(root: Path) -> Path:
    """Creates and returns a fresh staging folder under the datasets root."""
    path = Path(root) / ".staging" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


@contextmanager
def dataset_lock(root: Path, dataset_name: str):
    """Holds an exclusive lock on a dataset's files, across threads and processes."""
    locks = Path(root) / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with open(locks / f"{dataset_name}.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def add_file(
    root: Path,
    dataset_name: str,
    file: Path,
    file_name: str,
    *,
    create: bool = False,
    replace: bool = False,
) -> UploadResult:
    """
    Adds the tables in `file` (one per file, or one per non-empty Excel sheet) to
    a local dataset, creating the dataset if `create`. Raises UploadRefused.
    """
    fmt = file_format(file_name)
    if not NAME_RE.match(dataset_name):
        raise UploadRefused("invalid", f"Invalid dataset name {dataset_name!r}: use {NAME_RULE}.")
    root = Path(root)
    dataset_dir = root / dataset_name
    with dataset_lock(root, dataset_name):
        created = not dataset_dir.is_dir()
        if created and not create:
            raise UploadRefused("not_found", f"No local dataset named {dataset_name!r}.")
        staging = staging_dir(root)
        try:
            con = duckdb.connect()
            try:
                relations = _relations(con, file, file_name, fmt)
                existing = {p.name for p in dataset_dir.iterdir()} if dataset_dir.is_dir() else set()
                taken = sorted(name for name, _ in relations if name in existing)
                if taken and not replace:
                    raise UploadRefused(
                        "table_exists",
                        f"Table(s) {', '.join(taken)} already exist in {dataset_name!r}; replace them to continue.",
                    )
                tables = [_convert(con, name, select, staging) for name, select in relations]
            finally:
                con.close()
            dataset_dir.mkdir(parents=True, exist_ok=True)
            for table in tables:
                target = dataset_dir / table.name
                table.replaced = target.exists()
                if table.replaced:
                    target.rename(staging / f".old-{table.name}")
                (staging / table.name).rename(target)
            return UploadResult(dataset_name, created, tables)
        except UploadRefused:
            if created and dataset_dir.exists() and not any(dataset_dir.iterdir()):
                dataset_dir.rmdir()
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def delete_table(root: Path, dataset_name: str, table_name: str) -> None:
    """Deletes one table of a local dataset. Raises UploadRefused."""
    root = Path(root)
    with dataset_lock(root, dataset_name):
        target = root / dataset_name / table_name
        if not NAME_RE.match(table_name) or not target.is_dir():
            raise UploadRefused("not_found", f"No table {table_name!r} in {dataset_name!r}.")
        shutil.rmtree(target)


def delete_dataset_files(root: Path, dataset_name: str) -> None:
    """Deletes a local dataset's folder."""
    root = Path(root)
    with dataset_lock(root, dataset_name):
        shutil.rmtree(root / dataset_name, ignore_errors=True)


def _relations(con: duckdb.DuckDBPyConnection, file: Path, file_name: str, fmt: str) -> list[tuple[str, str]]:
    path = sql_literal(file.as_posix())
    stem = Path(file_name).stem
    if fmt == "parquet":
        return [(table_name_for(stem), f"SELECT * FROM read_parquet({path}, binary_as_string=true)")]
    if fmt in ("csv", "tsv"):
        delim = ", delim='\\t'" if fmt == "tsv" else _csv_dialect(con, file)
        return [
            (
                table_name_for(stem),
                f"SELECT * FROM read_csv({path}, header=true, ignore_errors=true, strict_mode=false, "
                f"null_padding=true, sample_size=100000, parallel=false{delim})",
            )
        ]
    con.execute("INSTALL excel; LOAD excel")
    found = []
    for sheet in _sheet_names(file):
        select = f"SELECT * FROM read_xlsx({path}, sheet={sql_literal(sheet)})"
        try:
            con.execute(f"DESCRIBE {select}")
        except duckdb.Error:
            continue  # an empty sheet is no table
        found.append((table_name_for(sheet), select))
    if not found:
        raise UploadRefused("invalid", f"{file_name!r} has no sheet with data.")
    names = [name for name, _ in found]
    if len(set(names)) != len(names):
        raise UploadRefused("invalid", f"Sheets of {file_name!r} map to the same table name; rename them first.")
    return found


def _csv_dialect(con: duckdb.DuckDBPyConnection, file: Path) -> str:
    """
    Returns options pinning the CSV dialect: with strict_mode off, DuckDB's
    sniffer can settle on a quoting that silently drops rows, so it's sniffed
    strictly first.
    """
    try:
        row = con.execute("SELECT Delimiter, Quote, Escape FROM sniff_csv(?)", [file.as_posix()]).fetchone()
    except duckdb.Error:
        return ""
    options = [
        f"{name}={sql_literal(value)}"
        for name, value in zip(("delim", "quote", "escape"), row or ())
        if isinstance(value, str) and len(value) == 1  # absent settings are reported as e.g. "(empty)"
    ]
    return "".join(f", {option}" for option in options)


def _sheet_names(file: Path) -> list[str]:
    try:
        with zipfile.ZipFile(file) as archive:
            workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError):
        raise UploadRefused("invalid", "This isn't a valid Excel (.xlsx) file.")
    return [el.attrib["name"] for el in workbook.iter() if el.tag.endswith("}sheet") and "name" in el.attrib]


def _convert(con: duckdb.DuckDBPyConnection, table: str, select: str, staging: Path) -> UploadedTable:
    try:
        columns = [row[0] for row in con.execute(f"DESCRIBE {select}").fetchall()]
        cleaned = clean_names(columns)
        select_list = ", ".join(f"{quote_ident(c)} AS {quote_ident(n)}" for c, n in zip(columns, cleaned))
        out = staging / table / f"{table}.parquet"
        out.parent.mkdir()
        con.execute(
            f"COPY (SELECT {select_list} FROM ({select})) TO {sql_literal(out.as_posix())} "
            "(FORMAT parquet, COMPRESSION zstd)"
        )
        (rows,) = con.execute(f"SELECT count(*) FROM read_parquet({sql_literal(out.as_posix())})").fetchone()
    except duckdb.Error as exc:
        raise UploadRefused("invalid", f"Couldn't read table {table!r}: {exc}")
    return UploadedTable(table, int(rows), cleaned, replaced=False)
