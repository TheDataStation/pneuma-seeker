"""
Downloads the benchmark datasets (from KramaBench, see data_src/README.md) into
DATASETS_ROOT. Each dataset is one zip, <dataset>.zip, in a shared Google Drive
folder (DRIVE_FOLDER), holding its tables as
tables/<table>/<table>.parquet and its table descriptions as metadata.csv; the
tables are extracted to <DATASETS_ROOT>/<dataset>/ and the descriptions are
stored in Postgres (ones already there are kept).

    python -m pneuma_seeker.services.db.datasets.download [DATASET ...] [--from DIR] [--force]
    python -m pneuma_seeker.services.db.datasets.download pack [DATASET ...]

`pack` builds those zips from the local datasets, into <DATASETS_ROOT>/.archives.
"""

import argparse
import html
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from pneuma_seeker.services.db.datasets.migrate import import_descriptions
from pneuma_seeker.services.db.datasets.registry import is_local_folder
from pneuma_seeker.services.db.datasets.uploads import dataset_lock, staging_dir
from pneuma_seeker.services.db.pneuma_db import PneumaDB
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger

REPO_ROOT = Path(__file__).resolve().parents[5]

# Shared Google Drive folder holding <dataset>.zip for each dataset.
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1xZhw6ITGIq2stEJun6spglWgPei4AxD3?usp=sharing"

_FOLDER_ID = re.compile(r"folders/([\w-]+)")
_FOLDER_ENTRY = re.compile(
    r'href="https://drive\.google\.com/file/d/([\w-]+)/view[^"]*"[^>]*>.*?<div class="flip-entry-title">(.*?)</div>',
    re.S,
)
_DRIVE_URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"


def drive_folder_files(folder_link: str) -> dict[str, str]:
    """Returns {file name: file ID} of a public Drive folder (read from its embedded view; at most 50 files)."""
    match = _FOLDER_ID.search(folder_link)
    folder_id = match.group(1) if match else folder_link.strip()
    with urllib.request.urlopen(f"https://drive.google.com/embeddedfolderview?id={folder_id}") as response:
        page = response.read().decode("utf-8", "replace")
    return {html.unescape(name).strip(): file_id for file_id, name in _FOLDER_ENTRY.findall(page)}


def pack(datasets_root: Path, name: str, out_dir: Path, metadata_root: Path) -> Path:
    """Builds <out_dir>/<name>.zip from a local dataset and data_src/<name>/metadata.csv."""
    dataset_dir = datasets_root / name
    if not is_local_folder(dataset_dir):
        raise SystemExit(f"{dataset_dir} isn't a dataset folder.")
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:  # Parquet is already compressed
        for file in sorted(dataset_dir.glob("*/*.parquet")):
            zf.write(file, f"tables/{file.parent.name}/{file.name}")
        metadata = metadata_root / name / "metadata.csv"
        if metadata.is_file():
            zf.write(metadata, "metadata.csv")
    return archive


def fetch(name: str, source_dir: Path | None, drive_files: dict[str, str], into: Path) -> Path:
    """Returns the dataset's zip: from `source_dir` if given, else downloaded from the Drive folder."""
    if source_dir is not None:
        archive = source_dir / f"{name}.zip"
        if not archive.is_file():
            raise SystemExit(f"{archive} not found.")
        return archive
    file_id = drive_files.get(f"{name}.zip")
    if not file_id:
        raise SystemExit(f"No {name}.zip in the Drive folder ({', '.join(sorted(drive_files)) or 'empty'}).")
    archive = into / f"{name}.zip"
    with urllib.request.urlopen(_DRIVE_URL.format(file_id)) as response, open(archive, "wb") as out:
        shutil.copyfileobj(response, out)
    if not zipfile.is_zipfile(archive):
        raise SystemExit(f"Downloading {name!r} didn't return a zip: is the Drive folder shared with anyone with the link?")
    return archive


def install(archive: Path, datasets_root: Path, name: str, scratch: Path) -> Path | None:
    """Extracts a dataset zip into <datasets_root>/<name>/; returns its metadata.csv (moved to `scratch`), if any."""
    with dataset_lock(datasets_root, name):
        staging = staging_dir(datasets_root)
        try:
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    if member.startswith("/") or ".." in Path(member).parts:
                        raise SystemExit(f"{archive.name} has an unsafe path: {member}")
                zf.extractall(staging)
            tables = staging / "tables"
            if not is_local_folder(tables):
                raise SystemExit(f"{archive.name} has no tables/<table>/<table>.parquet files.")
            target = datasets_root / name
            if target.exists():
                target.rename(staging / ".old")
            tables.rename(target)
            metadata = staging / "metadata.csv"
            if not metadata.is_file():
                return None
            kept = scratch / f"{name}.metadata.csv"
            shutil.move(metadata, kept)
            return kept
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="'pack', then datasets (default: all)")
    parser.add_argument("--root", help="datasets root (default: Config.DATASETS_ROOT)")
    parser.add_argument("--from", dest="source", help="local folder holding the <dataset>.zip files, instead of Drive")
    parser.add_argument("--folder", default=DRIVE_FOLDER, help="Google Drive folder link (default: DRIVE_FOLDER)")
    parser.add_argument("--force", action="store_true", help="replace datasets that already exist")
    args = parser.parse_args()

    config = Config()
    root = Path(args.root or config.DATASETS_ROOT)
    names = list(args.names)
    if names[:1] == ["pack"]:
        for name in names[1:] or sorted(p.name for p in root.iterdir() if is_local_folder(p)):
            archive = pack(root, name, root / ".archives", REPO_ROOT / "data_src")
            print(f"{name}: {archive} ({archive.stat().st_size / 1e6:.0f} MB)")
        return

    pneuma_db = PneumaDB(config, setup_logger("Dataset Download"), dataset_db_path=root.as_posix())
    registry = pneuma_db.dataset_manager.registry
    source = Path(args.source) if args.source else None

    def present(name: str) -> bool:
        return is_local_folder(root / name) and not args.force

    # The Drive folder is only listed when there's something to download.
    drive_files: dict[str, str] = {}
    if source is None and (not names or not all(present(n) for n in names)):
        drive_files = drive_folder_files(args.folder)
    available = sorted(p.stem for p in source.glob("*.zip")) if source else sorted(
        n.removesuffix(".zip") for n in drive_files if n.endswith(".zip")
    )
    wanted = names or available

    skipped = [n for n in wanted if present(n)]
    for name in skipped:
        registry.register_local(name, created_by="download")  # a no-op if already registered
    if skipped:
        print(f"Already registered: {', '.join(skipped)} (use --force to download again).")

    missing = [n for n in wanted if n not in skipped]
    if missing and pneuma_db.memory_manager is None:
        print("Postgres is unavailable: table descriptions won't be stored (re-run with --force once it's up).")
    with tempfile.TemporaryDirectory() as downloads:
        for name in missing:
            archive = fetch(name, source, drive_files, Path(downloads))
            metadata = install(archive, root, name, Path(downloads))
            added = import_descriptions(pneuma_db, name, metadata) if metadata else 0
            registry.register_local(name, created_by="download")
            tables = len(list((root / name).iterdir()))
            print(f"{name}: {tables} tables, {added} descriptions added")

if __name__ == "__main__":
    main()
