from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import pandas as pd

from pneuma_seeker.services.indexing.connectors.base import SourceConnector
from pneuma_seeker.shared.str_processor import clean_column_table_name


class CSVConnector(SourceConnector):
	"""Connector for CSV file or directory sources."""

	def __init__(self, config: dict[str, Any]):
		super().__init__(config)
		self.file_path: str | None = config.get("file_path")
		self.directory_path: str = config.get("directory_path", "")

		if not self.file_path and not self.directory_path:
			raise ValueError("CSV connector requires 'file_path' or 'directory_path'.")

	@property
	def source_type(self) -> str:
		return "csv"

	def _resolve_stream_map(self) -> dict[str, Path]:
		stream_map: dict[str, Path] = {}

		if self.file_path:
			file = Path(self.file_path).expanduser().resolve()
			stream_map[clean_column_table_name(file.stem)] = file
			return stream_map

		directory = Path(self.directory_path).expanduser().resolve()
		csv_files = sorted(directory.glob("*.csv"))

		dedupe: dict[str, int] = {}
		for file in csv_files:
			base_name = clean_column_table_name(file.stem)
			if base_name not in dedupe:
				dedupe[base_name] = 0
				stream_name = base_name
			else:
				dedupe[base_name] += 1
				stream_name = f"{base_name}_{dedupe[base_name]}"
			stream_map[stream_name] = file

		return stream_map

	def check_connection(self) -> bool:
		try:
			stream_map = self._resolve_stream_map()
			if not stream_map:
				return False
			return all(path.exists() and path.is_file() for path in stream_map.values())
		except Exception:
			return False

	def discover(self) -> list[dict[str, Any]]:
		stream_map = self._resolve_stream_map()
		return [
			{
				"stream": stream,
				"table_name": stream,
				"path": path.as_posix(),
			}
			for stream, path in stream_map.items()
		]

	def read(self, stream: str) -> Generator[dict[str, Any]]:
		stream_map = self._resolve_stream_map()
		if stream not in stream_map:
			raise KeyError(f"Unknown CSV stream: {stream}")

		table = pd.read_csv(stream_map[stream])
		for row in table.to_dict(orient="records"):
			yield row # type: ignore
