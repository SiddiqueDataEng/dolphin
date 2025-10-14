from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class IngestionReport:
	is_valid: bool
	missing_columns: List[str]
	row_count: Optional[int] = None
	encoding: Optional[str] = None


class DataIngestor:
	DEFAULT_NA_VALUES = ["NULL", "#VALUE!", "N/A", "na", "", " "]

	@staticmethod
	def detect_encoding(file_path: str | Path, sample_size_bytes: int = 262144) -> str:
		"""Detect file encoding using chardet on a byte sample."""
		import chardet  # lazy import
		with open(file_path, "rb") as f:
			data = f.read(sample_size_bytes)
			result = chardet.detect(data)
			encoding = result.get("encoding") or "utf-8"
			return encoding

	@staticmethod
	def validate_file_structure(df: pd.DataFrame, required_columns: Iterable[str]) -> IngestionReport:
		required = list(required_columns)
		missing = [c for c in required if c not in df.columns]
		return IngestionReport(is_valid=len(missing) == 0, missing_columns=missing, row_count=len(df))

	@staticmethod
	def load_csv_data(
		file_path: str | Path,
		use_columns: Optional[List[str]] = None,
		chunk_size: Optional[int] = None,
		parse_dates: Optional[List[str]] = None,
		day_first: bool = True,
		dtype_overrides: Optional[dict] = None,
		required_columns: Optional[Iterable[str]] = None,
	) -> Tuple[pd.DataFrame, IngestionReport]:
		"""
		Load CSV with robust defaults: detect encoding, parse dates, coerce errors, and validate columns.
		If chunk_size is provided, only the first chunk is returned for preview; use handle_large_files for streaming.
		"""
		encoding = DataIngestor.detect_encoding(file_path)
		kwargs = {
			"encoding": encoding,
			"na_values": DataIngestor.DEFAULT_NA_VALUES,
			"keep_default_na": True,
			"on_bad_lines": "skip",
			"low_memory": False,
		}
		if parse_dates:
			kwargs["parse_dates"] = parse_dates
			kwargs["dayfirst"] = day_first
		if use_columns:
			kwargs["usecols"] = use_columns
		if dtype_overrides:
			kwargs["dtype"] = dtype_overrides

		if chunk_size:
			kwargs["chunksize"] = chunk_size
			chunk_iter = pd.read_csv(file_path, **kwargs)
			first_chunk = next(iter(chunk_iter))
			report = DataIngestor.validate_file_structure(first_chunk, required_columns or (use_columns or first_chunk.columns))
			# Attempt to coerce numeric types for known fields
			if "sales" in first_chunk.columns:
				first_chunk["sales"] = pd.to_numeric(first_chunk["sales"], errors="coerce")
			return first_chunk, IngestionReport(report.is_valid, report.missing_columns, row_count=None, encoding=encoding)

		df = pd.read_csv(file_path, **kwargs)
		# Coerce numeric fields
		if "sales" in df.columns:
			df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
		report = DataIngestor.validate_file_structure(df, required_columns or (use_columns or df.columns))
		return df, IngestionReport(report.is_valid, report.missing_columns, row_count=len(df), encoding=encoding)

	@staticmethod
	def handle_large_files(
		file_path: str | Path,
		chunk_size: int = 100000,
		parse_dates: Optional[List[str]] = None,
		day_first: bool = True,
	) -> Generator[pd.DataFrame, None, None]:
		"""Yield DataFrame chunks for very large files."""
		encoding = DataIngestor.detect_encoding(file_path)
		for chunk in pd.read_csv(
			file_path,
			chunksize=chunk_size,
			encoding=encoding,
			na_values=DataIngestor.DEFAULT_NA_VALUES,
			keep_default_na=True,
			on_bad_lines="skip",
			low_memory=False,
			parse_dates=parse_dates or None,
			dayfirst=day_first,
		):
			if "sales" in chunk.columns:
				chunk["sales"] = pd.to_numeric(chunk["sales"], errors="coerce")
			yield chunk
