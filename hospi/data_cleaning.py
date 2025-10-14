from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd


MissingStrategy = Literal["flag", "drop", "zero"]


@dataclass(frozen=True)
class CleaningSummary:
	rows_before: int
	rows_after: int
	null_sales_before: int
	null_sales_after: int
	negative_sales_count: int
	duplicates_removed: int


class DataCleaner:
	@staticmethod
	def fix_data_types(df: pd.DataFrame, date_format: str = "%d/%m/%Y") -> pd.DataFrame:
		df = df.copy()
		if "orderdate" in df.columns:
			df["orderdate"] = pd.to_datetime(df["orderdate"], errors="coerce", dayfirst=True)
		if "sales" in df.columns:
			df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
		return df

	@staticmethod
	def standardize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
		df = df.copy()
		if "site" in df.columns:
			df["site"] = df["site"].astype(str).str.strip()
		if "ordertype" in df.columns:
			df["ordertype"] = df["ordertype"].astype(str).str.strip().str.lower()
		if "site type" in df.columns:
			df["site type"] = df["site type"].astype(str).str.strip().str.lower()
		if "region" in df.columns:
			df["region"] = df["region"].astype(str).str.strip().str.lower()
		return df

	@staticmethod
	def handle_missing_values(df: pd.DataFrame, strategy: MissingStrategy = "flag") -> pd.DataFrame:
		df = df.copy()
		if strategy == "flag":
			df["is_missing_sales"] = df["sales"].isna()
		elif strategy == "drop":
			df = df[~df["sales"].isna()].copy()
		elif strategy == "zero":
			df.loc[df["sales"].isna(), "sales"] = 0.0
		return df

	@staticmethod
	def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
		cols = [c for c in ["orderdate", "ordernumber", "site"] if c in df.columns]
		if not cols:
			return df
		return df.drop_duplicates(subset=cols)

	@staticmethod
	def fix_sales_anomalies(df: pd.DataFrame) -> pd.DataFrame:
		df = df.copy()
		# Add a transaction_type classification
		if "sales" in df.columns:
			df["transaction_type"] = pd.Series(pd.NA, index=df.index, dtype="string")
			df.loc[df["sales"].isna(), "transaction_type"] = "missing"
			df.loc[df["sales"] > 0, "transaction_type"] = "sale"
			df.loc[df["sales"] < 0, "transaction_type"] = "refund"
		return df

	@staticmethod
	def clean_dataset(df: pd.DataFrame, missing_strategy: MissingStrategy = "flag") -> tuple[pd.DataFrame, CleaningSummary]:
		rows_before = len(df)
		null_before = int(df["sales"].isna().sum()) if "sales" in df.columns else 0
		neg_count = int((df["sales"] < 0).sum()) if "sales" in df.columns else 0

		df1 = DataCleaner.fix_data_types(df)
		df2 = DataCleaner.standardize_categorical_values(df1)
		df3 = DataCleaner.handle_missing_values(df2, strategy=missing_strategy)
		df4 = DataCleaner.remove_duplicates(df3)
		dupes_removed = rows_before - len(df4)
		df5 = DataCleaner.fix_sales_anomalies(df4)

		null_after = int(df5["sales"].isna().sum()) if "sales" in df5.columns else 0
		summary = CleaningSummary(
			rows_before=rows_before,
			rows_after=len(df5),
			null_sales_before=null_before,
			null_sales_after=null_after,
			negative_sales_count=neg_count,
			duplicates_removed=dupes_removed,
		)
		return df5, summary
