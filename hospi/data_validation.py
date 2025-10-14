from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
	row_count: int
	null_sales: int
	negative_sales: int
	future_dates: int
	missing_required: List[str]
	issues: List[str]


class DataValidator:
	@staticmethod
	def validate_sales_ranges(df: pd.DataFrame) -> Dict[str, int]:
		null_sales = int(df["sales"].isna().sum()) if "sales" in df.columns else 0
		negative_sales = int((df["sales"] < 0).sum()) if "sales" in df.columns else 0
		return {"null_sales": null_sales, "negative_sales": negative_sales}

	@staticmethod
	def check_date_ranges(df: pd.DataFrame) -> Dict[str, int]:
		if "orderdate" not in df.columns:
			return {"future_dates": 0}
		future_dates = int((df["orderdate"] > pd.Timestamp(date.today())).sum())
		return {"future_dates": future_dates}

	@staticmethod
	def validate_business_rules(df: pd.DataFrame) -> List[str]:
		issues: List[str] = []
		if "site" in df.columns:
			ws = df["site"].astype(str)
			if ((ws.str.startswith(" ")) | (ws.str.endswith(" "))).any():
				issues.append("Sites contain leading/trailing whitespace")
		if "ordertype" in df.columns and df["ordertype"].isna().any():
			issues.append("Missing ordertype values present")
		return issues

	@staticmethod
	def generate_data_quality_report(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> ValidationReport:
		required_columns = required_columns or ["orderdate", "ordernumber", "ordertype", "sales", "site type", "site", "region"]
		missing_required = [c for c in required_columns if c not in df.columns]
		sales = DataValidator.validate_sales_ranges(df)
		date_checks = DataValidator.check_date_ranges(df)
		issues = DataValidator.validate_business_rules(df)
		return ValidationReport(
			row_count=len(df),
			null_sales=sales["null_sales"],
			negative_sales=sales["negative_sales"],
			future_dates=date_checks["future_dates"],
			missing_required=missing_required,
			issues=issues,
		)
