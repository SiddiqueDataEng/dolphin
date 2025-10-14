from pathlib import Path

import pandas as pd

from hospi.data_ingestion import DataIngestor
from hospi.data_cleaning import DataCleaner
from hospi.data_validation import DataValidator
from hospi.bi_questions import generate_bi_questions, run_sql_over_dataframe

DATA_FILE = Path("sample sales.csv")


def head_preview(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
	return df.head(n)


def main() -> None:
	if not DATA_FILE.exists():
		print(f"Data file not found: {DATA_FILE}")
		return

	required_cols = ["orderdate", "ordernumber", "ordertype", "sales", "site type", "site", "region"]
	raw_df, report = DataIngestor.load_csv_data(
		DATA_FILE,
		use_columns=required_cols,
		parse_dates=["orderdate"],
		day_first=True,
		required_columns=required_cols,
	)
	if not report.is_valid:
		print("Missing required columns:", report.missing_columns)
		return

	print(f"Loaded data with encoding={report.encoding}")
	print("Sample preview:")
	print(head_preview(raw_df))

	# Cleaning
	clean_df, clean_summary = DataCleaner.clean_dataset(raw_df, missing_strategy="flag")
	print("\nCleaning summary:")
	print({
		"rows_before": clean_summary.rows_before,
		"rows_after": clean_summary.rows_after,
		"null_sales_before": clean_summary.null_sales_before,
		"null_sales_after": clean_summary.null_sales_after,
		"negative_sales_count": clean_summary.negative_sales_count,
		"duplicates_removed": clean_summary.duplicates_removed,
	})

	# Validation
	dq = DataValidator.generate_data_quality_report(clean_df, required_columns=required_cols)
	print("\nData quality report:")
	print({
		"row_count": dq.row_count,
		"null_sales": dq.null_sales,
		"negative_sales": dq.negative_sales,
		"future_dates": dq.future_dates,
		"missing_required": dq.missing_required,
		"issues": dq.issues,
	})

	# BI: run a few sample queries on cleaned data
	bi_list = generate_bi_questions()
	for q in bi_list[:5]:
		print(f"\n[BI] {q.title}")
		res = run_sql_over_dataframe(clean_df, q.sql)
		print(res.head())


if __name__ == "__main__":
	main()
