from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import sys

# Ensure project root is on sys.path for Streamlit runs
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
	sys.path.insert(0, str(_project_root))

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from hospi.data_ingestion import DataIngestor
from hospi.data_cleaning import DataCleaner
from hospi.data_validation import DataValidator
from hospi.bi_questions import BIQuery, generate_bi_questions, run_sql_over_dataframe

DATA_FILE = Path("sample sales.csv")


@st.cache_data(show_spinner=False)
def _load_clean_data() -> pd.DataFrame:
	required_cols = ["orderdate", "ordernumber", "ordertype", "sales", "site type", "site", "region"]
	df, _ = DataIngestor.load_csv_data(
		DATA_FILE,
		use_columns=required_cols,
		parse_dates=["orderdate"],
		day_first=True,
		required_columns=required_cols,
	)
	clean_df, _ = DataCleaner.clean_dataset(df, missing_strategy="flag")
	return clean_df


def _load_raw_orderdate_strings() -> pd.Series:
	# Load orderdate strictly as raw string to detect mixed formats
	df_raw, _ = DataIngestor.load_csv_data(
		DATA_FILE,
		use_columns=["orderdate"],
		parse_dates=None,
		day_first=True,
		required_columns=["orderdate"],
	)
	return df_raw["orderdate"].astype(str)


def _init_session() -> None:
	if "original_df" not in st.session_state:
		st.session_state["original_df"] = _load_clean_data()
		try:
			raw_series = _load_raw_orderdate_strings()
			if len(raw_series) == len(st.session_state["original_df"]):
				st.session_state["original_df"]["orderdate_raw"] = raw_series.values
		except Exception:
			pass
	if "transform_queue" not in st.session_state:
		st.session_state["transform_queue"] = []  # list of (id, description, func)
	if "refined_df" not in st.session_state:
		st.session_state["refined_df"] = st.session_state["original_df"].copy()


def load_data() -> pd.DataFrame:
	_init_session()
	return st.session_state["refined_df"]


@st.cache_data(show_spinner=False)
def bi_catalog() -> List[BIQuery]:
	return generate_bi_questions()


# ===================== Transform Queue Utilities =====================

def queue_transform(tid: str, description: str, func: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
	# replace if exists
	st.session_state["transform_queue"] = [t for t in st.session_state["transform_queue"] if t[0] != tid]
	st.session_state["transform_queue"].append((tid, description, func))


def clear_transform(tid: str) -> None:
	st.session_state["transform_queue"] = [t for t in st.session_state["transform_queue"] if t[0] != tid]


def apply_all_transforms() -> None:
	df = st.session_state["original_df"].copy()
	for tid, _desc, func in st.session_state["transform_queue"]:
		df = func(df)
	st.session_state["refined_df"] = df


# ===================== Deep Profiling Helpers =====================

def dataset_summary(df: pd.DataFrame) -> Dict[str, object]:
	mem_bytes = df.memory_usage(deep=True).sum()
	return {
		"rows": int(len(df)),
		"columns": int(df.shape[1]),
		"memory_mb": round(mem_bytes / (1024 * 1024), 2),
		"date_min": str(pd.to_datetime(df["orderdate"]).min()) if "orderdate" in df.columns else "n/a",
		"date_max": str(pd.to_datetime(df["orderdate"]).max()) if "orderdate" in df.columns else "n/a",
	}

# Date format validation helpers

def detect_invalid_mmdd(series: pd.Series) -> pd.DataFrame:
	"""Detect rows not conforming to MM/DD/YYYY. Returns DataFrame with parsed parts and reason."""
	s = series.fillna("").astype(str).str.strip()
	parts = s.str.extract(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
	parts.columns = ["m1", "d1", "y1"]
	valid_pattern = parts.notna().all(axis=1)
	# Initialize defaults
	res = pd.DataFrame({
		"raw": s,
		"m": pd.to_numeric(parts["m1"], errors="coerce"),
		"d": pd.to_numeric(parts["d1"], errors="coerce"),
		"y": pd.to_numeric(parts["y1"], errors="coerce"),
		"pattern_ok": valid_pattern,
	})
	# Valid if pattern ok and 1<=m<=12 and 1<=d<=31
	bounds_ok = (res["m"].between(1, 12, inclusive="both")) & (res["d"].between(1, 31, inclusive="both"))
	res["mmdd_valid"] = res["pattern_ok"] & bounds_ok
	# Swap candidate: day-month swapped (first>12 and second<=12)
	swap_candidate = (res["m"] > 12) & (res["d"].between(1, 12, inclusive="both"))
	res["swap_candidate"] = swap_candidate
	# Reason text
	reason = np.where(~res["pattern_ok"], "bad pattern", np.where(~bounds_ok, "out of bounds", "ok"))
	reason = np.where((reason == "ok") & swap_candidate, "likely dd/mm", reason)
	res["reason"] = reason
	return res


def swap_month_day(raw: pd.Series) -> pd.Series:
	"""Swap first two date parts in strings like A/B/Y -> B/A/Y when possible."""
	s = raw.fillna("").astype(str)
	return s.str.replace(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", r"\2/\1/\3", regex=True)

# Fix functions for queue

def fix_trim_standardize(df: pd.DataFrame) -> pd.DataFrame:
	return apply_fix_trim_standardize(df)


def fix_missing_sales(strategy: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
	return lambda frame: apply_fix_null_sales(frame, strategy)


def fix_dates_swap_invalid(df: pd.DataFrame) -> pd.DataFrame:
	if "orderdate_raw" not in df.columns:
		return df
	dates_eval = detect_invalid_mmdd(df["orderdate_raw"])
	invalid_mask = ~dates_eval["mmdd_valid"]
	new_raw = df["orderdate_raw"].copy()
	new_raw.loc[invalid_mask] = swap_month_day(new_raw.loc[invalid_mask])
	parsed = pd.to_datetime(new_raw, errors="coerce", format="%m/%d/%Y")
	df2 = df.copy()
	df2["orderdate_raw"] = new_raw
	df2["orderdate"] = parsed
	return df2


def cap_outliers_column(col: str, quantile: float) -> Callable[[pd.DataFrame], pd.DataFrame]:
	def _cap(df: pd.DataFrame) -> pd.DataFrame:
		if col not in df.columns:
			return df
		cap = pd.to_numeric(df[col], errors="coerce").quantile(quantile)
		df2 = df.copy()
		df2.loc[df2[col] > cap, col] = cap
		return df2
	return _cap


# ===================== Advanced Profiling & Fixes =====================

def per_column_stats(df: pd.DataFrame) -> pd.DataFrame:
	stats_rows = []
	for col in df.columns:
		series = df[col]
		dtype = str(series.dtype)
		null_count = int(series.isna().sum())
		is_string = series.dtype == object or pd.api.types.is_string_dtype(series)
		blank_count = int((series.astype(str).str.len() == 0).sum()) if is_string else 0
		whitespace_count = int((series.astype(str).str.match(r"^\s+$")).sum()) if is_string else 0
		distinct_count = int(series.nunique(dropna=True))

		row = {
			"column": col,
			"dtype": dtype,
			"null_count": null_count,
			"blank_count": blank_count,
			"whitespace_only": whitespace_count,
			"distinct_count": distinct_count,
		}

		if pd.api.types.is_bool_dtype(series):
			true_count = int(series.fillna(False).sum())
			false_count = int((~series.fillna(False)).sum())
			row.update({
				"true_count": true_count,
				"false_count": false_count,
			})
		elif pd.api.types.is_numeric_dtype(series):
			# Safely coerce to float and drop NaNs for percentile stats
			vals = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
			vals = vals[~np.isnan(vals)]
			if vals.size > 0:
				row.update({
					"min": float(np.nanmin(vals)),
					"p05": float(np.nanpercentile(vals, 5)),
					"p25": float(np.nanpercentile(vals, 25)),
					"median": float(np.nanpercentile(vals, 50)),
					"p75": float(np.nanpercentile(vals, 75)),
					"p95": float(np.nanpercentile(vals, 95)),
					"max": float(np.nanmax(vals)),
					"mean": float(np.nanmean(vals)),
					"std": float(np.nanstd(vals)),
					"zeros": int((series == 0).sum()),
					"negatives": int((series < 0).sum()),
				})
		stats_rows.append(row)
	return pd.DataFrame(stats_rows)


def value_frequencies(df: pd.DataFrame, column: str, top_n: int = 20) -> pd.DataFrame:
	vc = df[column].value_counts(dropna=False).head(top_n)
	out = vc.reset_index()
	# Use safe, generic headers to avoid duplicates regardless of column name
	out.columns = ["value", "frequency"]
	return out


def correlation_heatmap(df: pd.DataFrame, *, key: Optional[str] = None) -> None:
	numeric_df = df.select_dtypes(include=["number"]).copy()
	if numeric_df.empty or numeric_df.shape[1] < 2:
		st.info("Not enough numeric columns for correlation heatmap.")
		return
	corr = numeric_df.corr(numeric_only=True)
	fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
	st.plotly_chart(fig, use_container_width=True, key=key or f"corr_{id(df)}")


def missing_heatmap(df: pd.DataFrame, *, key: Optional[str] = None) -> None:
	miss = df.isna().astype(int)
	if miss.values.sum() == 0:
		st.info("No missing values detected.")
		return
	# Aggregate by column to keep chart readable
	col_missing = miss.sum(axis=0).reset_index()
	col_missing.columns = ["column", "missing"]
	fig = px.bar(col_missing, x="column", y="missing", title="Missing Values by Column")
	st.plotly_chart(fig, use_container_width=True, key=key or f"missing_{id(df)}")


def memory_usage_by_column(df: pd.DataFrame) -> pd.DataFrame:
	mem = df.memory_usage(deep=True)
	return pd.DataFrame({"column": mem.index.astype(str), "memory_bytes": mem.values})


def time_distributions(df: pd.DataFrame, *, key_prefix: Optional[str] = None) -> None:
	if "orderdate" not in df.columns:
		st.info("No orderdate column for time distributions.")
		return
	dd = pd.to_datetime(df["orderdate"])  # safe
	by_day = df.copy()
	by_day["day"] = dd.dt.date
	daily = by_day.groupby("day").agg(net_sales=("sales", lambda x: float(np.nansum(x)))).reset_index()
	st.plotly_chart(px.line(daily, x="day", y="net_sales", title="Daily Net Sales"), use_container_width=True, key=(key_prefix or "time")+"_daily")

	# If time present, derive hour; else skip gracefully
	if hasattr(dd.dt, "hour"):
		by_hour = dd.dt.hour.dropna()
		if not by_hour.empty:
			hist = by_hour.value_counts().sort_index().reset_index()
			hist.columns = ["hour", "count"]
			st.plotly_chart(px.bar(hist, x="hour", y="count", title="Records by Hour"), use_container_width=True, key=(key_prefix or "time")+"_hour")


# ===================== BI Visuals =====================

def kpi_cards(df: pd.DataFrame) -> None:
	con = duckdb.connect()
	con.register("sales", df)
	kpis = con.execute(
		"""
		SELECT
			SUM(CASE WHEN sales > 0 THEN sales ELSE 0 END) AS total_sales,
			SUM(CASE WHEN sales < 0 THEN sales ELSE 0 END) AS total_refunds,
			SUM(COALESCE(sales,0)) AS net_sales,
			AVG(NULLIF(sales, 0)) AS avg_txn,
			COUNT(*) AS txns
		FROM sales
		"""
	).df().iloc[0]
	con.close()

	c1, c2, c3, c4, c5 = st.columns(5)
	c1.metric("Total Sales", f"{kpis.total_sales:,.2f}")
	c2.metric("Total Refunds", f"{kpis.total_refunds:,.2f}")
	c3.metric("Net Sales", f"{kpis.net_sales:,.2f}")
	c4.metric("Avg Transaction", f"{kpis.avg_txn:,.2f}")
	c5.metric("Transactions", f"{int(kpis.txns):,}")


def time_series(df: pd.DataFrame) -> None:
	series = run_sql_over_dataframe(
		df,
		"SELECT DATE_TRUNC('day', orderdate) AS d, SUM(COALESCE(sales,0)) AS net_sales FROM sales GROUP BY 1 ORDER BY 1",
	)
	fig = px.line(series, x="d", y="net_sales", title="Net Sales Over Time (Daily)")
	st.plotly_chart(fig, use_container_width=True)


def additional_bi_visuals(df: pd.DataFrame) -> None:
	left, right = st.columns(2)
	with left:
		df_hist = df[df["sales"].notna()].copy()
		fig = px.histogram(df_hist, x="sales", nbins=60, title="Sales Value Distribution")
		st.plotly_chart(fig, use_container_width=True)
	with right:
		box = df[["ordertype", "sales"]].dropna()
		fig2 = px.box(box, x="ordertype", y="sales", title="Sales by Order Type (Box Plot)")
		st.plotly_chart(fig2, use_container_width=True)
	
	heat = run_sql_over_dataframe(
		df,
		"SELECT region, ordertype, SUM(COALESCE(sales,0)) AS net_sales FROM sales GROUP BY 1,2",
	)
	fig3 = px.density_heatmap(heat, x="ordertype", y="region", z="net_sales", title="Heatmap: Region x Ordertype (Net Sales)")
	st.plotly_chart(fig3, use_container_width=True)


def top_entities(df: pd.DataFrame) -> None:
	left, right = st.columns(2)
	with left:
		by_site = run_sql_over_dataframe(
			df,
			"SELECT site, SUM(COALESCE(sales,0)) AS net_sales FROM sales GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
		)
		st.subheader("Top 10 Sites")
		st.dataframe(by_site)
	with right:
		by_region = run_sql_over_dataframe(
			df,
			"SELECT region, SUM(COALESCE(sales,0)) AS net_sales FROM sales GROUP BY 1 ORDER BY 2 DESC",
		)
		fig = px.bar(by_region, x="region", y="net_sales", title="Net Sales by Region")
		st.plotly_chart(fig, use_container_width=True)


def refunds_section(df: pd.DataFrame) -> None:
	refunds = run_sql_over_dataframe(
		df,
		"SELECT region, (COUNT(*) FILTER (WHERE sales<0))::DOUBLE/NULLIF(COUNT(*),0) AS refund_rate FROM sales GROUP BY 1 ORDER BY 2 DESC",
	)
	fig = px.bar(refunds, x="region", y="refund_rate", title="Refund Rate by Region")
	st.plotly_chart(fig, use_container_width=True)


def dq_section(df: pd.DataFrame) -> None:
	report = DataValidator.generate_data_quality_report(df)
	st.subheader("Data Quality")
	st.write({
		"rows": report.row_count,
		"null_sales": report.null_sales,
		"negative_sales": report.negative_sales,
		"future_dates": report.future_dates,
		"issues": report.issues,
	})


def filtered_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	st.sidebar.header("Filters")
	mind = pd.to_datetime(df["orderdate"]).min()
	maxd = pd.to_datetime(df["orderdate"]).max()
	date_range = st.sidebar.date_input("Date range", value=(mind.date(), maxd.date()))
	region = st.sidebar.multiselect("Region", sorted(df["region"].dropna().unique().tolist()))
	ordertype = st.sidebar.multiselect("Order type", sorted(df["ordertype"].dropna().unique().tolist()))
	site_type = st.sidebar.multiselect("Site type", sorted(df["site type"].dropna().unique().tolist()))

	df2 = df
	if date_range and len(date_range) == 2:
		start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
		df2 = df2[(df2["orderdate"] >= start) & (df2["orderdate"] <= end)]
	if region:
		df2 = df2[df2["region"].isin(region)]
	if ordertype:
		df2 = df2[df2["ordertype"].isin(ordertype)]
	if site_type:
		df2 = df2[df2["site type"].isin(site_type)]
	return df2


def qa_section(df: pd.DataFrame) -> None:
	st.subheader("AI Q&A (SQL Templates)")
	queries = bi_catalog()
	# Simple retrieval by substring match in title/description
	qtext = st.text_input("Ask a question (e.g., 'top 10 sites', 'refund rate by region')")
	if qtext:
		candidates = [q for q in queries if qtext.lower() in (q.title + " " + q.description).lower()]
		options = candidates[:20] if candidates else queries[:20]
	else:
		options = queries[:20]
	labels = [f"{q.category} | {q.title}" for q in options]
	choice = st.selectbox("Pick a query to run", options=list(range(len(options))), format_func=lambda i: labels[i])
	if st.button("Run Query"):
		selected = options[choice]
		st.code(selected.sql)
		try:
			res = run_sql_over_dataframe(df, selected.sql)
			st.dataframe(res.head(100))
		except Exception as e:
			st.error(f"Query failed: {e}")


# ===================== Advanced Profiling & Fixes =====================

def compute_profiling(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, str]]:
	profile = per_column_stats(df)
	issues: Dict[str, int] = {}
	# Basic issues
	issues["negative_sales_rows"] = int((df["sales"] < 0).sum()) if "sales" in df.columns else 0
	issues["null_sales_rows"] = int(df["sales"].isna().sum()) if "sales" in df.columns else 0
	issues["future_date_rows"] = int((df["orderdate"] > pd.Timestamp.today()).sum()) if "orderdate" in df.columns else 0
	if "site" in df.columns:
		ws = df["site"].astype(str)
		issues["whitespace_site_rows"] = int(((ws.str.startswith(" ")) | (ws.str.endswith(" "))).sum())
	# Outliers via IQR
	advisories: Dict[str, str] = {}
	if "sales" in df.columns:
		q1 = df["sales"].quantile(0.25)
		q3 = df["sales"].quantile(0.75)
		iqr = q3 - q1
		upper = q3 + 1.5 * iqr
		outliers = int((df["sales"] > upper).sum())
		issues["high_sales_outliers"] = outliers
		advisories["high_sales_outliers"] = f"Detected {outliers} potential high outliers above {upper:.2f}. Consider capping or winsorizing."
	return profile, issues, advisories


def apply_fix_trim_standardize(df: pd.DataFrame) -> pd.DataFrame:
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


def apply_fix_drop_future_dates(df: pd.DataFrame) -> pd.DataFrame:
	if "orderdate" not in df.columns:
		return df
	return df[df["orderdate"] <= pd.Timestamp.today()].copy()


def apply_fix_null_sales(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
	df = df.copy()
	if strategy == "drop":
		df = df[~df["sales"].isna()].copy()
	elif strategy == "zero":
		df.loc[df["sales"].isna(), "sales"] = 0.0
	else:
		df["is_missing_sales"] = df["sales"].isna()
	return df


def apply_fix_cap_outliers(df: pd.DataFrame, cap_quantile: float = 0.99) -> pd.DataFrame:
	if "sales" not in df.columns:
		return df
	cap = df["sales"].quantile(cap_quantile)
	df = df.copy()
	df.loc[df["sales"] > cap, "sales"] = cap
	return df


def profiling_page(df: pd.DataFrame) -> None:
	st.title("Data Profile – Advanced Statistics & Fixes")

	# NEW: Source Data Profiling (Original)
	st.header("Source Data Profiling (Original)")
	with st.expander("Original Data (scroll to load more)", expanded=False):
		st.dataframe(st.session_state["original_df"], height=300)

	orig = st.session_state["original_df"]
	orig_summary = dataset_summary(orig)
	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Rows", f"{orig_summary['rows']:,}")
	c2.metric("Columns", f"{orig_summary['columns']:,}")
	c3.metric("Memory (MB)", f"{orig_summary['memory_mb']:,}")
	c4.metric("Date Range", f"{orig_summary['date_min']} → {orig_summary['date_max']}")

	dq_rows = {
		"null_sales": int(orig["sales"].isna().sum()) if "sales" in orig.columns else 0,
		"negative_sales": int((orig["sales"] < 0).sum()) if "sales" in orig.columns else 0,
		"duplicate_keys": int(orig.duplicated(subset=[c for c in ["orderdate", "ordernumber", "site"] if c in orig.columns]).sum()),
	}
	if "orderdate_raw" in orig.columns:
		dates_eval_orig = detect_invalid_mmdd(orig["orderdate_raw"])
		dq_rows["invalid_mmdd_dates"] = int((~dates_eval_orig["mmdd_valid"]).sum())
	st.subheader("Original Data Quality Snapshot")
	st.write(dq_rows)

	ot1, ot2, ot3 = st.tabs(["Overview", "Missing", "Correlations"])
	with ot1:
		st.subheader("Per-Column Stats (Original)")
		st.dataframe(per_column_stats(orig))
	with ot2:
		missing_heatmap(orig, key="missing_original")
	with ot3:
		correlation_heatmap(orig, key="corr_original")

	st.divider()
	st.header("Refined Dataset (After Applied Transformations)")

	summary = dataset_summary(df)
	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Rows", f"{summary['rows']:,}")
	c2.metric("Columns", f"{summary['columns']:,}")
	c3.metric("Memory (MB)", f"{summary['memory_mb']:,}")
	c4.metric("Date Range", f"{summary['date_min']} → {summary['date_max']}")

	tab_overview, tab_numeric, tab_cats, tab_missing, tab_dates, tab_corr, tab_time, tab_memory = st.tabs([
		"Overview", "Numeric", "Categoricals", "Missing", "Dates", "Correlations", "Time", "Memory",
	])

	with tab_overview:
		st.subheader("Per-Column Extended Statistics (Refined Dataset)")
		profile = per_column_stats(df)
		st.dataframe(profile)
		csv = profile.to_csv(index=False).encode("utf-8")
		st.download_button("Download Profile CSV", data=csv, file_name="data_profile.csv", mime="text/csv")

	with tab_numeric:
		num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
		if not num_cols:
			st.info("No numeric columns detected.")
		else:
			col = st.selectbox("Numeric column", num_cols, key="num_col_select")
			vals = pd.to_numeric(df[col], errors="coerce")
			fig = px.histogram(vals.dropna(), x=col, nbins=50, title=f"Distribution of {col}", marginal="box")
			st.plotly_chart(fig, use_container_width=True, key=f"hist_{col}")
			cap_q = st.slider("Queue: cap outliers at quantile", min_value=0.90, max_value=0.999, value=0.99, step=0.001, key="cap_q_numeric")
			st.caption("Pros: reduces skew from extreme values. Cons: may hide true spikes.")
			if st.button("Queue Outlier Capping for Selected Column"):
				queue_transform(
					f"cap_{col}",
					f"Cap outliers in {col} at Q{cap_q}",
					cap_outliers_column(col, cap_q),
				)
				st.success("Queued.")

	with tab_cats:
		cat_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
		if not cat_cols:
			st.info("No categorical columns detected.")
		else:
			col = st.selectbox("Categorical column", cat_cols, key="cat_col_select")
			st.dataframe(value_frequencies(df, col))
			freq = value_frequencies(df, col)
			st.plotly_chart(px.bar(freq, x="value", y="frequency", title=f"Top Values – {col}"), use_container_width=True, key=f"cat_freq_{col}")
			st.caption("Pros: standardization fixes grouping splits. Cons: loses original casing if needed for display.")
			if st.button("Queue Trim + Standardize All Categorical Text"):
				queue_transform("trim_std", "Trim and standardize categorical text", fix_trim_standardize)
				st.success("Queued.")

	with tab_missing:
		missing_heatmap(df, key="missing_refined")
		strategy = st.selectbox("Queue missing sales strategy", ["flag", "drop", "zero"], index=0, key="missing_strategy")
		st.caption("Pros: flag keeps history; drop removes noise; zero maintains volume but affects averages.")
		if st.button("Queue Missing Sales Strategy"):
			queue_transform(f"missing_{strategy}", f"Missing sales: {strategy}", fix_missing_sales(strategy))
			st.success("Queued.")

	with tab_dates:
		st.subheader("Invalid Date Detection & Correction (MM/DD/YYYY)")
		if "orderdate_raw" not in st.session_state["original_df"].columns:
			st.info("Raw date strings not available in original data; reload the app.")
		else:
			dates_eval_orig = detect_invalid_mmdd(st.session_state["original_df"]["orderdate_raw"])  # evaluate against original
			invalid_mask_orig = ~dates_eval_orig["mmdd_valid"]
			valid_count = int((dates_eval_orig["mmdd_valid"]).sum())
			invalid_count = int(invalid_mask_orig.sum())
			st.write({"valid_mmdd_rows": valid_count, "invalid_mmdd_rows": invalid_count})

			context_cols = [c for c in ["ordernumber", "site", "region", "ordertype", "sales"] if c in st.session_state["original_df"].columns]
			detail_orig = pd.concat([dates_eval_orig, st.session_state["original_df"][context_cols].reset_index(drop=True)], axis=1)
			# Show original invalids highlighted in red
			st.markdown("Original (invalid highlighted)")
			styled_orig = detail_orig.head(500).style.apply(lambda r: ["background-color: #ffcccc" if (not bool(r.loc["mmdd_valid"])) else "" for _ in r], axis=1)
			st.dataframe(styled_orig)

			b1, b2, b3 = st.columns(3)
			with b1:
				if st.button("Queue: Correct ALL invalid dates (swap MM/DD)"):
					queue_transform("dates_swap", "Correct invalid dates by swapping MM/DD", fix_dates_swap_invalid)
					st.success("Queued.")
			with b2:
				if st.button("Convert Dates to MM/DD/YYYY (Apply Now)"):
					# Apply immediately to refined dataset
					st.session_state["refined_df"] = fix_dates_swap_invalid(st.session_state["refined_df"])  
					st.success("Converted invalid dates in refined dataset.")
			with b3:
				if st.button("Re-evaluate Dates", key="dates_reval_now"):
					st.experimental_rerun()

			# Show refined data validity after potential conversion
			if "orderdate_raw" in st.session_state["refined_df"].columns:
				dates_eval_ref = detect_invalid_mmdd(st.session_state["refined_df"]["orderdate_raw"])
				invalid_mask_ref = ~dates_eval_ref["mmdd_valid"]
				st.markdown("Refined (corrected in green, still invalid in red)")
				# Determine corrected rows by index alignment
				idx = st.session_state["refined_df"].index
				was_invalid = pd.Series(False, index=idx)
				was_invalid.loc[dates_eval_orig.index] = invalid_mask_orig.values
				now_valid = pd.Series(False, index=idx)
				now_valid.loc[dates_eval_ref.index] = dates_eval_ref["mmdd_valid"].values
				status = np.where(was_invalid & now_valid, "corrected", np.where(~now_valid, "still_invalid", "ok"))
				ref_view = pd.concat([dates_eval_ref, st.session_state["refined_df"][context_cols].reset_index(drop=True)], axis=1)
				ref_view = ref_view.head(500)
				def _style_row(r):
					stat = status[r.name] if r.name in status.index else "ok"
					color = "#ccffcc" if stat == "corrected" else ("#ffcccc" if stat == "still_invalid" else "")
					return [f"background-color: {color}"] * len(r)
				st.dataframe(ref_view.style.apply(_style_row, axis=1))

	with tab_corr:
		correlation_heatmap(df, key="corr_refined")

	with tab_time:
		time_distributions(df, key_prefix="refined")

	with tab_memory:
		mem = memory_usage_by_column(df)
		st.dataframe(mem)
		st.plotly_chart(px.bar(mem, x="column", y="memory_bytes", title="Memory Usage by Column"), use_container_width=True, key="mem_refined")

	st.divider()
	st.subheader("Queued Transformations")
	if st.session_state["transform_queue"]:
		for tid, desc, _ in st.session_state["transform_queue"]:
			colA, colB = st.columns([4,1])
			with colA:
				st.write(f"- {desc}")
			with colB:
				if st.button("Remove", key=f"rm_{tid}"):
					clear_transform(tid)
	else:
		st.write("None queued.")

	apply1, apply2 = st.columns([1,4])
	with apply1:
		if st.button("Apply All (Create/Update Refined Dataset)"):
			apply_all_transforms()
			st.success("Refined dataset updated.")
	with apply2:
		st.caption("Refined dataset feeds Overview, Data Docs, and Executive Story.")

	st.divider()
	st.subheader("Refined Data Preview")
	st.dataframe(st.session_state["refined_df"].head(20))


# ===================== Data Docs Page =====================

def _schema_table(df: pd.DataFrame) -> pd.DataFrame:
	return pd.DataFrame({
		"column": df.columns,
		"dtype": [str(t) for t in df.dtypes.values],
		"null_count": [int(df[c].isna().sum()) for c in df.columns],
		"distinct_count": [int(df[c].nunique(dropna=True)) for c in df.columns],
	})


def _sample_anomalies(df: pd.DataFrame) -> Dict[str, int]:
	neg = int((df["sales"] < 0).sum()) if "sales" in df.columns else 0
	null_sales = int(df["sales"].isna().sum()) if "sales" in df.columns else 0
	future_dates = int((df["orderdate"] > pd.Timestamp.today()).sum()) if "orderdate" in df.columns else 0
	whitespace_sites = 0
	if "site" in df.columns:
		ws = df["site"].astype(str)
		whitespace_sites = int(((ws.str.startswith(" ")) | (ws.str.endswith(" "))).sum())
	return {
		"negative_sales_rows": neg,
		"null_sales_rows": null_sales,
		"future_date_rows": future_dates,
		"whitespace_site_rows": whitespace_sites,
	}


def _lineage_markdown() -> str:
	return (
		"- Source: Point-of-Sale extract (CSV) -> project root file `sample sales.csv`\n"
		"- Ingestion: Encoding detection (chardet), robust CSV read with NA handling\n"
		"- Cleaning: type coercion (dates, numerics), trimming text, standardization\n"
		"- Validation: sales range checks, future dates, business rule checks\n"
		"- Enrichment: transaction_type classification (sale/refund/missing)\n"
		"- Analytics: DuckDB SQL-on-pandas for KPIs, trends, and breakdowns\n"
	)


def _etl_steps_markdown() -> str:
	return (
		"1) Extract: read CSV with detected encoding, parse dates (day-first)\n"
		"2) Transform: coerce numerics, flag/refine missing, trim/standardize categoricals\n"
		"3) Load: keep in-memory pandas; optional SQLite/export not yet wired\n"
		"4) Validate: generate DQ report and surface key issues\n"
		"5) Enrich: add features (transaction_type), compute KPIs and aggregates\n"
	)


def data_docs_page(df: pd.DataFrame) -> None:
	st.title("Data Documentation & Lineage")
	st.markdown("Dataset: `sample sales.csv`")

	st.subheader("Schema & Profiling")
	st.dataframe(_schema_table(df))

	left, right = st.columns(2)
	with left:
		st.subheader("Key Metrics")
		st.write(dataset_summary(df))
	with right:
		st.subheader("Anomaly Snapshot")
		st.write(_sample_anomalies(df))

	with st.expander("Data Lineage", expanded=True):
		st.markdown(_lineage_markdown())

	with st.expander("ETL / Transformation Steps", expanded=True):
		st.markdown(_etl_steps_markdown())

	with st.expander("Business Readiness Notes"):
		st.markdown(
			"- Null sales retained (flagged) for transparency; can be dropped/zeroed per policy\n"
			"- Negative sales treated as refunds; net sales computed as sum(sales)\n"
			"- Categoricals standardized to lowercase to avoid grouping splits\n"
			"- Date parsing assumes day-first format `%d/%m/%Y`\n"
		)


# ===================== Executive Story =====================

def _issue_subsets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
	issues: Dict[str, pd.DataFrame] = {}
	if "sales" in df.columns:
		issues["null_sales_rows"] = df[df["sales"].isna()].copy()
		issues["negative_sales_rows"] = df[df["sales"] < 0].copy()
	if "site" in df.columns:
		ws = df["site"].astype(str)
		issues["whitespace_site_rows"] = df[(ws.str.startswith(" ")) | (ws.str.endswith(" "))].copy()
	if "orderdate" in df.columns:
		issues["future_date_rows"] = df[df["orderdate"] > pd.Timestamp.today()].copy()
	# Duplicate detector on common business key
	key_cols = [c for c in ["orderdate", "ordernumber", "site"] if c in df.columns]
	if key_cols:
		dupe_mask = df.duplicated(subset=key_cols, keep=False)
		issues["duplicate_key_rows"] = df[dupe_mask].copy()
	return issues


def _columns_with_missing(df: pd.DataFrame) -> pd.DataFrame:
	counts = df.isna().sum()
	pct = (counts / len(df) * 100.0).round(2)
	res = pd.DataFrame({"column": counts.index, "null_count": counts.values, "null_pct": pct.values})
	res = res[res["null_count"] > 0].sort_values("null_count", ascending=False)
	return res


def executive_story_page(df: pd.DataFrame) -> None:
	st.title("Executive Story – From Raw Data to Decision-Ready Insights")

	# Core KPIs for leadership
	con = duckdb.connect()
	con.register("sales", df)
	kpis = con.execute(
		"""
		SELECT
			SUM(CASE WHEN sales > 0 THEN sales ELSE 0 END) AS total_sales,
			SUM(CASE WHEN sales < 0 THEN -sales ELSE 0 END) AS refunds_abs,
			SUM(COALESCE(sales,0)) AS net_sales,
			AVG(NULLIF(sales, 0)) AS avg_txn,
			COUNT(*) AS txns
		FROM sales
		"""
	).df().iloc[0]
	by_region = con.execute(
		"SELECT region, SUM(COALESCE(sales,0)) AS net_sales FROM sales GROUP BY 1 ORDER BY 2 DESC"
	).df()
	trend = con.execute(
		"SELECT DATE_TRUNC('week', orderdate) AS wk, SUM(COALESCE(sales,0)) AS net FROM sales GROUP BY 1 ORDER BY 1"
	).df()
	con.close()

	c1, c2, c3, c4, c5 = st.columns(5)
	c1.metric("Total Sales", f"{kpis.total_sales:,.0f}")
	c2.metric("Refunds (Abs)", f"{kpis.refunds_abs:,.0f}")
	c3.metric("Net Sales", f"{kpis.net_sales:,.0f}")
	c4.metric("Avg Txn", f"{kpis.avg_txn:,.2f}")
	c5.metric("Transactions", f"{int(kpis.txns):,}")

	# Narrative
	st.markdown(
		"""
		**What we have**: A raw Point‑of‑Sale extract containing 100k+ transactions across sites, regions, and channels. 
		It includes positive sales (revenue) and negative values (refunds/voids), with some missing values and occasional formatting artifacts from spreadsheets.

		**Current quality**: We observe nulls in sales, negative transactions indicating refunds, and minor text inconsistencies (e.g., trailing spaces in site names). 
		Dates are day‑first and aligned to the 2025 calendar. This level of quality is common in manual exports and is correctable.

		**Business impact**: Without cleaning, groupings are fragmented ("leeds" vs "leeds "), averages are skewed by nulls/outliers, and refund share is invisible in headline revenue. 
		Our pipeline standardizes these issues to deliver trustworthy KPIs.
		"""
	)

	# Mini visuals
	colA, colB = st.columns(2)
	with colA:
		st.plotly_chart(px.line(trend, x="wk", y="net", title="Weekly Net Sales"), use_container_width=True)
	with colB:
		st.plotly_chart(px.bar(by_region, x="region", y="net_sales", title="Net Sales by Region"), use_container_width=True)

	# Explicit: which columns and which rows are problematic
	st.subheader("Where exactly are the issues?")
	cols_missing = _columns_with_missing(df)
	if not cols_missing.empty:
		st.markdown("**Columns with missing values** (count and % of total rows):")
		st.dataframe(cols_missing)
	else:
		st.write("No columns with missing values.")

	issues = _issue_subsets(df)
	# Helper to render a sample and download
	def _issue_block(title: str, key: str, subset: pd.DataFrame) -> None:
		st.markdown(f"**{title}** — rows: {len(subset):,}")
		if len(subset) == 0:
			st.write("None detected.")
			return
		st.dataframe(subset.head(15))
		csv = subset.to_csv(index=False).encode("utf-8")
		st.download_button(f"Download {title} (CSV)", data=csv, file_name=f"{key}.csv", mime="text/csv")

	left_i, right_i = st.columns(2)
	with left_i:
		_issue_block("Rows with NULL sales", "null_sales_rows", issues.get("null_sales_rows", pd.DataFrame()))
		_issue_block("Rows with negative sales (refunds)", "negative_sales_rows", issues.get("negative_sales_rows", pd.DataFrame()))
	with right_i:
		_issue_block("Rows with whitespace in site names", "whitespace_site_rows", issues.get("whitespace_site_rows", pd.DataFrame()))
		_issue_block("Rows dated in the future", "future_date_rows", issues.get("future_date_rows", pd.DataFrame()))

	# Duplicate keys if any
	st.markdown("**Potential duplicate business keys (orderdate, ordernumber, site)**")
	_issue_block("Duplicate key rows", "duplicate_key_rows", issues.get("duplicate_key_rows", pd.DataFrame()))

	# New: Mixed date format validation & correction
	st.subheader("Date Format Validation (Expected: MM/DD/YYYY)")
	if "orderdate_raw" in df.columns:
		dates_eval = detect_invalid_mmdd(df["orderdate_raw"])
		invalid_mask = ~dates_eval["mmdd_valid"]
		invalid_count = int(invalid_mask.sum())
		st.write({"invalid_mmdd_rows": invalid_count, "swap_candidates": int(dates_eval["swap_candidate"].sum()), "bad_pattern": int((dates_eval["reason"]=="bad pattern").sum())})
		# Show sample with reasons, highlight in red when invalid
		sample = dates_eval.join(df[["ordernumber", "site", "region"]], how="left") if "ordernumber" in df.columns else dates_eval
		styled = sample.head(30).style.apply(lambda r: ["background-color: #ffcccc" if (not bool(r.loc["mmdd_valid"])) else "" for _ in r], axis=1)
		st.dataframe(styled)
		col_fix1, col_fix2 = st.columns(2)
		with col_fix1:
			if st.button("Swap month/day for invalid rows"):
				new_raw = df["orderdate_raw"].copy()
				new_raw.loc[invalid_mask] = swap_month_day(new_raw.loc[invalid_mask])
				# Parse to datetime with US format month/day
				parsed = pd.to_datetime(new_raw, errors="coerce", format="%m/%d/%Y")
				st.session_state["working_df"]["orderdate_raw"] = new_raw
				st.session_state["working_df"]["orderdate"] = parsed
				st.success("Swapped month/day for invalid rows and re-parsed orderdate as MM/DD/YYYY.")
		with col_fix2:
			if st.button("Re-evaluate Dates"):
				st.experimental_rerun()
	else:
		st.info("Raw date strings not available; reload the app to attach 'orderdate_raw'.")

	st.markdown(
		"""
		**Remediation and enrichment (our ETL/ELT path)**:
		1. Ingest with encoding detection; parse dates as day‑first
		2. Coerce numeric types; standardize text (trim + lowercase)
		3. Validate business rules (no future dates; sales sanity checks)
		4. Enrich with `transaction_type` (sale/refund/missing)
		5. Make refund rate, net vs gross, and ATV standard KPIs
		6. Persist cleaned dataset for analytics and ML
		"""
	)

	st.info(
		"Use the Profiling & Fixes page to apply these corrections interactively; this story and KPIs will update accordingly."
	)


# ===================== Main App =====================

def main() -> None:
	st.set_page_config(page_title="HospiAnalytics Pro", layout="wide")

	if not DATA_FILE.exists():
		st.warning(f"Data file not found: {DATA_FILE}")
		st.stop()

	df = load_data()

	# Sidebar navigation
	st.sidebar.title("Navigation")
	page = st.sidebar.radio("Page", ["Overview", "Data Docs", "Profiling & Fixes", "Executive Story"], index=0)
	st.title("HospiAnalytics Pro – Sales Performance Dashboard")

	if page == "Data Docs":
		data_docs_page(df)
		return
	elif page == "Profiling & Fixes":
		profiling_page(df)
		return
	elif page == "Executive Story":
		executive_story_page(df)
		return

	# Overview page (enhanced visuals)
	df_f = filtered_dataframe(df)
	kpi_cards(df_f)
	time_series(df_f)
	top_entities(df_f)
	refunds_section(df_f)
	dq_section(df_f)
	additional_bi_visuals(df_f)
	qa_section(df_f)


if __name__ == "__main__":
	main()
