from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def load_data() -> pd.DataFrame:
	# Keep a mutable working copy in session for UI-applied fixes
	if "working_df" not in st.session_state:
		st.session_state["working_df"] = _load_clean_data()
	return st.session_state["working_df"]


@st.cache_data(show_spinner=False)
def bi_catalog() -> List[BIQuery]:
	return generate_bi_questions()


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


def correlation_heatmap(df: pd.DataFrame) -> None:
	numeric_df = df.select_dtypes(include=["number"]).copy()
	if numeric_df.empty or numeric_df.shape[1] < 2:
		st.info("Not enough numeric columns for correlation heatmap.")
		return
	corr = numeric_df.corr(numeric_only=True)
	fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
	st.plotly_chart(fig, use_container_width=True)


def missing_heatmap(df: pd.DataFrame) -> None:
	miss = df.isna().astype(int)
	if miss.values.sum() == 0:
		st.info("No missing values detected.")
		return
	# Aggregate by column to keep chart readable
	col_missing = miss.sum(axis=0).reset_index()
	col_missing.columns = ["column", "missing"]
	fig = px.bar(col_missing, x="column", y="missing", title="Missing Values by Column")
	st.plotly_chart(fig, use_container_width=True)


def memory_usage_by_column(df: pd.DataFrame) -> pd.DataFrame:
	mem = df.memory_usage(deep=True)
	return pd.DataFrame({"column": mem.index.astype(str), "memory_bytes": mem.values})


def time_distributions(df: pd.DataFrame) -> None:
	if "orderdate" not in df.columns:
		st.info("No orderdate column for time distributions.")
		return
	dd = pd.to_datetime(df["orderdate"])  # safe
	by_day = df.copy()
	by_day["day"] = dd.dt.date
	daily = by_day.groupby("day").agg(net_sales=("sales", lambda x: float(np.nansum(x)))).reset_index()
	st.plotly_chart(px.line(daily, x="day", y="net_sales", title="Daily Net Sales"), use_container_width=True)

	# If time present, derive hour; else skip gracefully
	if hasattr(dd.dt, "hour"):
		by_hour = dd.dt.hour.dropna()
		if not by_hour.empty:
			hist = by_hour.value_counts().sort_index().reset_index()
			hist.columns = ["hour", "count"]
			st.plotly_chart(px.bar(hist, x="hour", y="count", title="Records by Hour"), use_container_width=True)


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

	# Summary KPI cards
	summary = dataset_summary(df)
	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Rows", f"{summary['rows']:,}")
	c2.metric("Columns", f"{summary['columns']:,}")
	c3.metric("Memory (MB)", f"{summary['memory_mb']:,}")
	c4.metric("Date Range", f"{summary['date_min']} → {summary['date_max']}")

	tab_overview, tab_numeric, tab_cats, tab_missing, tab_corr, tab_time, tab_memory = st.tabs([
		"Overview", "Numeric", "Categoricals", "Missing", "Correlations", "Time", "Memory",
	])

	with tab_overview:
		st.subheader("Per-Column Extended Statistics")
		profile = per_column_stats(df)
		st.dataframe(profile)
		csv = profile.to_csv(index=False).encode("utf-8")
		st.download_button("Download Profile CSV", data=csv, file_name="data_profile.csv", mime="text/csv")

	with tab_numeric:
		num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
		if not num_cols:
			st.info("No numeric columns detected.")
		else:
			col = st.selectbox("Numeric column", num_cols)
			vals = pd.to_numeric(df[col], errors="coerce")
			fig = px.histogram(vals.dropna(), x=col, nbins=50, title=f"Distribution of {col}", marginal="box")
			st.plotly_chart(fig, use_container_width=True)

	with tab_cats:
		cat_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
		if not cat_cols:
			st.info("No categorical columns detected.")
		else:
			col = st.selectbox("Categorical column", cat_cols)
			st.dataframe(value_frequencies(df, col))
			freq = value_frequencies(df, col)
			st.plotly_chart(px.bar(freq, x="value", y="frequency", title=f"Top Values – {col}"), use_container_width=True)

	with tab_missing:
		missing_heatmap(df)

	with tab_corr:
		correlation_heatmap(df)

	with tab_time:
		time_distributions(df)

	with tab_memory:
		mem = memory_usage_by_column(df)
		st.dataframe(mem)
		st.plotly_chart(px.bar(mem, x="column", y="memory_bytes", title="Memory Usage by Column"), use_container_width=True)

	st.divider()
	st.subheader("Recommended Actions")
	c1, c2, c3 = st.columns(3)
	with c1:
		if st.button("Trim + Standardize Text"):
			st.session_state["working_df"] = apply_fix_trim_standardize(st.session_state["working_df"])
			st.success("Applied trim and standardization.")
	with c2:
		strategy = st.selectbox("Missing sales strategy", ["flag", "drop", "zero"], index=0)
		if st.button("Apply Missing Sales Strategy"):
			st.session_state["working_df"] = apply_fix_null_sales(st.session_state["working_df"], strategy)
			st.success(f"Applied missing sales strategy: {strategy}")
	with c3:
		if st.button("Drop Future Dates"):
			st.session_state["working_df"] = apply_fix_drop_future_dates(st.session_state["working_df"])
			st.success("Dropped future-dated rows.")

	c4, c5, c6 = st.columns(3)
	with c4:
		cap_q = st.slider("Cap outliers at quantile", min_value=0.90, max_value=0.999, value=0.99, step=0.001)
		if st.button("Cap High Outliers"):
			st.session_state["working_df"] = apply_fix_cap_outliers(st.session_state["working_df"], cap_quantile=cap_q)
			st.success(f"Capped outliers at Q{cap_q}")
	with c5:
		if st.button("Recompute Profile"):
			st.experimental_rerun()
	with c6:
		csv2 = st.session_state["working_df"].to_csv(index=False).encode("utf-8")
		st.download_button("Download Cleaned Data", data=csv2, file_name="cleaned_data.csv", mime="text/csv")

	st.divider()
	st.subheader("Preview After Fixes")
	st.dataframe(st.session_state["working_df"].head(20))


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


# ===================== Main App =====================

def main() -> None:
	st.set_page_config(page_title="HospiAnalytics Pro", layout="wide")

	if not DATA_FILE.exists():
		st.warning(f"Data file not found: {DATA_FILE}")
		st.stop()

	df = load_data()

	# Sidebar navigation
	st.sidebar.title("Navigation")
	page = st.sidebar.radio("Page", ["Overview", "Data Docs", "Profiling & Fixes"], index=0)
	st.title("HospiAnalytics Pro – Sales Performance Dashboard")

	if page == "Data Docs":
		data_docs_page(df)
		return
	elif page == "Profiling & Fixes":
		profiling_page(df)
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
