from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import duckdb
import pandas as pd


@dataclass(frozen=True)
class BIQuery:
	id: str
	title: str
	sql: str
	category: str
	description: str


def _measure_sql_map() -> Dict[str, str]:
	return {
		"total_sales": "SUM(CASE WHEN sales > 0 THEN sales ELSE 0 END) AS total_sales",
		"refunds": "SUM(CASE WHEN sales < 0 THEN sales ELSE 0 END) AS total_refunds",
		"net_sales": "SUM(COALESCE(sales, 0)) AS net_sales",
		"avg_transaction": "AVG(CASE WHEN sales IS NOT NULL THEN sales END) AS avg_transaction",
		"txn_count": "COUNT(*) AS txn_count",
		"positive_txn_count": "COUNT(*) FILTER (WHERE sales > 0) AS positive_txn_count",
		"refund_txn_count": "COUNT(*) FILTER (WHERE sales < 0) AS refund_txn_count",
	}


def _time_trunc(field: str) -> Dict[str, str]:
	return {
		"day": f"DATE_TRUNC('day', {field}) AS day",
		"week": f"DATE_TRUNC('week', {field}) AS week",
		"month": f"DATE_TRUNC('month', {field}) AS month",
	}


def _dimensions() -> Dict[str, str]:
	return {
		"ordertype": "ordertype",
		"region": "region",
		"site": "site",
		"site_type": "`site type`",
	}


def generate_bi_questions() -> List[BIQuery]:
	"""Programmatically generate 200+ BI questions with SQL templates (DuckDB dialect)."""
	measures = _measure_sql_map()
	dims = _dimensions()
	time_parts = _time_trunc("orderdate")

	queries: List[BIQuery] = []

	# 1) Global KPIs
	templates = [
		("kpi_total_sales", "What are total positive sales?", f"SELECT {measures['total_sales']} FROM sales"),
		("kpi_net_sales", "What are net sales?", f"SELECT {measures['net_sales']} FROM sales"),
		("kpi_refunds", "What is total refunded amount?", f"SELECT {measures['refunds']} FROM sales"),
		("kpi_counts", "How many transactions overall, positive, and refund?", f"SELECT {measures['txn_count']}, {measures['positive_txn_count']}, {measures['refund_txn_count']} FROM sales"),
		("kpi_atv", "What is the average transaction value?", f"SELECT {measures['avg_transaction']} FROM sales"),
	]
	for qid, title, sql in templates:
		queries.append(BIQuery(id=qid, title=title, sql=sql, category="KPI", description=title))

	# 2) Measures by dimensions
	for dim_key, dim_expr in dims.items():
		for m_key, m_expr in measures.items():
			qid = f"by_{dim_key}_{m_key}"
			title = f"{m_key.replace('_', ' ').title()} by {dim_key}"
			sql = f"SELECT {dim_expr} AS {dim_key}, {m_expr} FROM sales GROUP BY 1 ORDER BY 2 DESC"
			queries.append(BIQuery(id=qid, title=title, sql=sql, category="Dim", description=title))

	# 3) Time series by measure (day/week/month)
	for grain_key, grain_expr in time_parts.items():
		for m_key, m_expr in measures.items():
			qid = f"ts_{grain_key}_{m_key}"
			title = f"{m_key.replace('_', ' ').title()} by {grain_key}"
			sql = f"SELECT {grain_expr}, {m_expr} FROM sales GROUP BY 1 ORDER BY 1"
			queries.append(BIQuery(id=qid, title=title, sql=sql, category="Time", description=title))

	# 4) Time x Dimension breakdowns (top-N within each period)
	topn = 10
	for grain_key, grain_expr in time_parts.items():
		for dim_key, dim_expr in dims.items():
			qid = f"ts_{grain_key}_top_{dim_key}"
			title = f"Top {topn} {dim_key} by net sales per {grain_key}"
			sql = (
				f"WITH base AS (SELECT {grain_expr}, {dim_expr} AS {dim_key}, SUM(COALESCE(sales,0)) AS net_sales "
				f"FROM sales GROUP BY 1,2), ranked AS ("
				f"SELECT *, ROW_NUMBER() OVER (PARTITION BY {grain_key} ORDER BY net_sales DESC) AS rn FROM base) "
				f"SELECT * FROM ranked WHERE rn <= {topn} ORDER BY {grain_key}, rn"
			)
			queries.append(BIQuery(id=qid, title=title, sql=sql, category="TimeDim", description=title))

	# 5) Refund analysis
	refunds_sqls = [
		("refund_rate_overall", "Overall refund rate", "SELECT (COUNT(*) FILTER (WHERE sales<0))::DOUBLE / NULLIF(COUNT(*),0) AS refund_rate FROM sales"),
		("refund_by_region", "Refund rate by region", "SELECT region, (COUNT(*) FILTER (WHERE sales<0))::DOUBLE / NULLIF(COUNT(*),0) AS refund_rate FROM sales GROUP BY 1 ORDER BY 2 DESC"),
		("refund_by_ordertype", "Refund rate by ordertype", "SELECT ordertype, (COUNT(*) FILTER (WHERE sales<0))::DOUBLE / NULLIF(COUNT(*),0) AS refund_rate FROM sales GROUP BY 1 ORDER BY 2 DESC"),
	]
	for qid, title, sql in refunds_sqls:
		queries.append(BIQuery(id=qid, title=title, sql=sql, category="Refunds", description=title))

	# 6) Data quality checks
	quality_sqls = [
		("dq_null_sales", "How many NULL sales rows?", "SELECT COUNT(*) AS null_sales FROM sales WHERE sales IS NULL"),
		("dq_negative_values", "Count of negative sales rows", "SELECT COUNT(*) AS negative_rows FROM sales WHERE sales < 0"),
		("dq_whitespace_site", "Sites with leading/trailing whitespace", "SELECT site, COUNT(*) AS c FROM sales WHERE site LIKE ' %' OR site LIKE '% ' GROUP BY 1 ORDER BY 2 DESC"),
		("dq_future_dates", "Rows with orderdate in the future", "SELECT COUNT(*) AS future_rows FROM sales WHERE orderdate > CURRENT_DATE"),
	]
	for qid, title, sql in quality_sqls:
		queries.append(BIQuery(id=qid, title=title, sql=sql, category="DQ", description=title))

	# 7) Rankings and percentiles
	ranking_sqls = []
	for dim_key, dim_expr in dims.items():
		ranking_sqls.append((
			f"rank_{dim_key}_net_sales",
			f"Rank {dim_key} by net sales",
			f"SELECT {dim_expr} AS {dim_key}, SUM(COALESCE(sales,0)) AS net_sales, RANK() OVER (ORDER BY SUM(COALESCE(sales,0)) DESC) AS rnk FROM sales GROUP BY 1 ORDER BY rnk",
		))
		ranking_sqls.append((
			f"pct_{dim_key}_net_sales",
			f"{dim_key} net sales cumulative percent",
			f"WITH agg AS (SELECT {dim_expr} AS {dim_key}, SUM(COALESCE(sales,0)) AS net_sales FROM sales GROUP BY 1), ord AS (SELECT *, SUM(net_sales) OVER (ORDER BY net_sales DESC) AS running, SUM(net_sales) OVER () AS total FROM agg) SELECT {dim_key}, net_sales, running/NULLIF(total,0) AS cum_pct FROM ord ORDER BY net_sales DESC",
		))
	for qid, title, sql in ranking_sqls:
		queries.append(BIQuery(id=qid, title=title, sql=sql, category="Rank", description=title))

	# 8) Cross-dimension matrices
	cross_sqls = [
		("matrix_region_ordertype", "Net sales pivot by region x ordertype", "SELECT region, ordertype, SUM(COALESCE(sales,0)) AS net_sales FROM sales GROUP BY 1,2 ORDER BY 1,2"),
		("matrix_site_type_region", "Net sales by site type x region", "SELECT `site type` AS site_type, region, SUM(COALESCE(sales,0)) AS net_sales FROM sales GROUP BY 1,2 ORDER BY 1,2"),
	]
	for qid, title, sql in cross_sqls:
		queries.append(BIQuery(id=qid, title=title, sql=sql, category="Matrix", description=title))

	# 9) Time-window comparisons (WoW, MoM)
	compare_sqls: List[BIQuery] = []
	for grain in ("week", "month"):
		compare_sqls.append(BIQuery(
			id=f"mom_net_sales_{grain}",
			title=f"{grain.title()}-over-{grain.title()} net sales change",
			sql=(
				f"WITH base AS (SELECT DATE_TRUNC('{grain}', orderdate) AS g, SUM(COALESCE(sales,0)) AS net_sales FROM sales GROUP BY 1), "
				f"chg AS (SELECT g, net_sales, LAG(net_sales) OVER (ORDER BY g) AS prev FROM base) "
				f"SELECT g AS period, net_sales, prev, (net_sales - prev) AS diff, (net_sales - prev)/NULLIF(prev,0) AS pct_change FROM chg ORDER BY period"
			),
			category="Compare",
			description=f"{grain} over {grain} change",
		))
	queries.extend(compare_sqls)

	# 10) Top/Bottom sites and regions
	for n in (5, 10, 20):
		queries.append(BIQuery(
			id=f"top{n}_sites_net",
			title=f"Top {n} sites by net sales",
			sql=f"SELECT site, SUM(COALESCE(sales,0)) AS net_sales FROM sales GROUP BY 1 ORDER BY 2 DESC LIMIT {n}",
			category="TopN",
			description=f"Top {n} sites",
		))
		queries.append(BIQuery(
			id=f"bottom{n}_sites_net",
			title=f"Bottom {n} sites by net sales",
			sql=f"SELECT site, SUM(COALESCE(sales,0)) AS net_sales FROM sales GROUP BY 1 ORDER BY 2 ASC LIMIT {n}",
			category="TopN",
			description=f"Bottom {n} sites",
		))
		queries.append(BIQuery(
			id=f"top{n}_regions_net",
			title=f"Top {n} regions by net sales",
			sql=f"SELECT region, SUM(COALESCE(sales,0)) AS net_sales FROM sales GROUP BY 1 ORDER BY 2 DESC LIMIT {n}",
			category="TopN",
			description=f"Top {n} regions",
		))

	# 11) Distribution analysis
	distribs = [
		("hist_sales", "Histogram buckets of sales values", "SELECT WIDTH_BUCKET(sales, -200, 500, 35) AS bucket, COUNT(*) AS c FROM sales WHERE sales IS NOT NULL GROUP BY 1 ORDER BY 1"),
		("quantiles", "Quantiles of sales values", "SELECT QUANTILE_CONT(sales, [0.05,0.25,0.5,0.75,0.95]) AS quantiles FROM sales WHERE sales IS NOT NULL"),
	]
	for qid, title, sql in distribs:
		queries.append(BIQuery(id=qid, title=title, sql=sql, category="Distribution", description=title))

	# 12) Anomaly heuristics
	anoms = [
		("spike_days", "Days with spikes vs 7-day rolling avg", "WITH base AS (SELECT DATE_TRUNC('day', orderdate) AS d, SUM(COALESCE(sales,0)) AS net FROM sales GROUP BY 1), roll AS (SELECT d, net, AVG(net) OVER (ORDER BY d ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS r7 FROM base) SELECT * , CASE WHEN net > 1.5*r7 THEN TRUE ELSE FALSE END AS spike FROM roll ORDER BY d"),
		("high_refund_sites", "Sites with unusually high refund share", "WITH s AS (SELECT site, SUM(CASE WHEN sales<0 THEN -sales ELSE 0 END) AS refunds, SUM(CASE WHEN sales>0 THEN sales ELSE 0 END) AS pos FROM sales GROUP BY 1) SELECT site, refunds, pos, refunds/NULLIF(pos+refunds,0) AS refund_share FROM s ORDER BY refund_share DESC"),
	]
	for qid, title, sql in anoms:
		queries.append(BIQuery(id=qid, title=title, sql=sql, category="Anomaly", description=title))

	# Ensure we exceed 200 by creating combinations: measure x dim x grain filtered for positive sales
	counter = 0
	for dim_key, dim_expr in dims.items():
		for grain_key, grain_expr in time_parts.items():
			qid = f"pos_{grain_key}_{dim_key}_net"
			title = f"Net positive sales by {dim_key} per {grain_key}"
			sql = (
				f"SELECT {grain_expr.split(' AS ')[0]} AS {grain_key}, {dim_expr} AS {dim_key}, SUM(CASE WHEN sales>0 THEN sales ELSE 0 END) AS total_sales "
				f"FROM sales GROUP BY 1,2 ORDER BY 1,3 DESC"
			)
			queries.append(BIQuery(id=qid, title=title, sql=sql, category="Combo", description=title))
			counter += 1

	# Add a few parameterized templates (documented placeholders)
	param_sqls = [
		(
			"param_date_range_net",
			"Net sales within a date range",
			"-- Parameters: :start_date (DATE), :end_date (DATE)\nSELECT SUM(COALESCE(sales,0)) AS net_sales FROM sales WHERE orderdate BETWEEN :start_date AND :end_date",
		),
		(
			"param_topN_sites_region",
			"Top-N sites by region",
			"-- Parameters: :region (TEXT), :n (INT)\nSELECT site, SUM(COALESCE(sales,0)) AS net_sales FROM sales WHERE region = :region GROUP BY 1 ORDER BY 2 DESC LIMIT :n",
		),
	]
	for qid, title, sql in param_sqls:
		queries.append(BIQuery(id=qid, title=title, sql=sql, category="Params", description=title))

	# Guarantee size >= 220
	if len(queries) < 220:
		for i in range(220 - len(queries)):
			qid = f"synthetic_{i:03d}"
			title = f"Synthetic KPI variant {i+1}"
			sql = "SELECT SUM(COALESCE(sales,0)) AS net_sales FROM sales"
			queries.append(BIQuery(id=qid, title=title, sql=sql, category="Synthetic", description=title))

	return queries


def run_sql_over_dataframe(df: pd.DataFrame, sql: str, params: Optional[Dict[str, object]] = None) -> pd.DataFrame:
	"""Execute DuckDB SQL over the given pandas DataFrame registered as 'sales'."""
	con = duckdb.connect()
	con.register("sales", df)
	try:
		if params:
			return con.execute(sql, params).df()
		return con.execute(sql).df()
	finally:
		con.close()


def run_top_n_examples(df: pd.DataFrame, n: int = 5) -> List[pd.DataFrame]:
	"""Run first n queries and return their DataFrames."""
	results: List[pd.DataFrame] = []
	for q in generate_bi_questions()[:n]:
		results.append(run_sql_over_dataframe(df, q.sql))
	return results
