"""Deterministic execution of a validated QueryPlan.

Every number a user sees is computed here, in plain pandas, from a plan that has
already passed validation. The LLM decides *what* to compute; this module decides
*how*, and it is fully unit-testable without a model.

No string is ever interpolated into an expression: filters are applied through
pandas comparisons on known columns, so there is no query language to inject into.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.domain import schema as S
from app.query.plan import (
    Aggregation,
    Filter,
    Intent,
    MetricOp,
    Operator,
    QueryPlan,
)
from app.query.timewindow import ResolvedWindow, resolve_window

logger = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    """The plan was valid but cannot be executed against this data."""


@dataclass
class QueryResult:
    """Outcome of running a plan. Carries provenance, not just the number."""

    intent: str
    value: float | int | None = None
    unit: str | None = None
    rows: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    matched_count: int = 0
    truncated: bool = False
    window: dict | None = None
    ties: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "value": self.value,
            "unit": self.unit,
            "rows": self.rows,
            "columns": self.columns,
            "row_count": self.row_count,
            "matched_count": self.matched_count,
            "truncated": self.truncated,
            "window": self.window,
            "ties": self.ties,
            "notes": self.notes,
        }


_METRIC_LABELS = {
    MetricOp.COUNT: "tickets",
    MetricOp.AVG: "average",
    MetricOp.SUM: "total",
    MetricOp.MIN: "minimum",
    MetricOp.MAX: "maximum",
    MetricOp.MEDIAN: "median",
}


def _clean(value: Any) -> Any:
    """Make a pandas scalar JSON-safe."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float):
        return round(value, 4)
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    """DataFrame to JSON-safe records."""
    out: list[dict] = []
    for row in frame.to_dict(orient="records"):
        out.append({k: _clean(v) for k, v in row.items()})
    return out


def _apply_filter(frame: pd.DataFrame, f: Filter) -> pd.DataFrame:
    col = f.field_name.value
    if col not in frame.columns:
        raise ExecutionError(f"Column '{col}' is not present in the dataset")

    series = frame[col]
    op = f.op

    if op == Operator.IS_NULL:
        return frame[series.isna()]
    if op == Operator.NOT_NULL:
        return frame[series.notna()]

    if op in (Operator.IN, Operator.NOT_IN):
        values = list(f.values or [])
        # Categorical columns compare cleanly against plain strings.
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype("string")
        mask = series.isin(values)
        return frame[~mask] if op == Operator.NOT_IN else frame[mask]

    if op == Operator.CONTAINS:
        return frame[series.astype("string").str.contains(str(f.value), case=False, na=False)]

    if col in S.NUMERIC_COLUMNS or col == S.CREATED_AT:
        target = pd.to_datetime(f.value) if col == S.CREATED_AT else f.value
    else:
        target = f.value
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype("string")

    comparisons = {
        Operator.EQ: lambda s, v: s == v,
        Operator.NE: lambda s, v: s != v,
        Operator.GT: lambda s, v: s > v,
        Operator.GTE: lambda s, v: s >= v,
        Operator.LT: lambda s, v: s < v,
        Operator.LTE: lambda s, v: s <= v,
    }
    fn = comparisons.get(op)
    if fn is None:
        raise ExecutionError(f"Unsupported operator '{op.value}'")

    try:
        mask = fn(series, target)
    except TypeError as exc:
        raise ExecutionError(
            f"Cannot compare column '{col}' with {f.value!r} using '{op.value}'"
        ) from exc

    # NaN comparisons yield False; ne must still include nulls as "not equal".
    if op == Operator.NE:
        mask = mask | series.isna()
    return frame[mask.fillna(False)]


def _apply_window(frame: pd.DataFrame, window: ResolvedWindow) -> pd.DataFrame:
    if window.is_open:
        return frame
    out = frame
    if window.start is not None:
        out = out[out[S.CREATED_AT] >= pd.Timestamp(window.start)]
    if window.end is not None:
        out = out[out[S.CREATED_AT] <= pd.Timestamp(window.end)]
    return out


def _compute_metric(frame: pd.DataFrame, agg: Aggregation) -> tuple[float | int | None, list[str]]:
    """Apply one aggregation. Returns the value plus any caveats."""
    notes: list[str] = []

    if agg.op == MetricOp.COUNT:
        return len(frame), notes

    col = agg.field_name.value
    series = pd.to_numeric(frame[col], errors="coerce")
    non_null = series.dropna()

    if non_null.empty:
        notes.append(f"No tickets with a {col} value matched, so the result is undefined.")
        return None, notes

    # Nulls here are meaningful: resolution_time and customer_rating are null for
    # unresolved tickets by design. Say so rather than silently averaging 327 of 500.
    missing = len(frame) - len(non_null)
    if missing > 0 and col in (S.RESOLUTION_TIME, S.CUSTOMER_RATING):
        notes.append(
            f"{missing} of {len(frame)} matching tickets have no {col} "
            "(unresolved tickets have neither); they are excluded."
        )

    ops = {
        MetricOp.AVG: non_null.mean,
        MetricOp.SUM: non_null.sum,
        MetricOp.MIN: non_null.min,
        MetricOp.MAX: non_null.max,
        MetricOp.MEDIAN: non_null.median,
    }
    return round(float(ops[agg.op]()), 4), notes


def _execute_grouped(
    frame: pd.DataFrame, plan: QueryPlan, limit: int
) -> tuple[list[dict], list[str], list[dict], list[str], bool]:
    """Aggregate per group, sorted. Returns rows, columns, ties, notes, truncated."""
    agg = plan.aggregation
    assert agg is not None  # guaranteed by QueryPlan validation
    group_cols = [g.value for g in plan.group_by]
    notes: list[str] = []

    if frame.empty:
        return [], group_cols + ["metric"], [], ["No tickets matched the filters."], False

    observed = {}
    for col in group_cols:
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            observed[col] = frame[col].astype("string")
    work = frame.assign(**observed) if observed else frame

    grouped = work.groupby(group_cols, dropna=False, observed=True)

    if agg.op == MetricOp.COUNT:
        metric = grouped.size()
    else:
        col = agg.field_name.value
        numeric = pd.to_numeric(work[col], errors="coerce")
        series = numeric.groupby([work[c] for c in group_cols], observed=True)
        fn = {
            MetricOp.AVG: "mean",
            MetricOp.SUM: "sum",
            MetricOp.MIN: "min",
            MetricOp.MAX: "max",
            MetricOp.MEDIAN: "median",
        }[agg.op]
        metric = getattr(series, fn)()
        counts = numeric.notna().groupby([work[c] for c in group_cols], observed=True).sum()
        # A group with two ratings should not outrank one with thirty.
        thin = counts[counts < 5]
        if not thin.empty:
            notes.append(
                f"{len(thin)} group(s) have fewer than 5 non-null {col} values; "
                "their averages are noisy."
            )
        metric = metric[counts > 0]

    result = metric.reset_index()
    result.columns = group_cols + ["metric"]
    result = result.dropna(subset=["metric"])

    ascending = plan.sort.direction.value == "asc" if plan.sort else False
    sort_col = "metric"
    if plan.sort and plan.sort.by != "metric" and plan.sort.by in result.columns:
        sort_col = plan.sort.by
    result = result.sort_values(sort_col, ascending=ascending, kind="mergesort")

    # Surface ties explicitly. On this dataset "which agent has the lowest average
    # rating" is a genuine two-way tie, and reporting one name would be wrong.
    ties: list[dict] = []
    if not result.empty and sort_col == "metric":
        best = result.iloc[0]["metric"]
        tied = result[result["metric"] == best]
        if len(tied) > 1:
            ties = _records(tied)
            notes.append(f"{len(tied)} groups tie at {round(float(best), 4)}.")

    total = len(result)
    truncated = total > limit
    rows = _records(result.head(limit))
    if truncated:
        notes.append(f"Showing {limit} of {total} groups.")
    return rows, list(result.columns), ties, notes, truncated


def execute_plan(plan: QueryPlan, frame: pd.DataFrame, reference, max_rows: int) -> QueryResult:
    """Run a validated plan against the dataset.

    Args:
        plan: A QueryPlan that has already passed validation.
        frame: The ticket DataFrame.
        reference: Reference datetime for relative windows.
        max_rows: Hard ceiling on returned rows.

    Returns:
        A QueryResult carrying the answer and its provenance.
    """
    if plan.intent == Intent.UNSUPPORTED:
        return QueryResult(intent=plan.intent.value, notes=[plan.unsupported_reason])

    window = resolve_window(plan.time_window, reference)
    working = _apply_window(frame, window)
    for f in plan.filters:
        working = _apply_filter(working, f)

    # filters_any are ORed together, then ANDed with the result above.
    if plan.filters_any:
        keep = pd.Index([])
        for f in plan.filters_any:
            keep = keep.union(_apply_filter(working, f).index)
        working = working.loc[working.index.intersection(keep)]

    matched = len(working)
    limit = min(plan.limit, max_rows)

    if plan.intent == Intent.AGGREGATE:
        if plan.group_by:
            rows, columns, ties, notes, truncated = _execute_grouped(working, plan, limit)
            return QueryResult(
                intent=plan.intent.value,
                rows=rows,
                columns=columns,
                row_count=len(rows),
                matched_count=matched,
                truncated=truncated,
                window=window.as_dict(),
                ties=ties,
                notes=notes,
            )

        value, notes = _compute_metric(working, plan.aggregation)
        unit = None
        if plan.aggregation.field_name:
            col = plan.aggregation.field_name.value
            unit = "hours" if col.endswith("_hrs") else "rating"
        elif plan.aggregation.op == MetricOp.COUNT:
            unit = "tickets"
        return QueryResult(
            intent=plan.intent.value,
            value=value,
            unit=unit,
            matched_count=matched,
            window=window.as_dict(),
            notes=notes,
        )

    # LIST
    display = [
        S.TICKET_ID,
        S.CREATED_AT,
        S.CATEGORY,
        S.PRIORITY,
        S.STATUS,
        S.RESPONSE_TIME,
        S.RESOLUTION_TIME,
        S.AGENT_ID,
        S.CUSTOMER_RATING,
        S.ISSUE_SUMMARY,
    ]
    out = working[display]
    if plan.sort and plan.sort.by in out.columns:
        out = out.sort_values(
            plan.sort.by, ascending=plan.sort.direction.value == "asc", kind="mergesort"
        )
    else:
        out = out.sort_values(S.CREATED_AT, ascending=False, kind="mergesort")

    truncated = matched > limit
    notes = [f"Showing {limit} of {matched} matching tickets."] if truncated else []
    return QueryResult(
        intent=plan.intent.value,
        rows=_records(out.head(limit)),
        columns=display,
        row_count=min(matched, limit),
        matched_count=matched,
        truncated=truncated,
        window=window.as_dict(),
        notes=notes,
    )
