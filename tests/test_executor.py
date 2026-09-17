"""Executor tests.

Assertions state exact numbers. Values against the real dataset were derived by
independent pandas calculation during data profiling, not by copying executor
output, so these tests can actually catch a regression.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.query.executor import execute_plan
from app.query.plan import QueryPlan, TimePreset, TimeWindow
from app.query.timewindow import resolve_window


def run(plan_dict, frame, reference, max_rows=200):
    return execute_plan(QueryPlan.model_validate(plan_dict), frame, reference, max_rows)


class TestAgainstTinyFixture:
    def test_count_all(self, tiny_frame, tiny_reference):
        r = run({"intent": "aggregate", "aggregation": {"op": "count"}}, tiny_frame, tiny_reference)
        assert r.value == 12

    def test_count_with_filter(self, tiny_frame, tiny_reference):
        r = run(
            {
                "intent": "aggregate",
                "aggregation": {"op": "count"},
                "filters": [{"field": "status", "op": "eq", "value": "Open"}],
            },
            tiny_frame,
            tiny_reference,
        )
        assert r.value == 1

    def test_in_operator(self, tiny_frame, tiny_reference):
        r = run(
            {
                "intent": "aggregate",
                "aggregation": {"op": "count"},
                "filters": [{"field": "status", "op": "in", "values": ["Open", "Escalated"]}],
            },
            tiny_frame,
            tiny_reference,
        )
        assert r.value == 2

    def test_average_excludes_nulls_and_says_so(self, tiny_frame, tiny_reference):
        r = run(
            {"intent": "aggregate", "aggregation": {"op": "avg", "field": "customer_rating"}},
            tiny_frame,
            tiny_reference,
        )
        # 10 rated tickets: 5+4+4+3+5+4+5+4+1+2 = 37 -> 3.7
        assert r.value == pytest.approx(3.7)
        assert any("have no customer_rating" in n for n in r.notes)

    def test_max_resolution_time(self, tiny_frame, tiny_reference):
        r = run(
            {"intent": "aggregate", "aggregation": {"op": "max", "field": "resolution_time_hrs"}},
            tiny_frame,
            tiny_reference,
        )
        assert r.value == pytest.approx(90.0)

    def test_group_by_agent_counts(self, tiny_frame, tiny_reference):
        r = run(
            {
                "intent": "aggregate",
                "aggregation": {"op": "count"},
                "group_by": ["agent_id"],
                "sort": {"by": "metric", "direction": "desc"},
            },
            tiny_frame,
            tiny_reference,
        )
        assert {row["agent_id"]: row["metric"] for row in r.rows} == {
            "AGT-01": 4,
            "AGT-02": 4,
            "AGT-03": 4,
        }

    def test_ties_are_reported(self, tiny_frame, tiny_reference):
        r = run(
            {
                "intent": "aggregate",
                "aggregation": {"op": "count"},
                "group_by": ["agent_id"],
                "sort": {"by": "metric", "direction": "desc"},
            },
            tiny_frame,
            tiny_reference,
        )
        assert len(r.ties) == 3
        assert any("tie" in n for n in r.notes)

    def test_filters_any_is_ored(self, tiny_frame, tiny_reference):
        r = run(
            {
                "intent": "list",
                "filters": [{"field": "priority", "op": "eq", "value": "Critical"}],
                "filters_any": [
                    {"field": "resolution_time_hrs", "op": "gt", "value": 12},
                    {"field": "status", "op": "ne", "value": "Resolved"},
                ],
            },
            tiny_frame,
            tiny_reference,
        )
        # TKT-011 is Critical and Open. TKT-010 is Critical but resolved in 1.0h.
        assert {row["ticket_id"] for row in r.rows} == {"TKT-011"}

    def test_ne_includes_nulls(self, tiny_frame, tiny_reference):
        """A null resolution time is 'not equal to 2.0' and must be counted."""
        r = run(
            {
                "intent": "aggregate",
                "aggregation": {"op": "count"},
                "filters": [{"field": "resolution_time_hrs", "op": "ne", "value": 2.0}],
            },
            tiny_frame,
            tiny_reference,
        )
        assert r.value == 10  # 12 total minus two rows at exactly 2.0

    def test_contains_on_summary(self, tiny_frame, tiny_reference):
        r = run(
            {
                "intent": "list",
                "filters": [{"field": "issue_summary", "op": "contains", "value": "refund"}],
            },
            tiny_frame,
            tiny_reference,
        )
        assert r.matched_count == 1

    def test_empty_result_is_not_an_error(self, tiny_frame, tiny_reference):
        r = run(
            {
                "intent": "aggregate",
                "aggregation": {"op": "avg", "field": "customer_rating"},
                "filters": [{"field": "agent_id", "op": "eq", "value": "AGT-99"}],
            },
            tiny_frame,
            tiny_reference,
        )
        assert r.value is None
        assert r.notes

    def test_list_respects_limit_and_flags_truncation(self, tiny_frame, tiny_reference):
        r = run({"intent": "list", "limit": 3}, tiny_frame, tiny_reference)
        assert len(r.rows) == 3
        assert r.matched_count == 12
        assert r.truncated is True

    def test_unsupported_intent_returns_reason(self, tiny_frame, tiny_reference):
        r = run(
            {"intent": "unsupported", "unsupported_reason": "no customer names in data"},
            tiny_frame,
            tiny_reference,
        )
        assert r.notes == ["no customer names in data"]


class TestAgainstRealDataset:
    """Expected values measured independently from the CSV."""

    def test_open_tickets(self, real_store):
        r = run(
            {
                "intent": "aggregate",
                "aggregation": {"op": "count"},
                "filters": [{"field": "status", "op": "eq", "value": "Open"}],
            },
            real_store.frame,
            real_store.reference_time,
        )
        assert r.value == 111

    def test_average_technical_rating(self, real_store):
        r = run(
            {
                "intent": "aggregate",
                "aggregation": {"op": "avg", "field": "customer_rating"},
                "filters": [{"field": "category", "op": "eq", "value": "Technical"}],
            },
            real_store.frame,
            real_store.reference_time,
        )
        assert r.value == pytest.approx(3.7404, abs=1e-3)

    def test_critical_breaching_12_hours(self, real_store):
        r = run(
            {
                "intent": "list",
                "filters": [{"field": "priority", "op": "eq", "value": "Critical"}],
                "filters_any": [
                    {"field": "resolution_time_hrs", "op": "gt", "value": 12},
                    {"field": "status", "op": "ne", "value": "Resolved"},
                ],
                "limit": 200,
            },
            real_store.frame,
            real_store.reference_time,
        )
        assert r.matched_count == 34

    def test_top_agent_this_month(self, real_store):
        r = run(
            {
                "intent": "aggregate",
                "aggregation": {"op": "count"},
                "filters": [{"field": "status", "op": "eq", "value": "Resolved"}],
                "group_by": ["agent_id"],
                "time_window": {"preset": "this_month"},
                "sort": {"by": "metric", "direction": "desc"},
                "limit": 1,
            },
            real_store.frame,
            real_store.reference_time,
        )
        assert r.rows[0]["agent_id"] == "AGT-01"
        assert r.rows[0]["metric"] == 16

    def test_unresolved_count(self, real_store):
        r = run(
            {
                "intent": "aggregate",
                "aggregation": {"op": "count"},
                "filters": [{"field": "status", "op": "in", "values": ["Open", "Escalated"]}],
            },
            real_store.frame,
            real_store.reference_time,
        )
        assert r.value == 173


class TestTimeWindows:
    REF = datetime(2024, 3, 30, 18, 6)

    def test_all_time_is_open(self):
        w = resolve_window(TimeWindow(preset=TimePreset.ALL_TIME), self.REF)
        assert w.is_open

    def test_this_month_starts_at_the_first(self):
        w = resolve_window(TimeWindow(preset=TimePreset.THIS_MONTH), self.REF)
        assert w.start == datetime(2024, 3, 1, 0, 0)
        assert w.end.day == 30

    def test_last_month_is_february(self):
        w = resolve_window(TimeWindow(preset=TimePreset.LAST_MONTH), self.REF)
        assert w.start == datetime(2024, 2, 1, 0, 0)
        assert w.end.month == 2 and w.end.day == 29  # 2024 is a leap year

    def test_this_week_starts_monday(self):
        w = resolve_window(TimeWindow(preset=TimePreset.THIS_WEEK), self.REF)
        assert w.start.weekday() == 0
        assert w.start == datetime(2024, 3, 25, 0, 0)

    def test_last_7_days_is_inclusive(self):
        w = resolve_window(TimeWindow(preset=TimePreset.LAST_7_DAYS), self.REF)
        assert w.start == datetime(2024, 3, 24, 0, 0)

    def test_window_relative_to_dataset_returns_rows(self, real_store):
        """The point of the dataset-anchored clock: this must not be zero."""
        r = run(
            {
                "intent": "aggregate",
                "aggregation": {"op": "count"},
                "time_window": {"preset": "this_month"},
            },
            real_store.frame,
            real_store.reference_time,
        )
        assert r.value == 188
