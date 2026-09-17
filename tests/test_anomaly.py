"""Anomaly detector tests.

Expected counts on the real dataset were measured by independent pandas
calculation during data profiling. Expected counts on the tiny fixture were
derived by hand from its twelve rows. Neither was copied from detector output.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.anomaly.detectors import (
    detect_aging_unresolved,
    detect_data_integrity,
    detect_low_ratings,
    detect_resolution_outliers,
    detect_response_sla_breach,
    run_detectors,
)
from app.config import Settings
from app.domain import schema as S
from app.query.plan import AnomalyType, TimePreset, TimeWindow
from app.query.timewindow import resolve_window


@pytest.fixture
def cfg() -> Settings:
    return Settings(llm_provider="stub")


class TestTinyFixture:
    """Twelve rows with hand-checked properties."""

    def test_resolution_outlier_finds_the_planted_90_hour_ticket(
        self, tiny_frame, cfg
    ):
        # High-priority resolved values are 2.0-3.5 except one at 90.0.
        # Q1=2.2, Q3=3.0, fence=4.2.
        g = detect_resolution_outliers(tiny_frame, cfg)
        assert g.count == 1
        assert g.tickets[0][S.TICKET_ID] == "TKT-009"
        assert g.tickets[0]["threshold_hrs"] == pytest.approx(4.2, abs=0.01)

    def test_small_groups_are_skipped_not_guessed(self, tiny_frame, cfg):
        """Critical has one resolved ticket; an IQR on n=1 is meaningless."""
        g = detect_resolution_outliers(tiny_frame, cfg)
        assert g.stats["groups"]["Critical"]["skipped"] == "group too small"

    def test_aging_unresolved(self, tiny_frame, cfg, tiny_reference):
        g = detect_aging_unresolved(tiny_frame, cfg, tiny_reference)
        assert g.count == 1
        assert g.tickets[0][S.TICKET_ID] == "TKT-011"
        assert g.tickets[0]["age_hours"] == pytest.approx(122.0)

    def test_aging_ignores_low_priority(self, tiny_frame, cfg, tiny_reference):
        """TKT-012 is old and Escalated but Low, so it is out of scope."""
        g = detect_aging_unresolved(tiny_frame, cfg, tiny_reference)
        assert "TKT-012" not in {t[S.TICKET_ID] for t in g.tickets}

    def test_response_sla_breach(self, tiny_frame, cfg):
        # Critical target is 2.0h: TKT-010 at 4.0 and TKT-011 at 3.0 breach.
        g = detect_response_sla_breach(tiny_frame, cfg)
        assert g.count == 2
        assert {t[S.TICKET_ID] for t in g.tickets} == {"TKT-010", "TKT-011"}

    def test_low_ratings(self, tiny_frame, cfg):
        g = detect_low_ratings(tiny_frame, cfg)
        assert g.count == 2
        assert {t[S.TICKET_ID] for t in g.tickets} == {"TKT-009", "TKT-010"}

    def test_data_integrity_catches_resolution_before_response(self, tiny_frame, cfg):
        g = detect_data_integrity(tiny_frame, cfg)
        assert g.count == 1
        assert g.tickets[0][S.TICKET_ID] == "TKT-010"
        assert g.stats["by_issue"] == {"resolved before first response": 1}

    def test_run_all_totals(self, tiny_frame, cfg, tiny_reference):
        out = run_detectors(tiny_frame, cfg, tiny_reference)
        assert out["total_flags"] == 7
        assert len(out["detectors"]) == 5

    def test_selecting_one_detector_runs_only_that_one(
        self, tiny_frame, cfg, tiny_reference
    ):
        out = run_detectors(
            tiny_frame, cfg, tiny_reference, [AnomalyType.LOW_RATING]
        )
        assert len(out["detectors"]) == 1
        assert out["detectors"][0]["type"] == "low_rating"


class TestRealDataset:
    def test_measured_counts(self, real_store, cfg):
        frame, ref = real_store.frame, real_store.reference_time
        assert detect_resolution_outliers(frame, cfg).count == 18
        assert detect_aging_unresolved(frame, cfg, ref).count == 80
        assert detect_response_sla_breach(frame, cfg).count == 61
        assert detect_low_ratings(frame, cfg).count == 47
        assert detect_data_integrity(frame, cfg).count == 28

    def test_per_priority_grouping_beats_global(self, real_store, cfg):
        """Grouping by priority is a deliberate choice; prove it differs."""
        frame = real_store.frame
        grouped = detect_resolution_outliers(frame, cfg).count
        flat = detect_resolution_outliers(
            frame, Settings(llm_provider="stub", outlier_group_by="none")
        ).count
        assert grouped == 18
        assert flat == 21
        assert grouped != flat

    def test_sla_breaches_are_all_high_urgency(self, real_store, cfg):
        g = detect_response_sla_breach(real_store.frame, cfg)
        assert set(g.stats["by_priority"]) == {"Critical", "High"}

    def test_aging_detector_warns_that_it_is_noisy(self, real_store, cfg):
        g = detect_aging_unresolved(real_store.frame, cfg, real_store.reference_time)
        assert "dataset spans three months" in g.note

    def test_ticket_list_is_capped_but_count_is_exact(self, real_store):
        capped = Settings(llm_provider="stub", max_anomalies_per_detector=5)
        g = detect_aging_unresolved(
            real_store.frame, capped, real_store.reference_time
        )
        assert g.count == 80
        assert len(g.tickets) == 5
        assert g.truncated is True

    def test_window_scoping_narrows_the_population(self, real_store, cfg):
        frame, ref = real_store.frame, real_store.reference_time
        window = resolve_window(TimeWindow(preset=TimePreset.THIS_WEEK), ref)
        scoped = frame[
            (frame[S.CREATED_AT] >= window.start) & (frame[S.CREATED_AT] <= window.end)
        ]
        out = run_detectors(scoped, cfg, ref, [AnomalyType.ALL], window)
        assert out["tickets_in_scope"] < 500
        assert out["window"]["label"].startswith("this week")


class TestThresholdConfiguration:
    def test_raising_aging_hours_reduces_flags(self, real_store):
        frame, ref = real_store.frame, real_store.reference_time
        loose = detect_aging_unresolved(
            frame, Settings(llm_provider="stub", aging_hours=336), ref
        )
        assert loose.count == 63  # measured: 63 of 80 older than 14 days

    def test_mad_method_is_available(self, real_store):
        g = detect_resolution_outliers(
            real_store.frame, Settings(llm_provider="stub", outlier_method="mad")
        )
        assert g.stats["groups"]["High"]["method"] == "mad"
        assert g.count > 0

    def test_low_rating_threshold_respected(self, real_store):
        g = detect_low_ratings(
            real_store.frame, Settings(llm_provider="stub", low_rating_threshold=1)
        )
        assert g.count == 14  # measured: 14 tickets rated exactly 1


class TestEdgeCases:
    def test_empty_frame_does_not_crash(self, tiny_frame, cfg, tiny_reference):
        empty = tiny_frame.iloc[0:0]
        out = run_detectors(empty, cfg, tiny_reference)
        assert out["total_flags"] == 0
        assert all(d["count"] == 0 for d in out["detectors"])

    def test_no_resolved_tickets(self, tiny_frame, cfg):
        unresolved_only = tiny_frame[tiny_frame["is_unresolved"]]
        g = detect_resolution_outliers(unresolved_only, cfg)
        assert g.count == 0
        assert "no resolved tickets" in g.rule

    def test_identical_values_do_not_produce_false_outliers(self, cfg):
        """Zero variance must not flag everything as anomalous."""
        rows = [
            {
                S.TICKET_ID: f"TKT-{i:03d}",
                S.CREATED_AT: pd.Timestamp("2024-03-01 09:00"),
                S.CATEGORY: "Billing",
                S.PRIORITY: "High",
                S.STATUS: "Resolved",
                S.RESPONSE_TIME: 1.0,
                S.RESOLUTION_TIME: 5.0,
                S.AGENT_ID: "AGT-01",
                S.CUSTOMER_RATING: 4,
                S.ISSUE_SUMMARY: "same",
                "is_resolved": True,
                "is_unresolved": False,
            }
            for i in range(12)
        ]
        frame = pd.DataFrame(rows)
        assert detect_resolution_outliers(frame, cfg).count == 0
        mad_cfg = Settings(llm_provider="stub", outlier_method="mad")
        assert detect_resolution_outliers(frame, mad_cfg).count == 0

    def test_detector_output_is_json_serialisable(
        self, real_store, cfg
    ):
        import json

        out = run_detectors(real_store.frame, cfg, real_store.reference_time)
        assert json.loads(json.dumps(out))  # raises if any numpy type leaks through
