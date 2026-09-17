"""Ingestion tests, including the failure paths."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.loader import IngestError, load_tickets
from app.domain import schema as S


def _write(tmp_path, frame, name="t.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _valid_row(**over):
    row = {
        S.TICKET_ID: "TKT-001",
        S.CREATED_AT: "2024-03-01 09:00",
        S.CATEGORY: "Billing",
        S.PRIORITY: "High",
        S.STATUS: "Resolved",
        S.RESPONSE_TIME: 1.0,
        S.RESOLUTION_TIME: 2.0,
        S.AGENT_ID: "AGT-01",
        S.CUSTOMER_RATING: 4,
        S.ISSUE_SUMMARY: "test",
    }
    row.update(over)
    return row


class TestRealDataset:
    def test_loads_all_500_rows(self, real_store):
        assert real_store.profile.row_count == 500
        assert real_store.report.rows_rejected == 0

    def test_reference_clock_anchors_to_newest_ticket(self, real_store):
        assert real_store.reference_time.year == 2024
        assert real_store.reference_time == real_store.profile.latest

    def test_nulls_only_on_unresolved(self, real_store):
        df = real_store.frame
        unresolved = df[S.STATUS] != "Resolved"
        assert (df[S.RESOLUTION_TIME].isna() == unresolved).all()
        assert (df[S.CUSTOMER_RATING].isna() == unresolved).all()

    def test_integrity_violations_are_warned_not_dropped(self, real_store):
        assert any("resolve before first response" in w for w in real_store.report.warnings)
        assert real_store.profile.row_count == 500


class TestHeaderNormalisation:
    def test_brief_section_3_2_spelling_is_accepted(self, tmp_path):
        """The brief spells these columns differently in two places."""
        row = _valid_row()
        renamed = {
            "resp_time_hrs": row.pop(S.RESPONSE_TIME),
            "resol_time_hrs": row.pop(S.RESOLUTION_TIME),
            "cust_rating": row.pop(S.CUSTOMER_RATING),
        }
        row.update(renamed)
        frame, report = load_tickets(_write(tmp_path, pd.DataFrame([row])))
        assert S.RESPONSE_TIME in frame.columns
        assert S.RESOLUTION_TIME in frame.columns
        assert S.CUSTOMER_RATING in frame.columns
        assert report.renamed_columns

    def test_unknown_column_is_dropped_with_warning(self, tmp_path):
        row = _valid_row()
        row["internal_notes"] = "secret"
        frame, report = load_tickets(_write(tmp_path, pd.DataFrame([row])))
        assert "internal_notes" not in frame.columns
        assert any("internal_notes" in w for w in report.warnings)

    def test_missing_required_column_raises(self, tmp_path):
        row = _valid_row()
        del row[S.PRIORITY]
        with pytest.raises(IngestError, match="missing required columns"):
            load_tickets(_write(tmp_path, pd.DataFrame([row])))


class TestRowValidation:
    @pytest.mark.parametrize(
        "override,reason",
        [
            ({S.PRIORITY: "Urgent"}, "invalid priority"),
            ({S.STATUS: "Closed"}, "invalid status"),
            ({S.CATEGORY: "Sales"}, "invalid category"),
            ({S.CREATED_AT: "not-a-date"}, "unparseable created_at"),
            ({S.CUSTOMER_RATING: 9}, "customer_rating outside"),
            ({S.RESPONSE_TIME: -3}, "negative response_time_hrs"),
            ({S.TICKET_ID: ""}, "missing ticket_id"),
        ],
    )
    def test_bad_row_is_quarantined(self, tmp_path, override, reason):
        good, bad = _valid_row(), _valid_row(**{S.TICKET_ID: "TKT-002", **override})
        frame, report = load_tickets(_write(tmp_path, pd.DataFrame([good, bad])))
        assert len(frame) == 1
        assert report.rows_rejected == 1
        assert reason in report.rejected[0].reason

    def test_duplicate_ticket_id_keeps_first(self, tmp_path):
        rows = [_valid_row(), _valid_row(**{S.ISSUE_SUMMARY: "dupe"})]
        frame, report = load_tickets(_write(tmp_path, pd.DataFrame(rows)))
        assert len(frame) == 1
        assert frame.iloc[0][S.ISSUE_SUMMARY] == "test"
        assert "duplicate ticket_id" in report.rejected[0].reason

    def test_whitespace_is_stripped(self, tmp_path):
        row = _valid_row(**{S.PRIORITY: "  High  ", S.STATUS: " Resolved "})
        frame, report = load_tickets(_write(tmp_path, pd.DataFrame([row])))
        assert report.rows_rejected == 0
        assert frame.iloc[0][S.PRIORITY] == "High"

    def test_strict_mode_aborts_on_any_bad_row(self, tmp_path):
        rows = [_valid_row(), _valid_row(**{S.TICKET_ID: "TKT-002", S.PRIORITY: "Nope"})]
        with pytest.raises(IngestError, match="strict ingest"):
            load_tickets(_write(tmp_path, pd.DataFrame(rows)), strict=True)

    def test_unresolved_row_with_null_metrics_is_valid(self, tmp_path):
        row = _valid_row(
            **{S.STATUS: "Open", S.RESOLUTION_TIME: None, S.CUSTOMER_RATING: None}
        )
        frame, report = load_tickets(_write(tmp_path, pd.DataFrame([row])))
        assert report.rows_rejected == 0
        assert bool(frame.iloc[0]["is_unresolved"]) is True


class TestFileLevelFailures:
    def test_missing_file(self, tmp_path):
        with pytest.raises(IngestError, match="not found"):
            load_tickets(tmp_path / "nope.csv")

    def test_header_only_file(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text(",".join(S.ALL_COLUMNS) + "\n")
        with pytest.raises(IngestError, match="no data rows"):
            load_tickets(path)

    def test_completely_empty_file(self, tmp_path):
        path = tmp_path / "blank.csv"
        path.write_text("")
        with pytest.raises(IngestError):
            load_tickets(path)

    def test_all_rows_invalid(self, tmp_path):
        rows = [_valid_row(**{S.TICKET_ID: f"TKT-{i}", S.PRIORITY: "Bogus"}) for i in range(3)]
        with pytest.raises(IngestError, match="Every row"):
            load_tickets(_write(tmp_path, pd.DataFrame(rows)))
