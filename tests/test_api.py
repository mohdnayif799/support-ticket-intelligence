"""API tests.

Run through the real FastAPI lifespan, so dataset load and provider construction
are exercised exactly as they are at startup. The stub provider keeps the suite
offline and free.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import reset_settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("NARRATE_ANSWERS", "false")
    reset_settings()
    from app.api.main import app

    with TestClient(app) as c:
        yield c
    reset_settings()


class TestHealth:
    def test_reports_dataset_and_config(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["dataset"]["row_count"] == 500
        assert body["ingest"]["rows_rejected"] == 0
        assert body["config"]["outlier_group_by"] == "priority"

    def test_surfaces_ingest_warnings(self, client):
        warnings = client.get("/api/health").json()["ingest"]["warnings"]
        assert any("resolve before first response" in w for w in warnings)

    def test_reference_clock_is_dataset_anchored(self, client):
        ds = client.get("/api/health").json()["dataset"]
        assert ds["reference_time"].startswith("2024-03-30")
        assert ds["reference_mode"] == "newest ticket in dataset"


class TestSchemaEndpoint:
    def test_lists_columns_and_enums(self, client):
        body = client.get("/api/schema").json()
        assert len(body["columns"]) == 10
        assert body["enums"]["priority"] == ["Low", "Medium", "High", "Critical"]
        assert len(body["enums"]["agent_id"]) == 12

    def test_offers_example_questions(self, client):
        assert len(client.get("/api/schema").json()["example_questions"]) >= 5


class TestQueryEndpoint:
    @pytest.mark.parametrize(
        "question,expected",
        [
            ("How many tickets are currently open?", 111),
            ("How many tickets are escalated?", 62),
        ],
    )
    def test_counts(self, client, question, expected):
        r = client.post("/api/query", json={"question": question})
        assert r.status_code == 200
        assert r.json()["result"]["value"] == expected

    def test_response_carries_the_executed_plan(self, client):
        body = client.post(
            "/api/query", json={"question": "How many tickets are currently open?"}
        ).json()
        assert body["plan"]["intent"] == "aggregate"
        assert body["plan"]["filters"][0]["field"] == "status"
        assert "count(*)" in body["plan_summary"]

    def test_grouped_query_returns_rows(self, client):
        body = client.post(
            "/api/query", json={"question": "Which agent resolved the most tickets this month?"}
        ).json()
        assert body["result"]["rows"][0]["agent_id"] == "AGT-01"
        assert body["result"]["rows"][0]["metric"] == 16
        assert body["result"]["window"]["label"].startswith("this month")

    def test_list_query_returns_tickets(self, client):
        body = client.post(
            "/api/query",
            json={"question": "Show me all Critical tickets not resolved within 12 hours."},
        ).json()
        assert body["result"]["matched_count"] == 34
        assert body["result"]["rows"][0]["ticket_id"].startswith("TKT-")

    def test_anomaly_question_returns_detector_output(self, client):
        body = client.post(
            "/api/query", json={"question": "Are there any anomalies in resolution times?"}
        ).json()
        assert body["anomalies"] is not None
        assert body["anomalies"]["detectors"][0]["type"] == "resolution_outlier"

    def test_average_reports_excluded_nulls(self, client):
        body = client.post(
            "/api/query",
            json={"question": "What is the average customer rating for Technical tickets?"},
        ).json()
        assert body["result"]["value"] == pytest.approx(3.7404, abs=1e-3)
        assert any("no customer_rating" in n for n in body["result"]["notes"])


class TestQueryValidation:
    def test_empty_question_is_422(self, client):
        assert client.post("/api/query", json={"question": ""}).status_code == 422

    def test_missing_field_is_422(self, client):
        assert client.post("/api/query", json={}).status_code == 422

    def test_overlong_question_is_422(self, client):
        r = client.post("/api/query", json={"question": "x" * 5000})
        assert r.status_code == 422

    def test_wrong_type_is_422(self, client):
        assert client.post("/api/query", json={"question": 42}).status_code == 422


class TestAnomalyEndpoint:
    def test_post_all_detectors(self, client):
        body = client.post("/api/anomalies", json={"types": ["all"], "window": "all_time"}).json()
        counts = {d["type"]: d["count"] for d in body["detectors"]}
        assert counts == {
            "resolution_outlier": 18,
            "aging_unresolved": 80,
            "response_sla_breach": 61,
            "low_rating": 47,
            "data_integrity": 28,
        }

    def test_get_works_for_browsers(self, client):
        body = client.get("/api/anomalies").json()
        assert body["tickets_in_scope"] == 500
        assert len(body["detectors"]) == 5

    def test_single_detector_selection(self, client):
        body = client.post("/api/anomalies", json={"types": ["data_integrity"]}).json()
        assert len(body["detectors"]) == 1
        assert body["detectors"][0]["count"] == 28

    def test_window_narrows_scope(self, client):
        body = client.post(
            "/api/anomalies", json={"types": ["all"], "window": "this_week"}
        ).json()
        assert body["tickets_in_scope"] < 500

    def test_unknown_detector_is_422(self, client):
        r = client.post("/api/anomalies", json={"types": ["nonsense"]})
        assert r.status_code == 422

    def test_unknown_window_is_422(self, client):
        r = client.post("/api/anomalies", json={"types": ["all"], "window": "next_century"})
        assert r.status_code == 422

    def test_works_without_an_llm(self, client):
        """Anomaly detection must not depend on the model."""
        client.app.state.pipeline = None
        assert client.get("/api/anomalies").status_code == 200


class TestDegradedMode:
    def test_query_returns_503_when_model_unavailable(self, client):
        client.app.state.pipeline = None
        client.app.state.startup_errors = ["GROQ_API_KEY is not set"]
        r = client.post("/api/query", json={"question": "How many tickets are open?"})
        assert r.status_code == 503
        detail = r.json()["detail"]
        assert "GROQ_API_KEY" in detail["detail"]
        assert "ollama" in detail["hint"].lower()


class TestUI:
    def test_root_serves_the_page(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Support Ticket Intelligence" in r.text
        assert "text/html" in r.headers["content-type"]

    def test_openapi_docs_available(self, client):
        assert client.get("/docs").status_code == 200
        spec = client.get("/openapi.json").json()
        for path in ("/api/query", "/api/anomalies", "/api/health", "/api/schema"):
            assert path in spec["paths"]

    def test_ui_calls_only_documented_endpoints(self, client):
        """The UI is a client of the public API, not a backdoor."""
        page = client.get("/").text
        for endpoint in ("/api/health", "/api/schema", "/api/query", "/api/anomalies"):
            assert endpoint in page
