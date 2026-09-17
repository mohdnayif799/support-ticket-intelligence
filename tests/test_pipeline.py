"""Pipeline tests.

A scripted provider is used so retry, repair and failure paths are exercised
deterministically. These are the behaviours that only show up when the model
misbehaves, which is exactly what cannot be tested against a live endpoint.
"""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.llm.base import LLMUnavailable
from app.pipeline import PipelineError, QueryPipeline


class ScriptedProvider:
    """Returns a queued response per call. Raises if the script runs dry."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, json_responses: list, text_response: str = ""):
        self._json = list(json_responses)
        self._text = text_response
        self.json_calls = 0
        self.text_calls = 0

    def complete_json(self, system, user, json_schema):
        self.json_calls += 1
        if not self._json:
            raise AssertionError("ScriptedProvider ran out of JSON responses")
        item = self._json.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)

    def complete_text(self, system, user, max_tokens=300):
        self.text_calls += 1
        if isinstance(self._text, Exception):
            raise self._text
        return self._text

    def health(self):
        return {"provider": self.name, "reachable": True}


COUNT_OPEN = {
    "intent": "aggregate",
    "filters": [{"field": "status", "op": "eq", "value": "Open"}],
    "aggregation": {"op": "count", "field": None},
}


@pytest.fixture
def cfg():
    return Settings(llm_provider="stub", narrate_answers=False)


def build(store, provider, settings):
    return QueryPipeline(store, provider, settings)


class TestHappyPath:
    def test_single_call_when_plan_is_valid(self, real_store, cfg):
        p = ScriptedProvider([COUNT_OPEN])
        r = build(real_store, p, cfg).answer("How many tickets are open?")
        assert r.result["value"] == 111
        assert p.json_calls == 1
        assert r.meta["attempts"][0]["ok"] is True

    def test_meta_reports_provider_and_timings(self, real_store, cfg):
        r = build(real_store, ScriptedProvider([COUNT_OPEN]), cfg).answer("open?")
        assert r.meta["provider"] == "scripted"
        assert r.meta["model"] == "scripted-1"
        assert isinstance(r.meta["execution_ms"], int)
        assert "reference_time" in r.meta


class TestRepairLoop:
    def test_invalid_plan_is_retried_and_recovered(self, real_store, cfg):
        """First response breaks a semantic rule; the repair attempt fixes it."""
        broken = {"intent": "aggregate"}  # aggregate with no aggregation
        p = ScriptedProvider([broken, COUNT_OPEN])
        r = build(real_store, p, cfg).answer("How many tickets are open?")
        assert r.result["value"] == 111
        assert p.json_calls == 2
        assert r.meta["attempts"][0]["ok"] is False
        assert "aggregation" in r.meta["attempts"][0]["error"]

    def test_malformed_json_is_retried(self, real_store, cfg):
        p = ScriptedProvider(["this is not json at all", COUNT_OPEN])
        r = build(real_store, p, cfg).answer("open?")
        assert r.result["value"] == 111
        assert p.json_calls == 2

    def test_markdown_fenced_json_is_accepted(self, real_store, cfg):
        fenced = "```json\n" + json.dumps(COUNT_OPEN) + "\n```"
        p = ScriptedProvider([fenced])
        r = build(real_store, p, cfg).answer("open?")
        assert r.result["value"] == 111
        assert p.json_calls == 1

    def test_gives_up_after_configured_retries(self, real_store):
        settings = Settings(llm_provider="stub", llm_max_retries=2, narrate_answers=False)
        p = ScriptedProvider([{"intent": "aggregate"}] * 3)
        with pytest.raises(PipelineError) as exc:
            build(real_store, p, settings).answer("open?")
        assert exc.value.stage == "planning"
        assert p.json_calls == 3

    def test_hallucinated_column_is_rejected_not_executed(self, real_store, cfg):
        """The model inventing a column must never reach pandas."""
        bad = {
            "intent": "aggregate",
            "filters": [{"field": "customer_email", "op": "eq", "value": "a@b.c"}],
            "aggregation": {"op": "count", "field": None},
        }
        p = ScriptedProvider([bad, COUNT_OPEN])
        r = build(real_store, p, cfg).answer("open?")
        assert r.result["value"] == 111
        assert "customer_email" not in json.dumps(r.plan)


class TestProviderFailure:
    def test_unavailable_provider_propagates(self, real_store, cfg):
        p = ScriptedProvider([LLMUnavailable("rate limited")])
        with pytest.raises(LLMUnavailable):
            build(real_store, p, cfg).answer("open?")

    def test_no_silent_fallback_to_keyword_matching(self, real_store, cfg):
        """A failed model must surface, not quietly degrade to a guess."""
        p = ScriptedProvider([LLMUnavailable("down")])
        with pytest.raises(LLMUnavailable):
            build(real_store, p, cfg).answer("How many tickets are open?")


class TestNarration:
    def test_llm_phrasing_is_used_when_available(self, real_store):
        settings = Settings(llm_provider="stub", narrate_answers=True)
        p = ScriptedProvider([COUNT_OPEN], text_response="There are 111 open tickets.")
        r = build(real_store, p, settings).answer("open?")
        assert r.answer == "There are 111 open tickets."
        assert r.meta["narrated_by_llm"] is True

    def test_falls_back_to_template_when_narration_fails(self, real_store):
        settings = Settings(llm_provider="stub", narrate_answers=True)
        p = ScriptedProvider([COUNT_OPEN], text_response=LLMUnavailable("narrator down"))
        r = build(real_store, p, settings).answer("open?")
        assert "111" in r.answer
        assert r.meta["narrated_by_llm"] is False

    def test_empty_narration_falls_back(self, real_store):
        settings = Settings(llm_provider="stub", narrate_answers=True)
        p = ScriptedProvider([COUNT_OPEN], text_response="   ")
        r = build(real_store, p, settings).answer("open?")
        assert "111" in r.answer
        assert r.meta["narrated_by_llm"] is False

    def test_narrator_never_computes(self, real_store):
        """The narrator receives results only; the number comes from Python."""
        settings = Settings(llm_provider="stub", narrate_answers=True)
        p = ScriptedProvider([COUNT_OPEN], text_response="Roughly 200 tickets.")
        r = build(real_store, p, settings).answer("open?")
        assert r.answer == "Roughly 200 tickets."  # prose is the model's
        assert r.result["value"] == 111  # the datum is not


class TestUnsupportedQuestions:
    def test_unsupported_intent_is_honoured(self, real_store, cfg):
        plan = {
            "intent": "unsupported",
            "unsupported_reason": "The dataset has no customer contact details.",
        }
        r = build(real_store, ScriptedProvider([plan]), cfg).answer(
            "What is the customer's phone number?"
        )
        assert "no customer contact" in r.answer
        assert r.result["value"] is None


class TestInputValidation:
    def test_empty_question_rejected(self, real_store, cfg):
        with pytest.raises(PipelineError) as exc:
            build(real_store, ScriptedProvider([]), cfg).answer("   ")
        assert exc.value.stage == "input"

    def test_overlong_question_rejected(self, real_store):
        settings = Settings(llm_provider="stub", max_query_length=50)
        with pytest.raises(PipelineError) as exc:
            build(real_store, ScriptedProvider([]), settings).answer("x" * 100)
        assert "too long" in exc.value.message


class TestAnomalyIntent:
    def test_anomaly_plan_runs_detectors(self, real_store, cfg):
        plan = {"intent": "anomaly", "anomaly_types": ["resolution_outlier"]}
        r = build(real_store, ScriptedProvider([plan]), cfg).answer("any anomalies?")
        assert r.anomalies is not None
        assert r.anomalies["detectors"][0]["count"] == 18

    def test_anomaly_plan_respects_window(self, real_store, cfg):
        plan = {
            "intent": "anomaly",
            "anomaly_types": ["all"],
            "time_window": {"preset": "this_week", "start": None, "end": None},
        }
        r = build(real_store, ScriptedProvider([plan]), cfg).answer("anomalies this week?")
        assert r.anomalies["tickets_in_scope"] < 500
