"""QueryPlan validation tests.

These cover the security boundary. A plan that fails validation never reaches
the executor, so the rejection cases matter as much as the acceptance ones.
"""

from __future__ import annotations

import contextlib
from unittest import mock

import pytest
from pydantic import ValidationError

from app.query.plan import (
    QueryPlan,
    _merge_union,
    _normalise_type_value,
    plan_json_schema,
    schema_fingerprint,
    validate_strict_schema,
)


def _plan(**over) -> dict:
    base = {"intent": "aggregate", "aggregation": {"op": "count", "field": None}}
    base.update(over)
    return base


class TestAcceptsValidPlans:
    def test_simple_count(self):
        p = QueryPlan.model_validate(
            _plan(filters=[{"field": "status", "op": "eq", "value": "Open"}])
        )
        assert p.filters[0].field_name.value == "status"

    def test_grouped_average_with_sort(self):
        p = QueryPlan.model_validate(
            {
                "intent": "aggregate",
                "aggregation": {"op": "avg", "field": "customer_rating"},
                "group_by": ["agent_id"],
                "sort": {"by": "metric", "direction": "asc"},
                "limit": 1,
            }
        )
        assert p.group_by[0].value == "agent_id"

    def test_filters_any_for_or_conditions(self):
        p = QueryPlan.model_validate(
            {
                "intent": "list",
                "filters": [{"field": "priority", "op": "eq", "value": "Critical"}],
                "filters_any": [
                    {"field": "resolution_time_hrs", "op": "gt", "value": 12},
                    {"field": "status", "op": "ne", "value": "Resolved"},
                ],
            }
        )
        assert len(p.filters_any) == 2

    def test_anomaly_intent_defaults_to_all_detectors(self):
        p = QueryPlan.model_validate({"intent": "anomaly"})
        assert p.anomaly_types[0].value == "all"

    def test_in_operator_with_values_list(self):
        p = QueryPlan.model_validate(
            _plan(filters=[{"field": "status", "op": "in", "values": ["Open", "Escalated"]}])
        )
        assert p.filters[0].values == ["Open", "Escalated"]

    def test_null_reasoning_coerced_to_empty_string(self):
        """Strict mode requires every field present, so nulls arrive for optionals."""
        p = QueryPlan.model_validate(_plan(reasoning=None, filters=None, group_by=None))
        assert p.reasoning == ""
        assert p.filters == []


class TestRejectsInvalidPlans:
    @pytest.mark.parametrize(
        "payload,label",
        [
            (_plan(filters=[{"field": "password", "op": "eq", "value": "x"}]), "unknown column"),
            (_plan(filters=[{"field": "status", "op": "DROP", "value": "x"}]), "unknown operator"),
            ({"intent": "aggregate", "aggregation": {"op": "avg", "field": "issue_summary"}}, "avg on text"),
            ({"intent": "aggregate"}, "aggregate with no aggregation"),
            ({"intent": "aggregate", "aggregation": {"op": "avg", "field": None}}, "avg with no field"),
            ({"intent": "list", "group_by": ["agent_id"]}, "group_by on list"),
            ({"intent": "unsupported"}, "unsupported with no reason"),
            (_plan(evil="payload"), "unexpected top-level key"),
            (_plan(filters=[{"field": "status", "op": "eq", "value": "x", "extra": 1}]), "unexpected filter key"),
            (_plan(filters=[{"field": "priority", "op": "gt", "value": "Critical"}]), "gt against a string"),
            (_plan(filters=[{"field": "status", "op": "in"}]), "in with no values"),
            (_plan(filters=[{"field": "status", "op": "eq"}]), "eq with no value"),
            (_plan(filters=[{"field": "agent_id", "op": "contains", "value": "x"}]), "contains off issue_summary"),
            (_plan(limit=99999), "limit above ceiling"),
            (_plan(limit=0), "limit below floor"),
            ({"intent": "teleport", "aggregation": {"op": "count"}}, "unknown intent"),
        ],
    )
    def test_rejected(self, payload, label):
        with pytest.raises(ValidationError):
            QueryPlan.model_validate(payload)

    def test_sql_injection_string_cannot_become_a_column(self):
        with pytest.raises(ValidationError):
            QueryPlan.model_validate(
                _plan(filters=[{"field": "status; DROP TABLE tickets", "op": "eq", "value": "Open"}])
            )

    def test_injection_payload_in_value_stays_inert(self):
        """A hostile value is allowed through as data; it is only ever compared."""
        p = QueryPlan.model_validate(
            _plan(filters=[{"field": "status", "op": "eq", "value": "'; DROP TABLE--"}])
        )
        assert p.filters[0].value == "'; DROP TABLE--"

    def test_custom_window_needs_bounds(self):
        with pytest.raises(ValidationError):
            QueryPlan.model_validate(_plan(time_window={"preset": "custom"}))

    def test_inverted_window_rejected(self):
        with pytest.raises(ValidationError):
            QueryPlan.model_validate(
                _plan(
                    time_window={
                        "preset": "custom",
                        "start": "2024-03-10T00:00:00",
                        "end": "2024-03-01T00:00:00",
                    }
                )
            )


class TestStrictSchema:
    """Pins the dialect Groq's structured-output validator accepts.

    Groq rejected an earlier schema with: "anyOf branches must be disambiguated
    via a required discriminator (const/enum) or by key-set exclusion with
    additionalProperties:false". It validates each branch without resolving
    $ref, so a bare reference branch is indistinguishable from its siblings.
    The strict dialect therefore carries no $ref and no anyOf at all.
    """

    @staticmethod
    def _walk(node, path="#"):
        yield path, node
        if isinstance(node, dict):
            for k, v in node.items():
                yield from TestStrictSchema._walk(v, f"{path}/{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from TestStrictSchema._walk(v, f"{path}/{i}")

    def test_no_anyof_anywhere(self):
        schema = plan_json_schema(strict=True)
        offenders = [p for p, n in self._walk(schema) if isinstance(n, dict) and "anyOf" in n]
        assert offenders == []

    def test_no_refs_or_defs(self):
        schema = plan_json_schema(strict=True)
        assert "$defs" not in schema
        offenders = [p for p, n in self._walk(schema) if isinstance(n, dict) and "$ref" in n]
        assert offenders == []

    def test_every_object_is_closed_and_fully_required(self):
        schema = plan_json_schema(strict=True)
        problems = []
        for path, node in self._walk(schema):
            if not isinstance(node, dict) or "properties" not in node:
                continue
            types = node.get("type")
            if types != "object" and not (isinstance(types, list) and "object" in types):
                continue
            if set(node.get("required", [])) != set(node["properties"]):
                problems.append(f"{path}: required != properties")
            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: not closed")
        assert problems == []

    def test_unsupported_validation_keywords_are_stripped(self):
        """Pydantic re-validates on arrival, so these bounds lose nothing."""
        schema = plan_json_schema(strict=True)
        banned = {"maxLength", "minLength", "maxItems", "minItems", "minimum", "maximum", "default"}
        offenders = [
            f"{p}:{k}"
            for p, n in self._walk(schema)
            if isinstance(n, dict)
            for k in n
            if k in banned
        ]
        assert offenders == []

    def test_nullable_enum_permits_null(self):
        """A nullable enum must list null, or the enum contradicts the type."""
        field = plan_json_schema(strict=True)["properties"]["aggregation"]["properties"]["field"]
        assert field["type"] == ["string", "null"]
        assert None in field["enum"]
        assert "customer_rating" in field["enum"]

    def test_nullable_object_keeps_its_properties(self):
        tw = plan_json_schema(strict=True)["properties"]["time_window"]
        assert "object" in tw["type"] and "null" in tw["type"]
        assert set(tw["properties"]) == {"preset", "start", "end"}

    def test_scalar_union_becomes_a_type_list(self):
        value = plan_json_schema(strict=True)["properties"]["filters"]["items"]["properties"]["value"]
        assert set(value["type"]) == {"string", "number", "boolean", "null"}

    def test_filter_value_does_not_carry_integer_and_number(self):
        """Regression: Groq rejected "cannot include both 'integer' and 'number'".

        Filter.value is typed str | float | int | bool | None, so Pydantic emits
        both numeric types. `integer` is a subset of `number`, so the pair is
        redundant and this validator refuses it.
        """
        value = plan_json_schema(strict=True)["properties"]["filters"]["items"]["properties"]["value"]
        assert "integer" not in value["type"]
        assert "number" in value["type"]
        assert "null" in value["type"]

    def test_no_type_list_anywhere_carries_both_numeric_types(self):
        """The rule holds across the whole tree, not just the field that failed."""
        offenders = [
            path
            for path, node in self._walk(plan_json_schema(strict=True))
            if isinstance(node, dict)
            and isinstance(node.get("type"), list)
            and {"integer", "number"} <= set(node["type"])
        ]
        assert offenders == []

    @pytest.mark.parametrize(
        "given,expected",
        [
            (["integer", "number"], "number"),
            (["number", "integer"], "number"),
            (["number", "integer", "null"], ["number", "null"]),
            (["string", "number", "integer", "boolean", "null"], ["string", "number", "boolean", "null"]),
            (["integer"], "integer"),
            (["integer", "null"], ["integer", "null"]),
            (["number", "number", "null"], ["number", "null"]),
            (["string", "null"], ["string", "null"]),
            ("string", "string"),
        ],
    )
    def test_type_normalisation_rules(self, given, expected):
        assert _normalise_type_value(given) == expected

    def test_integer_is_not_widened_to_number(self):
        """Only the integer+number collision is a problem.

        `limit` is nullable because strict mode forces it into `required`, but
        it must stay an integer: the normaliser must not gratuitously promote a
        lone integer to number.
        """
        limit = plan_json_schema(strict=True)["properties"]["limit"]["type"]
        assert "integer" in limit
        assert "number" not in limit

    def test_nullable_numeric_union_keeps_null(self):
        """Normalising must not strip nullability while removing integer."""
        merged = _merge_union(
            [{"type": "number"}, {"type": "integer"}, {"type": "null"}]
        )
        assert merged["type"] == ["number", "null"]

    def test_normalisation_preserves_enum_values(self):
        merged = _merge_union(
            [{"type": "string", "enum": ["a", "b"]}, {"type": "null"}]
        )
        assert merged["type"] == ["string", "null"]
        assert merged["enum"] == ["a", "b", None]

    def test_plain_dialect_is_untouched_pydantic_output(self):
        """Ollama's grammar compiler handles $ref and anyOf natively."""
        plain = plan_json_schema(strict=False)
        assert plain == QueryPlan.model_json_schema(by_alias=True)
        assert "$defs" in plain

    def test_plain_dialect_still_carries_both_numeric_types(self):
        """Proof the Groq-only normalisation did not leak into Ollama's path."""
        plain = plan_json_schema(strict=False)
        branches = plain["$defs"]["Filter"]["properties"]["value"]["anyOf"]
        types = {b.get("type") for b in branches}
        assert {"integer", "number"} <= types

    def test_strict_schema_is_json_serialisable(self):
        import json

        json.dumps(plan_json_schema(strict=True))

    def test_describe_is_readable(self):
        p = QueryPlan.model_validate(
            _plan(filters=[{"field": "status", "op": "eq", "value": "Open"}])
        )
        assert "count(*)" in p.describe()


class TestGroqRequestSchema:
    """Checks the schema in the real request body, not just the generator output.

    The generator being correct is necessary but not sufficient: what matters is
    the object that lands in response_format.json_schema.schema. These tests
    build the actual request the provider would POST and inspect that.
    """

    @staticmethod
    def _captured_request_schema():
        from unittest.mock import patch as mock_patch

        from app.config import Settings
        from app.data.store import TicketStore
        from app.llm.groq_client import GroqProvider
        from app.pipeline import QueryPipeline

        captured: dict = {}

        def fake_post(self, payload):
            captured["payload"] = payload
            raise RuntimeError("stop after capture")

        settings = Settings(llm_provider="groq", groq_api_key="dummy", narrate_answers=False)
        store = TicketStore.from_csv(settings)
        provider = GroqProvider(api_key="dummy", model="openai/gpt-oss-20b")
        pipeline = QueryPipeline(store, provider, settings)

        with contextlib.suppress(RuntimeError), mock_patch.object(
            GroqProvider, "_post", fake_post
        ):
            pipeline.answer("How many tickets are open?")
        return captured["payload"]

    def test_request_uses_strict_json_schema(self):
        body = self._captured_request_schema()
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True

    def test_sent_schema_has_no_rejected_constructs(self):
        """End-to-end guard over the exact payload, using the shared validator."""
        body = self._captured_request_schema()
        sent = body["response_format"]["json_schema"]["schema"]
        assert validate_strict_schema(sent) == []

    @pytest.mark.parametrize("array_field", ["filters", "filters_any"])
    def test_both_filter_arrays_are_clean_in_the_request(self, array_field):
        """Regression: Groq reported filters first, then filters_any.

        Both arrays hold the same Filter model, so a fix that misses one is a
        fix that misses both. Parametrised so neither can regress silently.
        """
        sent = self._captured_request_schema()["response_format"]["json_schema"]["schema"]
        value = sent["properties"][array_field]["items"]["properties"]["value"]
        assert "integer" not in value["type"]
        assert set(value["type"]) == {"string", "number", "boolean", "null"}

    def test_filters_and_filters_any_are_structurally_identical(self):
        """They share one model; divergence means the transform is order-dependent."""
        sent = self._captured_request_schema()["response_format"]["json_schema"]["schema"]
        assert sent["properties"]["filters"]["items"] == sent["properties"]["filters_any"]["items"]


class TestStrictSchemaGuard:
    """The invariant guard that turns a remote 400 into a local failure."""

    def test_clean_schema_reports_no_problems(self):
        assert validate_strict_schema(plan_json_schema(strict=True)) == []

    def test_guard_catches_numeric_collision(self):
        import copy as _copy

        bad = _copy.deepcopy(plan_json_schema(strict=True))
        bad["properties"]["filters_any"]["items"]["properties"]["value"]["type"] = [
            "string",
            "number",
            "integer",
            "null",
        ]
        problems = validate_strict_schema(bad)
        assert any("filters_any" in p and "integer" in p for p in problems)

    def test_guard_catches_anyof_and_refs(self):
        problems = validate_strict_schema(
            {"type": "object", "properties": {"x": {"anyOf": [{"$ref": "#/$defs/Y"}]}},
             "required": ["x"], "additionalProperties": False}
        )
        assert any("anyOf" in p for p in problems)
        assert any("$ref" in p for p in problems)

    def test_guard_catches_nullable_enum_without_null(self):
        problems = validate_strict_schema({"type": ["string", "null"], "enum": ["a", "b"]})
        assert any("nullable enum" in p for p in problems)

    def test_guard_catches_open_object(self):
        problems = validate_strict_schema({"type": "object", "properties": {"a": {"type": "string"}}})
        assert any("not closed" in p for p in problems)
        assert any("required" in p for p in problems)

    def test_fingerprint_is_stable_and_short(self):
        a = schema_fingerprint(plan_json_schema(strict=True))
        b = schema_fingerprint(plan_json_schema(strict=True))
        assert a == b and len(a) == 12

    def test_plain_dialect_bypasses_the_guard(self):
        """Ollama's schema legitimately contains anyOf and $ref."""
        plain = plan_json_schema(strict=False)
        assert validate_strict_schema(plain)  # would fail strict, and that is correct
        assert "$defs" in plain


class TestStrictModeNullability:
    """Strict mode has no optional properties.

    Every key must sit in `required` and the model must emit a value for each,
    so a field Pydantic treated as optional has to be widened to accept null.
    Groq rejected an earlier build with:

        '/limit' does not validate with /properties/limit/type:
        expected integer, but got null

    That was `limit` forced into `required` while still typed as a bare integer,
    leaving the model no legal way to skip it. Nine fields were affected; limit
    was simply the first one the validator reached.
    """

    @staticmethod
    def _optional_fields(model) -> set[str]:
        """Fields Pydantic leaves out of `required`, i.e. those with defaults."""
        raw = model.model_json_schema(by_alias=True)
        return set(raw.get("properties", {})) - set(raw.get("required", []))

    def test_every_optional_field_is_nullable_in_strict_schema(self):
        """The general invariant, checked against the models themselves."""
        from app.query.plan import Aggregation, Filter, Sort, TimeWindow

        strict = plan_json_schema(strict=True)
        targets = [
            ("QueryPlan", QueryPlan, strict),
            ("Filter", Filter, strict["properties"]["filters"]["items"]),
            ("TimeWindow", TimeWindow, strict["properties"]["time_window"]),
            ("Aggregation", Aggregation, strict["properties"]["aggregation"]),
            ("Sort", Sort, strict["properties"]["sort"]),
        ]
        problems = []
        for name, model, node in targets:
            for field in self._optional_fields(model):
                types = node["properties"][field].get("type")
                if types is not None and "null" not in types:
                    problems.append(f"{name}.{field} forced required but not nullable: {types}")
        assert problems == []

    def test_limit_accepts_null(self):
        """The exact field named in the Groq error."""
        assert plan_json_schema(strict=True)["properties"]["limit"]["type"] == ["integer", "null"]

    def test_nullable_enum_lists_null(self):
        preset = plan_json_schema(strict=True)["properties"]["time_window"]["properties"]["preset"]
        assert "null" in preset["type"]
        assert None in preset["enum"]

    def test_model_answering_null_everywhere_still_validates(self):
        """The round trip that would have caught this before it shipped.

        Simulates a strict-mode response where the model fills every optional
        slot with null, and asserts Pydantic recovers the declared defaults.
        """
        payload = {
            "intent": "aggregate",
            "filters": None,
            "filters_any": None,
            "time_window": None,
            "aggregation": {"op": "count", "field": None},
            "group_by": None,
            "sort": None,
            "limit": None,
            "anomaly_types": None,
            "reasoning": None,
            "unsupported_reason": None,
        }
        plan = QueryPlan.model_validate(payload)
        assert plan.limit == 20
        assert plan.filters == []
        assert plan.filters_any == []
        assert plan.group_by == []
        assert plan.anomaly_types == []
        assert plan.reasoning == ""
        assert plan.unsupported_reason == ""

    def test_nested_nulls_recover_defaults(self):
        plan = QueryPlan.model_validate(
            {
                "intent": "list",
                "time_window": {"preset": None, "start": None, "end": None},
                "sort": {"by": "created_at", "direction": None},
                "limit": None,
            }
        )
        assert plan.time_window.preset.value == "all_time"
        assert plan.sort.direction.value == "desc"
        assert plan.limit == 20

    def test_real_values_still_win_over_defaults(self):
        """Widening must not swallow values the model actually supplied."""
        plan = QueryPlan.model_validate(
            {
                "intent": "list",
                "limit": 5,
                "sort": {"by": "metric", "direction": "asc"},
                "time_window": {"preset": "this_month", "start": None, "end": None},
            }
        )
        assert plan.limit == 5
        assert plan.sort.direction.value == "asc"
        assert plan.time_window.preset.value == "this_month"

    def test_limit_bounds_still_enforced(self):
        """Null maps to the default; an out-of-range number is still rejected."""
        with pytest.raises(ValidationError):
            QueryPlan.model_validate({"intent": "list", "limit": 9999})


class TestPromptMatchesStrictSchema:
    """The few-shot examples must satisfy the same contract as the schema.

    Groq rejected a generation with:

        '/filters/0' does not validate with /properties/filters/items/required:
        missing properties: 'values'

    The strict schema forces every property into `required`, so a filter must
    carry field, op, value AND values. The hand-written examples showed only
    three of the four, the model copied them, and the output failed validation
    after generation. The examples are now derived from QueryPlan itself.
    """

    @staticmethod
    def _example_plans() -> list[dict]:
        import json as _json

        from app.llm.prompts import build_planner_messages

        _, user = build_planner_messages("test question", "schema description")
        return [
            _json.loads(line[3:]) for line in user.splitlines() if line.startswith("A: ")
        ]

    def test_there_are_examples_to_check(self):
        assert len(self._example_plans()) >= 6

    def test_every_example_filter_carries_values(self):
        """The exact field named in the Groq error."""
        for plan in self._example_plans():
            for key in ("filters", "filters_any"):
                for f in plan.get(key) or []:
                    assert "values" in f, f"{key} filter missing 'values': {f}"

    def test_every_example_filter_has_all_required_keys(self):
        required = set(
            plan_json_schema(strict=True)["properties"]["filters"]["items"]["required"]
        )
        for plan in self._example_plans():
            for key in ("filters", "filters_any"):
                for f in plan.get(key) or []:
                    assert set(f) == required, f"{f} != required {sorted(required)}"

    def test_every_example_has_all_top_level_keys(self):
        required = set(plan_json_schema(strict=True)["required"])
        for plan in self._example_plans():
            assert set(plan) == required, f"missing {sorted(required - set(plan))}"

    def test_every_example_carries_all_keys_on_nested_objects(self):
        strict = plan_json_schema(strict=True)
        for plan in self._example_plans():
            for name in ("time_window", "aggregation", "sort"):
                obj = plan.get(name)
                if obj is None:
                    continue
                expected = set(strict["properties"][name]["required"])
                assert set(obj) == expected, f"{name}: {set(obj)} != {expected}"

    def test_every_example_round_trips_through_validation(self):
        """A malformed example must fail loudly, not reach the model."""
        for plan in self._example_plans():
            QueryPlan.model_validate(plan)

    def test_a_filter_without_values_is_detectably_incomplete(self):
        """Documents why the omission matters: strict mode requires the key."""
        required = set(
            plan_json_schema(strict=True)["properties"]["filters"]["items"]["required"]
        )
        legacy = {"field": "priority", "op": "eq", "value": "Critical"}
        assert "values" in required
        assert set(legacy) != required

    def test_prompt_states_the_all_fields_rule(self):
        from app.llm.prompts import PLANNER_SYSTEM

        assert "every field" in PLANNER_SYSTEM.lower()


class TestBadGenerationIsRetried:
    """A json_validate 400 is a bad generation, not a dead provider."""

    def test_groq_maps_json_validate_400_to_bad_output(self):
        """Exercises the real _post, by faking only the HTTP response."""
        import httpx

        from app.llm.base import LLMBadOutput
        from app.llm.groq_client import GroqProvider

        body = (
            '{"error":{"message":"Generated JSON does not match the expected schema.",'
            '"type":"invalid_request_error","code":"json_validate"}}'
        )
        provider = GroqProvider(api_key="dummy", model="openai/gpt-oss-20b")

        def fake_http_post(self, url, **kwargs):  # noqa: ARG001
            return httpx.Response(400, text=body, request=httpx.Request("POST", url))

        with mock.patch.object(httpx.Client, "post", fake_http_post), pytest.raises(
            LLMBadOutput, match="failed the schema"
        ):
            provider.complete_json("sys", "user", {"type": "object"})

    def test_a_genuine_provider_failure_still_surfaces(self):
        """Only json_validate is retryable; a 500 must stay fatal."""
        import httpx

        from app.llm.base import LLMUnavailable
        from app.llm.groq_client import GroqProvider

        provider = GroqProvider(api_key="dummy", model="openai/gpt-oss-20b")

        def fake_http_post(self, url, **kwargs):  # noqa: ARG001
            return httpx.Response(500, text="upstream exploded", request=httpx.Request("POST", url))

        with mock.patch.object(httpx.Client, "post", fake_http_post), pytest.raises(
            LLMUnavailable
        ):
            provider.complete_json("sys", "user", {"type": "object"})

    def test_pipeline_retries_after_a_rejected_generation(self):
        """The repair loop must get a second chance instead of returning 503."""
        import json as _json

        from app.config import Settings
        from app.data.store import TicketStore
        from app.llm.base import LLMBadOutput
        from app.pipeline import QueryPipeline

        good = {
            "intent": "aggregate",
            "filters": [{"field": "status", "op": "eq", "value": "Open", "values": None}],
            "aggregation": {"op": "count", "field": None},
        }

        class Flaky:
            name, model, schema_dialect = "flaky", "flaky-1", "strict"

            def __init__(self):
                self.calls = 0

            def complete_json(self, system, user, json_schema):  # noqa: ARG002
                self.calls += 1
                if self.calls == 1:
                    raise LLMBadOutput("Generated JSON failed the schema: missing 'values'")
                return _json.dumps(good)

            def complete_text(self, system, user, max_tokens=300):  # noqa: ARG002
                return ""

            def health(self):
                return {"reachable": True}

        settings = Settings(llm_provider="stub", narrate_answers=False)
        store = TicketStore.from_csv(settings)
        provider = Flaky()
        result = QueryPipeline(store, provider, settings).answer("How many are open?")

        assert provider.calls == 2
        assert result.result["value"] == 111
        assert result.meta["attempts"][0]["ok"] is False
        assert "values" in result.meta["attempts"][0]["error"]
