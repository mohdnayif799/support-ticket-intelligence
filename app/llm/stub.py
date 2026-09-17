"""Deterministic stub provider.

This is a test double, not a product feature. It exists so the test suite and CI
can exercise the full pipeline, including the API layer, with no API key, no
network and no cost. It is selected only by an explicit LLM_PROVIDER=stub.

It is not a silent fallback for the real providers. When Groq or Ollama fails at
runtime the pipeline surfaces the error rather than quietly degrading to keyword
matching, because an assessment system that claims to use an LLM should not
pretend to when it isn't.
"""

from __future__ import annotations

import json
import re


class StubProvider:
    """Maps a small set of question shapes to plans by keyword."""

    name = "stub"
    model = "deterministic-stub"
    schema_dialect = "plain"

    def __init__(self, **_ignored):
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def _extract_question(user: str) -> str:
        """Pull the real question out of the few-shot prompt.

        The planner prompt ends with 'Now convert this question:\\nQ: ...'. Without
        this, keyword matching would hit the worked examples instead.
        """
        marker = "Now convert this question:"
        tail = user.rsplit(marker, 1)[-1] if marker in user else user
        for line in tail.splitlines():
            line = line.strip()
            if line.startswith("Q:"):
                return line[2:].strip()
        return tail.strip()

    def complete_json(self, system: str, user: str, json_schema: dict) -> str:
        self.calls.append(("json", user))
        return json.dumps(self._plan_for(self._extract_question(user).lower()))

    def complete_text(self, system: str, user: str, max_tokens: int = 300) -> str:
        self.calls.append(("text", user))
        return ""  # forces the deterministic template path

    def health(self) -> dict:
        return {
            "provider": self.name,
            "model": self.model,
            "reachable": True,
            "strict_mode": True,
            "note": "test double; not for evaluation runs",
        }

    @staticmethod
    def _window(text: str) -> dict | None:
        for phrase, preset in (
            ("this month", "this_month"),
            ("last month", "last_month"),
            ("this week", "this_week"),
            ("last 7 days", "last_7_days"),
            ("last 30 days", "last_30_days"),
            ("today", "today"),
        ):
            if phrase in text:
                return {"preset": preset, "start": None, "end": None}
        return None

    def _plan_for(self, q: str) -> dict:
        plan: dict = {"intent": "aggregate", "filters": [], "reasoning": "stub keyword match"}
        window = self._window(q)
        if window:
            plan["time_window"] = window

        if "anomal" in q or "unusual" in q or "outlier" in q:
            types = []
            if "resolution" in q:
                types = ["resolution_outlier"]
            elif "response" in q:
                types = ["response_sla_breach"]
            elif "rating" in q:
                types = ["low_rating"]
            return {
                "intent": "anomaly",
                "anomaly_types": types or ["all"],
                **({"time_window": window} if window else {}),
                "reasoning": "stub: anomaly request",
            }

        for word, value in (
            ("critical", "Critical"),
            ("high priority", "High"),
            ("medium", "Medium"),
            ("low priority", "Low"),
        ):
            if word in q:
                plan["filters"].append({"field": "priority", "op": "eq", "value": value})
                break

        categories = (("billing", "Billing"), ("technical", "Technical"), ("general", "General"))
        for word, value in categories:
            if word in q:
                plan["filters"].append({"field": "category", "op": "eq", "value": value})
                break

        # A "within N hours" question is expressed through filters_any below, so a
        # status filter here would wrongly exclude the resolved-but-late tickets.
        within_clause = re.search(r"within (\d+) hours?", q)
        if not within_clause:
            if "unresolved" in q or "not resolved" in q or "outstanding" in q:
                plan["filters"].append(
                    {"field": "status", "op": "in", "values": ["Open", "Escalated"]}
                )
            elif "open" in q:
                plan["filters"].append({"field": "status", "op": "eq", "value": "Open"})
            elif "resolved" in q or "closed" in q:
                plan["filters"].append({"field": "status", "op": "eq", "value": "Resolved"})
            elif "escalated" in q:
                plan["filters"].append({"field": "status", "op": "eq", "value": "Escalated"})

        if "agent" in q:
            plan["group_by"] = ["agent_id"]
            ascending = any(w in q for w in ("lowest", "worst", "fewest"))
            plan["sort"] = {"by": "metric", "direction": "asc" if ascending else "desc"}
            plan["limit"] = 1

        if "average" in q or "avg" in q or "mean" in q:
            field = "customer_rating" if "rating" in q else "resolution_time_hrs"
            plan["aggregation"] = {"op": "avg", "field": field}
        elif "show" in q or "list" in q:
            plan["intent"] = "list"
            plan["limit"] = 20
            plan.pop("group_by", None)
            plan.pop("sort", None)
        else:
            plan["aggregation"] = {"op": "count", "field": None}

        if plan["intent"] == "aggregate" and "aggregation" not in plan:
            plan["aggregation"] = {"op": "count", "field": None}

        if within_clause:
            plan["filters_any"] = [
                {
                    "field": "resolution_time_hrs",
                    "op": "gt",
                    "value": float(within_clause.group(1)),
                },
                {"field": "status", "op": "ne", "value": "Resolved"},
            ]
        return plan
