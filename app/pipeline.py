"""Query pipeline.

    question -> LLM -> QueryPlan -> validate -> deterministic execute -> narrate

The split is the point. The model decides what to compute; Python computes it.
A model that misreads a question produces a wrong but visible plan, which the
user can see in the response. A model cannot produce a wrong number, because it
never touches the arithmetic.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.anomaly.detectors import run_detectors
from app.config import Settings
from app.data.store import TicketStore
from app.domain import schema as S
from app.llm.base import LLMBadOutput, LLMError, LLMProvider
from app.llm.prompts import (
    build_narrator_messages,
    build_planner_messages,
    build_repair_messages,
)
from app.query.executor import ExecutionError, execute_plan
from app.query.plan import Intent, QueryPlan, plan_json_schema, schema_fingerprint
from app.query.timewindow import resolve_window

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Query could not be answered. Carries a user-facing message."""

    def __init__(self, message: str, *, stage: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.detail = detail


@dataclass
class QueryResponse:
    question: str
    answer: str
    plan: dict
    plan_summary: str
    result: dict
    anomalies: dict | None = None
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "plan": self.plan,
            "plan_summary": self.plan_summary,
            "result": self.result,
            "anomalies": self.anomalies,
            "meta": self.meta,
        }


def _strip_fences(text: str) -> str:
    """Remove markdown fences some models add around JSON."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.removesuffix("```")
        t = t.removeprefix("json").strip()
    return t.strip()


class QueryPipeline:
    def __init__(self, store: TicketStore, provider: LLMProvider, settings: Settings):
        self._store = store
        self._llm = provider
        self._settings = settings
        dialect = getattr(provider, "schema_dialect", "strict")
        self._schema = plan_json_schema(strict=dialect == "strict")
        logger.info(
            "Plan schema ready: dialect=%s fingerprint=%s",
            dialect,
            schema_fingerprint(self._schema),
        )

    # --- planning -----------------------------------------------------------

    def build_plan(self, question: str) -> tuple[QueryPlan, dict]:
        """Ask the model for a plan, retrying on validation failure.

        Strict mode makes schema violations rare, but semantic validators such as
        "avg needs a field" still reject plans. Those errors are fed back so the
        model can repair its own output.
        """
        schema_desc = self._store.schema_description()
        system, user = build_planner_messages(question, schema_desc)
        attempts: list[dict] = []
        last_raw, last_error = "", ""

        for attempt in range(self._settings.llm_max_retries + 1):
            started = time.perf_counter()
            if attempt > 0:
                system, user = build_repair_messages(
                    question, schema_desc, last_raw, last_error
                )
            # A dead provider propagates: it must be visible, not silently
            # swallowed. A bad generation is different and gets repaired below.
            try:
                raw = self._llm.complete_json(system, user, self._schema)
            except LLMBadOutput as exc:
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                last_error = str(exc)
                last_raw = ""
                attempts.append({"attempt": attempt + 1, "ok": False,
                                 "error": last_error, "latency_ms": elapsed_ms})
                logger.info("Generation rejected on attempt %d: %s", attempt + 1, last_error)
                continue
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            last_raw = raw

            try:
                payload = json.loads(_strip_fences(raw))
            except json.JSONDecodeError as exc:
                last_error = f"Response was not valid JSON: {exc}"
                attempts.append({"attempt": attempt + 1, "ok": False, "error": last_error,
                                 "latency_ms": elapsed_ms})
                continue

            try:
                plan = QueryPlan.model_validate(payload)
            except ValidationError as exc:
                last_error = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                    for e in exc.errors()[:5]
                )
                attempts.append({"attempt": attempt + 1, "ok": False, "error": last_error,
                                 "latency_ms": elapsed_ms})
                logger.info("Plan rejected on attempt %d: %s", attempt + 1, last_error)
                continue

            attempts.append({"attempt": attempt + 1, "ok": True, "latency_ms": elapsed_ms})
            return plan, {"attempts": attempts, "plan_latency_ms": elapsed_ms}

        raise PipelineError(
            "Could not turn that question into a valid query plan. Try rephrasing "
            "it, or ask about ticket counts, averages, agents, categories, "
            "priorities or time ranges.",
            stage="planning",
            detail=last_error,
        )

    # --- narration ----------------------------------------------------------

    def _template_answer(self, plan: QueryPlan, result: dict) -> str:
        """Deterministic phrasing. Used when narration is off or unavailable."""
        if plan.intent == Intent.UNSUPPORTED:
            return plan.unsupported_reason or "That question cannot be answered from this dataset."

        window = (result.get("window") or {}).get("label", "all time")
        scope = "" if window == "all time" else f" ({window})"

        if plan.intent == Intent.ANOMALY:
            return f"Anomaly scan complete{scope}."

        if plan.intent == Intent.LIST:
            n = result.get("matched_count", 0)
            if n == 0:
                return f"No tickets match that description{scope}."
            shown = result.get("row_count", n)
            tail = f", showing {shown}" if shown < n else ""
            return f"{n} ticket{'s' if n != 1 else ''} match{scope}{tail}."

        rows = result.get("rows") or []
        if plan.group_by and rows:
            key = plan.group_by[0].value
            top = rows[0]
            ascending = bool(plan.sort and plan.sort.direction.value == "asc")
            verb = "is lowest at" if ascending else "leads with"
            base = f"{top.get(key)} {verb} {top.get('metric')}{scope}."
            ties = result.get("ties") or []
            if len(ties) > 1:
                names = ", ".join(str(t.get(key)) for t in ties)
                base = f"{names} tie at {top.get('metric')}{scope}."
            return base

        value = result.get("value")
        if value is None:
            return f"No matching tickets had a value to compute{scope}."
        unit = result.get("unit") or ""
        return f"{value} {unit}{scope}.".replace("  ", " ")

    def _narrate(self, question: str, plan: QueryPlan, result: dict) -> tuple[str, bool]:
        """Phrase the result. Falls back to a template on any failure."""
        fallback = self._template_answer(plan, result)
        if not self._settings.narrate_answers or plan.intent == Intent.UNSUPPORTED:
            return fallback, False

        compact = {k: v for k, v in result.items() if k != "rows"}
        compact["rows"] = (result.get("rows") or [])[:5]
        system, user = build_narrator_messages(question, plan.describe(), compact)
        try:
            text = self._llm.complete_text(system, user, max_tokens=220)
        except LLMError as exc:
            logger.warning("Narration failed, using template: %s", exc)
            return fallback, False
        text = (text or "").strip()
        return (text, True) if text else (fallback, False)

    # --- entry point --------------------------------------------------------

    def answer(self, question: str) -> QueryResponse:
        """Answer a natural language question about the tickets."""
        question = question.strip()
        if not question:
            raise PipelineError("Ask a question about the tickets.", stage="input")
        if len(question) > self._settings.max_query_length:
            raise PipelineError(
                f"Question is too long (limit {self._settings.max_query_length} characters).",
                stage="input",
            )

        overall = time.perf_counter()
        plan, plan_meta = self.build_plan(question)

        frame = self._store.frame
        reference = self._store.reference_time
        anomalies = None

        exec_started = time.perf_counter()
        try:
            if plan.intent == Intent.ANOMALY:
                window = resolve_window(plan.time_window, reference)
                scoped = frame
                if not window.is_open:
                    if window.start is not None:
                        scoped = scoped[scoped[S.CREATED_AT] >= window.start]
                    if window.end is not None:
                        scoped = scoped[scoped[S.CREATED_AT] <= window.end]
                anomalies = run_detectors(
                    scoped, self._settings, reference, plan.anomaly_types, window
                )
                result = {
                    "intent": "anomaly",
                    "value": anomalies["total_flags"],
                    "unit": "flags",
                    "rows": [],
                    "columns": [],
                    "row_count": 0,
                    "matched_count": anomalies["tickets_in_scope"],
                    "truncated": False,
                    "window": window.as_dict(),
                    "ties": [],
                    "notes": [
                        f"{d['title']}: {d['count']}" for d in anomalies["detectors"]
                    ],
                }
            else:
                result = execute_plan(
                    plan, frame, reference, self._settings.max_row_limit
                ).as_dict()
        except ExecutionError as exc:
            raise PipelineError(
                f"The plan could not run against this data: {exc}",
                stage="execution",
                detail=str(exc),
            ) from exc
        exec_ms = round((time.perf_counter() - exec_started) * 1000)

        answer, narrated = self._narrate(question, plan, result)

        return QueryResponse(
            question=question,
            answer=answer,
            plan=plan.model_dump(mode="json", by_alias=True, exclude_none=False),
            plan_summary=plan.describe(),
            result=result,
            anomalies=anomalies,
            meta={
                "provider": self._llm.name,
                "model": self._llm.model,
                "narrated_by_llm": narrated,
                "execution_ms": exec_ms,
                "total_ms": round((time.perf_counter() - overall) * 1000),
                "reference_time": reference.isoformat(),
                **plan_meta,
            },
        )
