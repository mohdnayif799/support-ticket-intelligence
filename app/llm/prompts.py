"""Prompt construction.

Two prompts, each with one job.

PLANNER turns a question into a QueryPlan. It never sees ticket rows and never
does arithmetic, so it cannot invent a number. Few-shot examples cover the five
sample queries in the brief plus the failure modes that matter: unresolved
semantics, null handling, and questions the DSL cannot express.

NARRATOR turns already-computed results into a sentence. It receives only the
computed output, so it has nothing to miscalculate.
"""

from __future__ import annotations

import json

from app.query.plan import QueryPlan

PLANNER_SYSTEM = """\
You convert questions about a customer support ticket dataset into a QueryPlan \
JSON object. You do not answer questions and you never compute numbers. A \
separate deterministic engine executes your plan against the data.

{schema}

Rules:
1. Use only the column names and enum values listed above. Values are \
case-sensitive: "Critical", not "critical".
2. "Unresolved", "still open", "outstanding", "pending" all mean status is Open \
OR Escalated. Use a filter with op "in" and values ["Open","Escalated"]. \
"Open" alone, when contrasted with other statuses, means status equals "Open".
3. resolution_time_hrs and customer_rating are null for every ticket that is not \
Resolved. Aggregates skip nulls automatically, so do not add a not_null filter \
unless the question is specifically about tickets that have the value.
4. Relative dates use a time_window preset. Do not compute dates yourself. \
Presets resolve against the reference date shown above, not today.
5. Put conditions that must all hold in "filters". Put alternatives in \
"filters_any", which are ORed together and then ANDed with "filters". \
"Not resolved within N hours" means resolution_time_hrs > N OR the ticket is \
still unresolved, so it needs filters_any.
6. Use intent "aggregate" for a number or a per-group ranking, "list" to show \
matching tickets, "anomaly" to run the anomaly detectors.
7. "Top", "most", "highest", "best" mean sort by metric descending with limit 1. \
"Lowest", "worst", "fewest" mean ascending with limit 1. Raise the limit only if \
the question asks for several.
8. If the question cannot be expressed with these columns and operators, or asks \
for something outside this dataset, set intent to "unsupported" and explain why \
in unsupported_reason. Never guess.
9. Every object must contain every field defined for it. Where a field does not \
apply, send null rather than leaving it out. A filter always carries both \
"value" and "values": set "value" and leave "values" null for single-operand \
operators, and the reverse for "in" and "not_in".
10. Fill "reasoning" with one short line describing the mapping.
"""

# Each example is (question, plan). Kept compact: long few-shots crowd the
# context and the schema already constrains the output shape.
_EXAMPLES: list[tuple[str, dict]] = [
    (
        "How many tickets are currently open?",
        {
            "intent": "aggregate",
            "filters": [{"field": "status", "op": "eq", "value": "Open"}],
            "aggregation": {"op": "count", "field": None},
            "reasoning": "Count tickets whose status is Open.",
        },
    ),
    (
        "Which agent resolved the most tickets this month?",
        {
            "intent": "aggregate",
            "filters": [{"field": "status", "op": "eq", "value": "Resolved"}],
            "time_window": {"preset": "this_month", "start": None, "end": None},
            "aggregation": {"op": "count", "field": None},
            "group_by": ["agent_id"],
            "sort": {"by": "metric", "direction": "desc"},
            "limit": 1,
            "reasoning": "Count resolved tickets per agent within this month, take the top.",
        },
    ),
    (
        "Show me all Critical tickets not resolved within 12 hours.",
        {
            "intent": "list",
            "filters": [{"field": "priority", "op": "eq", "value": "Critical"}],
            "filters_any": [
                {"field": "resolution_time_hrs", "op": "gt", "value": 12},
                {"field": "status", "op": "ne", "value": "Resolved"},
            ],
            "limit": 100,
            "reasoning": "Critical tickets that either took over 12h or are still unresolved.",
        },
    ),
    (
        "What is the average customer rating for Technical category tickets?",
        {
            "intent": "aggregate",
            "filters": [{"field": "category", "op": "eq", "value": "Technical"}],
            "aggregation": {"op": "avg", "field": "customer_rating"},
            "reasoning": "Mean customer_rating over Technical tickets.",
        },
    ),
    (
        "Are there any anomalies in resolution times this week?",
        {
            "intent": "anomaly",
            "time_window": {"preset": "this_week", "start": None, "end": None},
            "anomaly_types": ["resolution_outlier"],
            "reasoning": "Run the resolution-time outlier detector over this week.",
        },
    ),
    (
        "Which agent has the lowest average customer rating?",
        {
            "intent": "aggregate",
            "aggregation": {"op": "avg", "field": "customer_rating"},
            "group_by": ["agent_id"],
            "sort": {"by": "metric", "direction": "asc"},
            "limit": 1,
            "reasoning": "Mean rating per agent, ascending, take the lowest.",
        },
    ),
    (
        "How many unresolved billing tickets are there?",
        {
            "intent": "aggregate",
            "filters": [
                {"field": "category", "op": "eq", "value": "Billing"},
                {"field": "status", "op": "in", "values": ["Open", "Escalated"]},
            ],
            "aggregation": {"op": "count", "field": None},
            "reasoning": "Billing tickets in a non-resolved state.",
        },
    ),
    (
        "Who is the customer on ticket TKT-001 and what is their phone number?",
        {
            "intent": "unsupported",
            "unsupported_reason": (
                "The dataset has no customer name or contact columns; it only "
                "covers ticket metadata, timings, agent and rating."
            ),
        },
    ),
]


def _complete(example: dict) -> dict:
    """Round-trip an example through QueryPlan so it carries every field.

    The strict schema forces every property into `required`, so a filter must
    show all of field/op/value/values. Hand-written examples drifted from that
    and the model copied them, producing filters with no "values" key that Groq
    rejected after generation. Deriving the examples from the model itself means
    they cannot drift again: add a field to QueryPlan and every example gains it.

    Validating here also turns a malformed example into an import-time error
    instead of a silent prompt bug.
    """
    return QueryPlan.model_validate(example).model_dump(
        mode="json", by_alias=True, exclude_none=False
    )


def build_planner_messages(question: str, schema_description: str) -> tuple[str, str]:
    """Build the (system, user) pair for plan generation."""
    lines = ["Worked examples:", ""]
    for q, plan in _EXAMPLES:
        lines.append(f"Q: {q}")
        lines.append(f"A: {json.dumps(_complete(plan), separators=(',', ':'))}")
        lines.append("")
    lines.append(f"Now convert this question:\nQ: {question}")
    return PLANNER_SYSTEM.format(schema=schema_description), "\n".join(lines)


def build_repair_messages(
    question: str, schema_description: str, bad_output: str, error: str
) -> tuple[str, str]:
    """Ask the model to fix a plan that failed validation."""
    system = PLANNER_SYSTEM.format(schema=schema_description)
    user = (
        f"Your previous plan for this question was rejected by the validator.\n\n"
        f"Question: {question}\n\n"
        f"Your plan:\n{bad_output[:1500]}\n\n"
        f"Validation error:\n{error[:800]}\n\n"
        "Return a corrected plan. Change only what the error requires."
    )
    return system, user


NARRATOR_SYSTEM = """\
You state the result of a completed data query in plain English.

The numbers below were computed by a deterministic engine and are final. Report \
them exactly as given. Never recalculate, round differently, estimate, or add \
figures that are not present.

Write one or two short sentences. Lead with the answer. If a note mentions \
excluded nulls, a tie, or truncation, mention it briefly. No preamble, no \
bullet points, no restating the question.
"""


def build_narrator_messages(question: str, plan_summary: str, result: dict) -> tuple[str, str]:
    """Build the (system, user) pair for phrasing a computed result."""
    payload = {
        "question": question,
        "plan": plan_summary,
        "result": result,
    }
    return NARRATOR_SYSTEM, json.dumps(payload, separators=(",", ":"), default=str)[:4000]
