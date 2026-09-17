"""The query plan DSL.

This is the contract between the LLM and the data, and the system's security
boundary. The model never emits SQL, pandas expressions or code of any kind. It
emits a QueryPlan: a closed vocabulary of field names, operators and enum values.

Anything outside the vocabulary fails Pydantic validation and never reaches the
executor. That makes injection structurally impossible rather than filtered: there
is no code path from model output to code execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain import schema as S

#: Default row cap when the model expresses no preference.
DEFAULT_LIMIT = 20


class Intent(str, Enum):
    """What the user wants back."""

    AGGREGATE = "aggregate"  # one number, or one number per group
    LIST = "list"  # matching ticket rows
    ANOMALY = "anomaly"  # hand off to the anomaly detectors
    UNSUPPORTED = "unsupported"  # question cannot be expressed in this DSL


class FilterField(str, Enum):
    TICKET_ID = S.TICKET_ID
    CREATED_AT = S.CREATED_AT
    CATEGORY = S.CATEGORY
    PRIORITY = S.PRIORITY
    STATUS = S.STATUS
    RESPONSE_TIME = S.RESPONSE_TIME
    RESOLUTION_TIME = S.RESOLUTION_TIME
    AGENT_ID = S.AGENT_ID
    CUSTOMER_RATING = S.CUSTOMER_RATING
    ISSUE_SUMMARY = S.ISSUE_SUMMARY


class MetricField(str, Enum):
    """Only numeric columns can be aggregated."""

    RESPONSE_TIME = S.RESPONSE_TIME
    RESOLUTION_TIME = S.RESOLUTION_TIME
    CUSTOMER_RATING = S.CUSTOMER_RATING


class GroupField(str, Enum):
    CATEGORY = S.CATEGORY
    PRIORITY = S.PRIORITY
    STATUS = S.STATUS
    AGENT_ID = S.AGENT_ID


class Operator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"
    CONTAINS = "contains"


class MetricOp(str, Enum):
    COUNT = "count"
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


class TimePreset(str, Enum):
    """Relative windows, resolved against the store's reference clock."""

    ALL_TIME = "all_time"
    TODAY = "today"
    YESTERDAY = "yesterday"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    THIS_WEEK = "this_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    CUSTOM = "custom"


class SortDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


class AnomalyType(str, Enum):
    ALL = "all"
    RESOLUTION_OUTLIER = "resolution_outlier"
    AGING_UNRESOLVED = "aging_unresolved"
    RESPONSE_SLA_BREACH = "response_sla_breach"
    LOW_RATING = "low_rating"
    DATA_INTEGRITY = "data_integrity"


# --- Plan components --------------------------------------------------------

#: Scalars a filter may compare against. Deliberately narrow.
FilterValue = str | float | int | bool | None


class Filter(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    field_name: FilterField = Field(alias="field", description="Column to filter on.")
    op: Operator
    value: FilterValue = Field(
        default=None, description="Comparison value. Null for is_null/not_null."
    )
    values: list[str] | None = Field(
        default=None, description="Value list for the in/not_in operators."
    )

    @model_validator(mode="after")
    def _check_operand_shape(self) -> Filter:
        if self.op in (Operator.IN, Operator.NOT_IN):
            if not self.values:
                raise ValueError(f"operator '{self.op.value}' requires a non-empty 'values'")
        elif self.op in (Operator.IS_NULL, Operator.NOT_NULL):
            if self.value is not None or self.values:
                raise ValueError(f"operator '{self.op.value}' takes no operand")
        else:
            if self.value is None:
                raise ValueError(f"operator '{self.op.value}' requires a 'value'")

        numeric_ops = {Operator.GT, Operator.GTE, Operator.LT, Operator.LTE}
        if self.op in numeric_ops and not isinstance(self.value, (int, float)):
            raise ValueError(
                f"operator '{self.op.value}' needs a number, got {self.value!r}"
            )

        if self.op == Operator.CONTAINS and self.field_name != FilterField.ISSUE_SUMMARY:
            raise ValueError("'contains' is only valid on issue_summary")

        return self


class TimeWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: TimePreset = TimePreset.ALL_TIME
    start: datetime | None = None
    end: datetime | None = None

    @field_validator("preset", mode="before")
    @classmethod
    def _default_preset(cls, v: Any) -> Any:
        return TimePreset.ALL_TIME if v is None else v

    @model_validator(mode="after")
    def _check_custom_bounds(self) -> TimeWindow:
        if self.preset == TimePreset.CUSTOM and self.start is None and self.end is None:
            raise ValueError("preset 'custom' requires start and/or end")
        if self.start and self.end and self.start > self.end:
            raise ValueError("time window start is after end")
        return self


class Aggregation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    op: MetricOp
    field_name: MetricField | None = Field(
        default=None, alias="field", description="Null only when op is 'count'."
    )

    @model_validator(mode="after")
    def _check_field_presence(self) -> Aggregation:
        if self.op == MetricOp.COUNT:
            return self
        if self.field_name is None:
            raise ValueError(f"aggregation '{self.op.value}' requires a field")
        return self


class Sort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by: str = Field(
        description="A group_by column, a plain column, or 'metric' for the aggregate."
    )
    direction: SortDirection = SortDirection.DESC

    @field_validator("direction", mode="before")
    @classmethod
    def _default_direction(cls, v: Any) -> Any:
        return SortDirection.DESC if v is None else v


class QueryPlan(BaseModel):
    """A validated, executable description of what to compute."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    intent: Intent
    filters: list[Filter] = Field(
        default_factory=list, max_length=10, description="Combined with AND."
    )
    filters_any: list[Filter] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "Combined with OR, then ANDed with 'filters'. Needed for questions like "
            "'Critical tickets not resolved within 12 hours', which means resolution "
            "took over 12h OR the ticket is still unresolved."
        ),
    )
    time_window: TimeWindow | None = None
    aggregation: Aggregation | None = None
    group_by: list[GroupField] = Field(default_factory=list, max_length=2)
    sort: Sort | None = None
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=500)
    anomaly_types: list[AnomalyType] = Field(default_factory=list, max_length=6)
    reasoning: str = Field(
        default="",
        max_length=400,
        description="One line on how the question maps to this plan.",
    )
    unsupported_reason: str = Field(
        default="", max_length=400, description="Why the question cannot be answered."
    )

    @field_validator("limit", mode="before")
    @classmethod
    def _default_limit(cls, v: Any) -> Any:
        """Strict mode makes the model send null when it has no preference."""
        return DEFAULT_LIMIT if v is None else v

    @field_validator("reasoning", "unsupported_reason", mode="before")
    @classmethod
    def _none_to_empty(cls, v: Any) -> Any:
        return "" if v is None else v

    @field_validator("filters", "filters_any", "group_by", "anomaly_types", mode="before")
    @classmethod
    def _none_to_list(cls, v: Any) -> Any:
        return [] if v is None else v

    @model_validator(mode="after")
    def _check_intent_coherence(self) -> QueryPlan:
        if self.intent == Intent.AGGREGATE and self.aggregation is None:
            raise ValueError("intent 'aggregate' requires an aggregation")
        if self.intent == Intent.UNSUPPORTED and not self.unsupported_reason:
            raise ValueError("intent 'unsupported' requires unsupported_reason")
        if self.intent != Intent.AGGREGATE and self.group_by:
            raise ValueError("group_by is only valid with intent 'aggregate'")
        if self.intent == Intent.ANOMALY and not self.anomaly_types:
            self.anomaly_types = [AnomalyType.ALL]
        return self

    def describe(self) -> str:
        """Human-readable one-liner, shown in the UI beside the answer."""
        bits: list[str] = [f"intent={self.intent.value}"]
        if self.aggregation:
            target = self.aggregation.field_name.value if self.aggregation.field_name else "*"
            bits.append(f"{self.aggregation.op.value}({target})")
        if self.filters:
            bits.append(
                "where " + " and ".join(f"{f.field_name.value} {f.op.value}" for f in self.filters)
            )
        if self.filters_any:
            joined = " or ".join(f"{f.field_name.value} {f.op.value}" for f in self.filters_any)
            bits.append(("and (" if self.filters else "where (") + joined + ")")
        if self.group_by:
            bits.append("by " + ", ".join(g.value for g in self.group_by))
        if self.time_window and self.time_window.preset != TimePreset.ALL_TIME:
            bits.append(f"window={self.time_window.preset.value}")
        return " ".join(bits)


# --- JSON schema generation -------------------------------------------------

#: Validation keywords that constrained-decoding backends commonly reject.
#: Dropping them from the wire schema costs nothing: every plan is re-validated
#: by Pydantic on arrival, so these bounds are still enforced, just on our side
#: instead of the decoder's.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "default",
        "maxLength",
        "minLength",
        "pattern",
        "maxItems",
        "minItems",
        "uniqueItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    }
)


def _inline_refs(node: Any, defs: dict) -> Any:
    """Replace every $ref with the definition it points at.

    Groq validates each anyOf branch without resolving $ref first, so a branch
    that is only a reference cannot be told apart from its siblings. Inlining
    removes the indirection. Safe here because the plan schema has no recursion.
    """
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            if name not in defs:
                raise ValueError(f"Unresolvable $ref in plan schema: {node['$ref']}")
            resolved = copy.deepcopy(defs[name])
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            resolved.update(siblings)  # keep description/title from the property
            return _inline_refs(resolved, defs)
        if "allOf" in node and len(node["allOf"]) == 1:
            merged = _inline_refs(node["allOf"][0], defs)
            merged.update({k: v for k, v in node.items() if k != "allOf"})
            return merged
        return {k: _inline_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(item, defs) for item in node]
    return node


def _normalise_type_value(types: Any) -> Any:
    """Clean up a JSON-schema `type` value.

    `integer` is a strict subset of `number`, so a type list carrying both is
    redundant. Groq rejects it outright: "cannot include both 'integer' and
    'number' in 'type'". Dropping `integer` widens nothing in practice, and
    Pydantic still parses a whole number back into an int on arrival.

    Also dedupes and unwraps a single-element list back to a plain string.
    """
    if not isinstance(types, list):
        return types
    ordered = list(dict.fromkeys(types))
    if "number" in ordered and "integer" in ordered:
        ordered = [t for t in ordered if t != "integer"]
    return ordered[0] if len(ordered) == 1 else ordered


def _normalise_types(node: Any) -> Any:
    """Apply type normalisation to every schema node in the tree."""
    if isinstance(node, dict):
        node = {k: _normalise_types(v) for k, v in node.items()}
        if "type" in node:
            node["type"] = _normalise_type_value(node["type"])
        return node
    if isinstance(node, list):
        return [_normalise_types(item) for item in node]
    return node


def _merge_union(branches: list[dict]) -> dict:
    """Fold an anyOf into a single schema using a type list.

    A union of {X, null} becomes X with "null" added to its type, and a union of
    several scalars becomes one schema with a list of types. Either way no anyOf
    survives, so there is nothing left for the backend to disambiguate.
    """
    nullable = any(b.get("type") == "null" for b in branches)
    real = [b for b in branches if b.get("type") != "null"]

    if not real:
        return {"type": "null"}

    if len(real) == 1:
        merged = dict(real[0])
    else:
        if any("properties" in b for b in real):
            # Two object shapes in one union genuinely cannot be collapsed. The
            # plan DSL has no such field; fail loudly if one is ever added.
            raise ValueError("Cannot collapse a union of object schemas")
        types: list[str] = []
        for branch in real:
            t = branch.get("type")
            if isinstance(t, list):
                types.extend(t)
            elif t:
                types.append(t)
        merged = {"type": _normalise_type_value(types)}

    if nullable:
        current = merged.get("type")
        if isinstance(current, str):
            merged["type"] = _normalise_type_value([current, "null"])
        elif isinstance(current, list) and "null" not in current:
            merged["type"] = _normalise_type_value([*current, "null"])
        # A nullable enum must list null as a permitted value, otherwise the
        # enum contradicts the type and null can never be emitted.
        if "enum" in merged and None not in merged["enum"]:
            merged["enum"] = [*merged["enum"], None]

    return merged


def _collapse_unions(node: Any) -> Any:
    """Remove every anyOf in the tree, depth first."""
    if isinstance(node, dict):
        node = {k: _collapse_unions(v) for k, v in node.items()}
        if "anyOf" in node:
            branches = node.pop("anyOf")
            merged = _merge_union(branches)
            merged.update(node)  # preserve description and friends
            return merged
        return node
    if isinstance(node, list):
        return [_collapse_unions(item) for item in node]
    return node


def _widen_to_nullable(prop: dict) -> dict:
    """Allow null on a property that strict mode forces into `required`.

    Strict mode has no optional properties: every key must appear in `required`
    and the model must emit a value for each. The documented way to express
    "not applicable" is a union with null, so a field that Pydantic treated as
    optional has to be widened here or the model's only legal answer, null,
    fails validation against its own declared type.
    """
    prop = dict(prop)
    types = prop.get("type")
    if types is None:  # untyped already accepts null
        return prop
    if isinstance(types, str):
        if types != "null":
            prop["type"] = _normalise_type_value([types, "null"])
    elif isinstance(types, list) and "null" not in types:
        prop["type"] = _normalise_type_value([*types, "null"])
    if "enum" in prop and None not in prop["enum"]:
        prop["enum"] = [*prop["enum"], None]
    return prop


def _close_objects(node: Any) -> Any:
    """Mark every property required, close every object, drop unsupported keys.

    Strict mode requires the full property list in `required` and
    `additionalProperties: false` on each object. Pydantic omits defaulted
    fields from `required`, so they are added back here and simultaneously
    widened to accept null, which is how strict mode expresses optionality.
    Adding to `required` without widening is a bug: it leaves the model no
    legal way to skip the field.
    """
    if isinstance(node, dict):
        node = {
            k: _close_objects(v) for k, v in node.items() if k not in _UNSUPPORTED_KEYWORDS
        }
        types = node.get("type")
        is_object = types == "object" or (isinstance(types, list) and "object" in types)
        if is_object and "properties" in node:
            # Anything Pydantic left out of `required` has a default, so the
            # model must be able to answer null for it once we force it in.
            optional = set(node["properties"]) - set(node.get("required", []))
            for name in optional:
                node["properties"][name] = _widen_to_nullable(node["properties"][name])
            node["required"] = list(node["properties"].keys())
            node["additionalProperties"] = False
        return node
    if isinstance(node, list):
        return [_close_objects(item) for item in node]
    return node


def validate_strict_schema(schema: Any) -> list[str]:
    """Return every construct Groq's structured-output validator is known to reject.

    Each entry here corresponds to a 400 we actually received:
      - bare $ref branches inside anyOf  ("must be disambiguated via a
        required discriminator")
      - any anyOf at all, once refs are inlined
      - a type list holding both integer and number ("cannot include both
        'integer' and 'number' in 'type'")
      - a nullable enum that does not list null, which would contradict itself
      - an object that is not closed or not fully required
    """
    problems: list[str] = []

    def walk(node: Any, path: str = "#") -> None:
        if isinstance(node, dict):
            if "anyOf" in node:
                problems.append(f"{path}: anyOf is not supported in the strict dialect")
            if "$ref" in node:
                problems.append(f"{path}: unresolved $ref")
            types = node.get("type")
            if isinstance(types, list):
                if {"integer", "number"} <= set(types):
                    problems.append(f"{path}: type cannot include both 'integer' and 'number'")
                if "null" in types and "enum" in node and None not in node["enum"]:
                    problems.append(f"{path}: nullable enum does not permit null")
            is_object = types == "object" or (isinstance(types, list) and "object" in types)
            if is_object and "properties" in node:
                if set(node.get("required", [])) != set(node["properties"]):
                    problems.append(f"{path}: required does not cover every property")
                if node.get("additionalProperties") is not False:
                    problems.append(f"{path}: object is not closed")
            for key, value in node.items():
                walk(value, f"{path}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}/{index}")

    walk(schema)
    return problems


def schema_fingerprint(schema: Any) -> str:
    """Short stable hash, logged at startup so the loaded build is identifiable."""
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def plan_json_schema(strict: bool = True) -> dict:
    """JSON schema for QueryPlan.

    Args:
        strict: True produces the Groq strict-mode dialect: no $ref, no anyOf,
            every object closed and fully required. False returns Pydantic's
            schema unchanged, which is what Ollama's grammar compiler expects.
    """
    schema = copy.deepcopy(QueryPlan.model_json_schema(by_alias=True))
    if not strict:
        return schema
    defs = schema.pop("$defs", {})
    schema = _inline_refs(schema, defs)
    schema = _collapse_unions(schema)
    schema = _normalise_types(schema)
    schema = _close_objects(schema)

    # Fail here, loudly and locally, rather than shipping a schema the provider
    # will reject with a 400 that costs a round trip to diagnose.
    problems = validate_strict_schema(schema)
    if problems:
        raise ValueError(
            "Generated strict schema contains constructs Groq rejects:\n  "
            + "\n  ".join(problems)
        )
    return schema
