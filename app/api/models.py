"""API request and response schemas.

Kept separate from the internal QueryPlan models so the wire contract can evolve
without loosening the validation that protects the executor.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=500,
        description="Natural language question about the tickets.",
        examples=["How many tickets are currently open?"],
    )


class QueryResponseModel(BaseModel):
    question: str
    answer: str = Field(description="Plain-language statement of the result.")
    plan: dict = Field(description="The validated plan that was executed.")
    plan_summary: str
    result: dict
    anomalies: dict | None = None
    meta: dict


class AnomalyRequest(BaseModel):
    types: list[
        Literal[
            "all",
            "resolution_outlier",
            "aging_unresolved",
            "response_sla_breach",
            "low_rating",
            "data_integrity",
        ]
    ] = Field(default_factory=lambda: ["all"])
    window: Literal[
        "all_time",
        "today",
        "yesterday",
        "last_7_days",
        "last_30_days",
        "this_week",
        "this_month",
        "last_month",
    ] = "all_time"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    dataset: dict
    ingest: dict
    llm: dict
    config: dict


class ErrorResponse(BaseModel):
    error: str
    stage: str | None = None
    detail: str | None = None
    hint: str | None = None


class SchemaResponse(BaseModel):
    columns: list[dict[str, Any]]
    enums: dict[str, list[str]]
    reference_time: str
    reference_mode: str
    example_questions: list[str]
