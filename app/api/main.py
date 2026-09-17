"""FastAPI application.

Serves the REST API and the minimal UI from one process, so the whole system
starts with a single uvicorn command on a single port. The UI is a plain client
of the documented endpoints, which means demonstrating the UI also demonstrates
the API.

Data and provider are constructed once at startup, not per request: a failure in
either is visible immediately rather than on the first question.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.anomaly.detectors import run_detectors
from app.api.models import (
    AnomalyRequest,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponseModel,
    SchemaResponse,
)
from app.config import Settings, get_settings
from app.data.loader import IngestError
from app.data.store import TicketStore
from app.domain import schema as S
from app.llm.base import LLMError, LLMUnavailable
from app.llm.factory import build_provider
from app.pipeline import PipelineError, QueryPipeline
from app.query.plan import AnomalyType, TimePreset, TimeWindow
from app.query.timewindow import resolve_window

VERSION = "1.0.0"
UI_DIR = Path(__file__).resolve().parent.parent / "ui"

logger = logging.getLogger(__name__)

EXAMPLE_QUESTIONS = [
    "How many tickets are currently open?",
    "Which agent resolved the most tickets this month?",
    "Show me all Critical tickets not resolved within 12 hours.",
    "What is the average customer rating for Technical category tickets?",
    "Are there any anomalies in resolution times this week?",
    "Which agent has the lowest average customer rating?",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and build the provider once, at startup."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    app.state.settings = settings
    app.state.startup_errors = []

    try:
        app.state.store = TicketStore.from_csv(settings)
        logger.info(
            "Dataset ready: %d rows, %s to %s",
            app.state.store.profile.row_count,
            app.state.store.profile.earliest.date(),
            app.state.store.profile.latest.date(),
        )
    except IngestError:
        # Without data there is nothing to serve; fail loudly.
        logger.exception("Dataset failed to load")
        raise

    try:
        app.state.provider = build_provider(settings)
    except LLMUnavailable as exc:
        # The API still starts so /api/health and /api/anomalies work and the
        # error is visible in the UI rather than as a crashed process.
        logger.exception("LLM provider unavailable")
        app.state.provider = None
        app.state.startup_errors.append(str(exc))

    app.state.pipeline = (
        QueryPipeline(app.state.store, app.state.provider, settings)
        if app.state.provider
        else None
    )
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Support Ticket Intelligence",
    version=VERSION,
    description=(
        "Natural language questions and anomaly detection over a customer support "
        "ticket dataset. An LLM interprets the question into a validated query "
        "plan; all arithmetic is deterministic Python."
    ),
    lifespan=lifespan,
)


# --- dependencies -----------------------------------------------------------


def get_store(request: Request) -> TicketStore:
    return request.app.state.store


def get_pipeline(request: Request) -> QueryPipeline:
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "The language model is not configured, so natural "
                "language queries are unavailable.",
                "stage": "startup",
                "detail": "; ".join(request.app.state.startup_errors),
                "hint": "Set GROQ_API_KEY, or set LLM_PROVIDER=ollama to run "
                "locally. /api/anomalies works without a model.",
            },
        )
    return pipeline


def settings_dep(request: Request) -> Settings:
    return request.app.state.settings


# --- error handling ---------------------------------------------------------


@app.exception_handler(PipelineError)
async def _pipeline_error(_: Request, exc: PipelineError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error=exc.message, stage=exc.stage, detail=exc.detail or None
        ).model_dump(),
    )


@app.exception_handler(LLMError)
async def _llm_error(_: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="The language model could not be reached.",
            stage="llm",
            detail=str(exc),
            hint="Check GROQ_API_KEY and rate limits, or switch to LLM_PROVIDER=ollama.",
        ).model_dump(),
    )


# --- routes -----------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health(request: Request, store: TicketStore = Depends(get_store)) -> HealthResponse:
    """Liveness plus a full picture of what loaded and what did not."""
    settings: Settings = request.app.state.settings
    provider = request.app.state.provider

    llm_info = {"configured": provider is not None}
    if provider is not None:
        try:
            llm_info.update(provider.health())
        except Exception as exc:  # health must never raise
            llm_info.update({"reachable": False, "detail": str(exc)})
    else:
        llm_info["errors"] = request.app.state.startup_errors

    healthy = provider is not None and llm_info.get("reachable", False)
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=VERSION,
        dataset=store.profile.as_dict(),
        ingest=store.report.as_dict(),
        llm=llm_info,
        config={
            "reference_clock": settings.reference_clock,
            "outlier_method": settings.outlier_method,
            "outlier_group_by": settings.outlier_group_by,
            "aging_hours": settings.aging_hours,
            "response_sla_hours": settings.response_sla_hours,
            "low_rating_threshold": settings.low_rating_threshold,
        },
    )


@app.post("/api/query", response_model=QueryResponseModel, tags=["query"])
def query(
    payload: QueryRequest, pipeline: QueryPipeline = Depends(get_pipeline)
) -> QueryResponseModel:
    """Answer a natural language question about the tickets."""
    response = pipeline.answer(payload.question)
    return QueryResponseModel(**response.as_dict())


@app.post("/api/anomalies", tags=["anomalies"])
def anomalies(
    payload: AnomalyRequest,
    store: TicketStore = Depends(get_store),
    settings: Settings = Depends(settings_dep),
) -> dict:
    """Run the anomaly detectors. Needs no LLM."""
    preset = TimePreset(payload.window)
    window = resolve_window(TimeWindow(preset=preset), store.reference_time)

    frame = store.frame
    if not window.is_open:
        if window.start is not None:
            frame = frame[frame[S.CREATED_AT] >= window.start]
        if window.end is not None:
            frame = frame[frame[S.CREATED_AT] <= window.end]

    types = [AnomalyType(t) for t in payload.types]
    return run_detectors(frame, settings, store.reference_time, types, window)


@app.get("/api/anomalies", tags=["anomalies"])
def anomalies_get(
    store: TicketStore = Depends(get_store), settings: Settings = Depends(settings_dep)
) -> dict:
    """Convenience GET so the endpoint is reachable from a browser or curl."""
    window = resolve_window(TimeWindow(preset=TimePreset.ALL_TIME), store.reference_time)
    return run_detectors(
        store.frame, settings, store.reference_time, [AnomalyType.ALL], window
    )


@app.get("/api/schema", response_model=SchemaResponse, tags=["system"])
def dataset_schema(store: TicketStore = Depends(get_store)) -> SchemaResponse:
    """Describe the queryable surface. Used by the UI to show what can be asked."""
    p = store.profile
    columns = [
        {"name": S.TICKET_ID, "type": "string", "description": "Unique ticket identifier"},
        {"name": S.CREATED_AT, "type": "datetime", "description": "Creation timestamp"},
        {"name": S.CATEGORY, "type": "enum", "description": "Issue category"},
        {"name": S.PRIORITY, "type": "enum", "description": "Urgency level"},
        {"name": S.STATUS, "type": "enum", "description": "Current state"},
        {"name": S.RESPONSE_TIME, "type": "float", "description": "Hours to first response"},
        {
            "name": S.RESOLUTION_TIME,
            "type": "float",
            "description": "Hours to resolution; null when unresolved",
        },
        {"name": S.AGENT_ID, "type": "enum", "description": "Assigned agent"},
        {
            "name": S.CUSTOMER_RATING,
            "type": "integer",
            "description": "1-5 satisfaction; null when unresolved",
        },
        {"name": S.ISSUE_SUMMARY, "type": "text", "description": "Free-text summary"},
    ]
    return SchemaResponse(
        columns=columns,
        enums={
            "category": p.categories,
            "priority": p.priorities,
            "status": p.statuses,
            "agent_id": p.agents,
        },
        reference_time=p.reference_time.isoformat(),
        reference_mode=p.reference_mode,
        example_questions=EXAMPLE_QUESTIONS,
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")
