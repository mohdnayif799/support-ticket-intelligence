"""Application configuration.

All tunables live here and can be overridden by environment variables or a .env
file. Nothing secret is hardcoded; GROQ_API_KEY is read from the environment.

Threshold defaults are not guesses. They were calibrated against the shipped
500-row dataset; see docstrings on each field for the measured effect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Data ---------------------------------------------------------------
    csv_path: Path = Field(
        default=PROJECT_ROOT / "data" / "support_tickets.csv",
        description="Path to the support ticket CSV.",
    )

    strict_ingest: bool = Field(
        default=False,
        description=(
            "If True, any unparseable row aborts startup. If False (default), bad "
            "rows are quarantined and reported via /api/health."
        ),
    )

    # --- Reference clock ----------------------------------------------------
    # The dataset spans 2024-01-01 to 2024-03-30. Resolving "this month" against
    # the real wall clock would make every relative-date query return zero rows,
    # which looks like a bug. Anchoring to the newest ticket keeps those queries
    # meaningful. Switch to "system" for live data.
    reference_clock: Literal["dataset_max", "system"] = "dataset_max"

    # --- LLM ----------------------------------------------------------------
    llm_provider: Literal["groq", "ollama", "stub"] = "groq"

    groq_api_key: str | None = None
    groq_model: str = Field(
        default="openai/gpt-oss-20b",
        description=(
            "Must support Groq Structured Outputs strict mode. As of 2026-09: "
            "openai/gpt-oss-20b, openai/gpt-oss-120b, qwen/qwen3.8-27b. Note that "
            "llama-3.3-70b-versatile and llama-3.1-8b-instant were deprecated "
            "2026-06-17 and will fail."
        ),
    )
    groq_strict_schema: bool = Field(
        default=True,
        description="Use constrained decoding. Disable only for a model that lacks it.",
    )

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"

    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = Field(
        default=2, description="Repair attempts after a schema-invalid plan."
    )
    llm_temperature: float = 0.0
    narrate_answers: bool = Field(
        default=True,
        description=(
            "Second LLM call to phrase the computed result in prose. Numbers always "
            "come from Python; if this is off or fails, a deterministic template is "
            "used instead."
        ),
    )

    # --- Anomaly thresholds -------------------------------------------------
    # Resolution-time outliers are grouped by PRIORITY, not category. Measured on
    # the shipped data: category medians are 11.4 / 12.1 / 13.2 hrs (no signal),
    # while priority medians are 3.7 / 6.9 / 14.5 / 26.6 hrs. A single global
    # fence flags 21 tickets but is dominated by Low-priority tickets that are
    # normal for their class, while missing Critical tickets running 5x their
    # class median. Per-priority flags 18 with far better semantics.
    outlier_group_by: Literal["priority", "category", "none"] = "priority"
    outlier_method: Literal["iqr", "mad"] = "iqr"
    iqr_multiplier: float = Field(default=1.5, ge=0.0)
    mad_z_threshold: float = Field(default=3.5, ge=0.0)
    outlier_min_group_size: int = Field(
        default=8, ge=2, description="Groups smaller than this are skipped as unreliable."
    )

    # Aging unresolved tickets. 24h is the figure named in the brief.
    aging_hours: float = Field(default=24.0, ge=0.0)

    # First-response SLA in hours, per priority. A statistical detector is useless
    # here: response_time_hrs is uniform on [0.2, 5.0], so its IQR fence sits at
    # 7.65 and never fires. These fixed targets flag 61 tickets (34 Critical,
    # 27 High) on the shipped data.
    response_sla_hours: dict[str, float] = Field(
        default_factory=lambda: {"Critical": 2.0, "High": 4.0, "Medium": 6.0, "Low": 12.0}
    )

    low_rating_threshold: int = Field(
        default=2, ge=1, le=5, description="Ratings at or below this are flagged."
    )

    max_anomalies_per_detector: int = Field(
        default=50,
        ge=1,
        description=(
            "Caps rows returned per detector. The aging rule matches 80 tickets "
            "because the data spans three months with no backfilled resolutions; "
            "returning all of them buries the signal. Counts are always exact."
        ),
    )

    # --- API ----------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    max_query_length: int = Field(default=500, ge=1)
    default_row_limit: int = Field(default=20, ge=1)
    max_row_limit: int = Field(default=200, ge=1)
    log_level: str = "INFO"

    @field_validator("response_sla_hours")
    @classmethod
    def _validate_sla(cls, v: dict[str, float]) -> dict[str, float]:
        from app.domain.schema import PRIORITY_ORDER

        unknown = set(v) - set(PRIORITY_ORDER)
        if unknown:
            raise ValueError(f"Unknown priority in response_sla_hours: {sorted(unknown)}")
        return v


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Clear the cache. Used by tests that override the environment."""
    global _settings
    _settings = None
