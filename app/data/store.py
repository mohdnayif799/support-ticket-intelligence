"""In-memory ticket store.

Holds the validated DataFrame plus the metadata the LLM prompt and the API need.
At 500 rows a DataFrame is the right container: it loads in milliseconds, needs
no external service, and keeps the whole system runnable with one command.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.config import Settings
from app.data.loader import IngestReport, load_tickets
from app.domain import schema as S


@dataclass(frozen=True)
class DatasetProfile:
    """Facts about the loaded data, shown in the UI and fed to the prompt."""

    row_count: int
    earliest: datetime
    latest: datetime
    reference_time: datetime
    reference_mode: str
    agents: list[str]
    categories: list[str]
    priorities: list[str]
    statuses: list[str]

    def as_dict(self) -> dict:
        return {
            "row_count": self.row_count,
            "earliest_ticket": self.earliest.isoformat(),
            "latest_ticket": self.latest.isoformat(),
            "reference_time": self.reference_time.isoformat(),
            "reference_mode": self.reference_mode,
            "agents": self.agents,
            "categories": self.categories,
            "priorities": self.priorities,
            "statuses": self.statuses,
        }


class TicketStore:
    """Owns the validated dataset and everything derived from it."""

    def __init__(self, frame: pd.DataFrame, report: IngestReport, settings: Settings):
        self._df = frame
        self._report = report
        self._settings = settings
        self._profile = self._build_profile()

    @classmethod
    def from_csv(cls, settings: Settings, csv_path: str | Path | None = None) -> TicketStore:
        path = csv_path or settings.csv_path
        frame, report = load_tickets(path, strict=settings.strict_ingest)
        return cls(frame, report, settings)

    @property
    def frame(self) -> pd.DataFrame:
        """Defensive copy. Callers filter and mutate freely without corrupting state."""
        return self._df.copy()

    @property
    def report(self) -> IngestReport:
        return self._report

    @property
    def profile(self) -> DatasetProfile:
        return self._profile

    @property
    def reference_time(self) -> datetime:
        return self._profile.reference_time

    def _build_profile(self) -> DatasetProfile:
        created = self._df[S.CREATED_AT]
        earliest, latest = created.min(), created.max()

        if self._settings.reference_clock == "system":
            reference = datetime.now()
            mode = "system clock"
        else:
            reference = latest.to_pydatetime()
            mode = "newest ticket in dataset"

        return DatasetProfile(
            row_count=len(self._df),
            earliest=earliest.to_pydatetime(),
            latest=latest.to_pydatetime(),
            reference_time=reference,
            reference_mode=mode,
            agents=sorted(self._df[S.AGENT_ID].dropna().unique().tolist()),
            categories=sorted(self._df[S.CATEGORY].dropna().unique().tolist()),
            priorities=[
                p for p in S.PRIORITY_ORDER if p in set(self._df[S.PRIORITY].dropna())
            ],
            statuses=sorted(self._df[S.STATUS].dropna().unique().tolist()),
        )

    def schema_description(self) -> str:
        """Compact schema summary injected into the LLM prompt.

        Real enum values are listed so the model matches on actual data rather
        than inventing plausible-looking categories.
        """
        p = self._profile
        return "\n".join(
            [
                "Table: support_tickets",
                f"Rows: {p.row_count}",
                f"Date range: {p.earliest:%Y-%m-%d} to {p.latest:%Y-%m-%d}",
                (
                    f'Reference date for relative terms like "today"/"this month": '
                    f"{p.reference_time:%Y-%m-%d %H:%M} ({p.reference_mode})"
                ),
                "",
                "Columns:",
                f"  {S.TICKET_ID} (string, unique)",
                f"  {S.CREATED_AT} (datetime)",
                f"  {S.CATEGORY} (enum: {', '.join(p.categories)})",
                f"  {S.PRIORITY} (enum: {', '.join(p.priorities)})",
                f"  {S.STATUS} (enum: {', '.join(p.statuses)})",
                f"  {S.RESPONSE_TIME} (float, hours to first response, never null)",
                (
                    f"  {S.RESOLUTION_TIME} (float, hours to resolution, "
                    "null when status is not Resolved)"
                ),
                f"  {S.AGENT_ID} (enum: {', '.join(p.agents)})",
                (
                    f"  {S.CUSTOMER_RATING} (integer 1-5, "
                    "null when status is not Resolved)"
                ),
                f"  {S.ISSUE_SUMMARY} (free text)",
            ]
        )
