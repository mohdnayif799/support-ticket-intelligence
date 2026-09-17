"""CSV ingestion, validation and normalisation.

Design intent: never let malformed input reach the query engine, and never fail
silently. Rows that cannot be coerced are quarantined with a reason rather than
dropped, so the evaluator can see exactly what was rejected and why.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.domain import schema as S

logger = logging.getLogger(__name__)


class IngestError(RuntimeError):
    """Raised when the file cannot be loaded at all."""


@dataclass
class RejectedRow:
    row_number: int
    ticket_id: str | None
    reason: str


@dataclass
class IngestReport:
    """What happened during load. Surfaced through /api/health."""

    source: str
    rows_read: int = 0
    rows_accepted: int = 0
    rejected: list[RejectedRow] = field(default_factory=list)
    renamed_columns: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def rows_rejected(self) -> int:
        return len(self.rejected)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "rows_read": self.rows_read,
            "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "renamed_columns": self.renamed_columns,
            "warnings": self.warnings,
            "rejected_sample": [
                {"row": r.row_number, "ticket_id": r.ticket_id, "reason": r.reason}
                for r in self.rejected[:20]
            ],
        }


def _normalise_headers(df: pd.DataFrame, report: IngestReport) -> pd.DataFrame:
    """Map raw headers onto canonical names, recording every rename."""
    mapping: dict[str, str] = {}
    unknown: list[str] = []
    for raw in df.columns:
        canon = S.canonical_column(str(raw))
        if canon is None:
            unknown.append(str(raw))
        elif canon != raw:
            mapping[str(raw)] = canon

    if mapping:
        df = df.rename(columns=mapping)
        report.renamed_columns = mapping
        logger.info("Renamed columns to canonical names: %s", mapping)

    if unknown:
        report.warnings.append(f"Ignored unrecognised columns: {sorted(unknown)}")
        df = df.drop(columns=unknown)

    missing = S.REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise IngestError(
            f"CSV is missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )

    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        raise IngestError(f"CSV has duplicate columns after normalisation: {dupes}")

    return df[list(S.ALL_COLUMNS)]


def _coerce_and_validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[RejectedRow]]:
    """Coerce dtypes and reject rows that violate the schema contract."""
    rejected: list[RejectedRow] = []
    reject_idx: set[int] = set()

    def reject(idx, reason: str) -> None:
        for i in idx:
            if i in reject_idx:
                continue
            reject_idx.add(i)
            tid = df.at[i, S.TICKET_ID] if S.TICKET_ID in df.columns else None
            rejected.append(
                RejectedRow(
                    row_number=int(i) + 2,  # +2: 1-indexed plus header row
                    ticket_id=None if pd.isna(tid) else str(tid),
                    reason=reason,
                )
            )

    # Strings: strip whitespace so " High " matches "High".
    for col in (S.TICKET_ID, S.CATEGORY, S.PRIORITY, S.STATUS, S.AGENT_ID, S.ISSUE_SUMMARY):
        df[col] = df[col].astype("string").str.strip()

    # Identity must exist and be unique.
    reject(df.index[df[S.TICKET_ID].isna() | (df[S.TICKET_ID] == "")], "missing ticket_id")
    dupe_mask = df[S.TICKET_ID].duplicated(keep="first") & df[S.TICKET_ID].notna()
    reject(df.index[dupe_mask], "duplicate ticket_id")

    # Timestamps.
    df[S.CREATED_AT] = pd.to_datetime(df[S.CREATED_AT], errors="coerce")
    reject(df.index[df[S.CREATED_AT].isna()], "unparseable created_at")

    # Controlled vocabularies. Unknown values are rejected rather than coerced:
    # silently bucketing an unexpected priority would corrupt every aggregate.
    for col, enum_cls in (
        (S.CATEGORY, S.Category),
        (S.PRIORITY, S.Priority),
        (S.STATUS, S.Status),
    ):
        allowed = {m.value for m in enum_cls}
        bad = df[col].notna() & ~df[col].isin(allowed)
        reject(df.index[bad], f"invalid {col} (expected one of {sorted(allowed)})")
        reject(df.index[df[col].isna()], f"missing {col}")

    # Numerics.
    for col in (S.RESPONSE_TIME, S.RESOLUTION_TIME, S.CUSTOMER_RATING):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # response_time is mandatory; the other two are legitimately null when a
    # ticket is unresolved.
    reject(df.index[df[S.RESPONSE_TIME].isna()], "missing response_time_hrs")
    for col in (S.RESPONSE_TIME, S.RESOLUTION_TIME):
        reject(df.index[df[col].notna() & (df[col] < 0)], f"negative {col}")

    rating = df[S.CUSTOMER_RATING]
    out_of_range = rating.notna() & ((rating < S.RATING_MIN) | (rating > S.RATING_MAX))
    reject(df.index[out_of_range], f"customer_rating outside {S.RATING_MIN}-{S.RATING_MAX}")

    clean = df.drop(index=sorted(reject_idx)).reset_index(drop=True)
    return clean, rejected


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns the engine needs that are not in the source file."""
    df["is_resolved"] = df[S.STATUS] == S.Status.RESOLVED.value
    df["is_unresolved"] = df[S.STATUS].isin(S.UNRESOLVED_STATUSES)
    df[S.PRIORITY] = pd.Categorical(
        df[S.PRIORITY], categories=list(S.PRIORITY_ORDER), ordered=True
    )
    return df


def _consistency_warnings(df: pd.DataFrame, report: IngestReport) -> None:
    """Flag contract violations that are worth reporting but not fatal.

    These rows stay in the dataset: they are real observations and excluding
    them would silently change every aggregate. The integrity anomaly detector
    surfaces them to the user instead.
    """
    resolved_no_time = (df[S.STATUS] == S.Status.RESOLVED.value) & df[
        S.RESOLUTION_TIME
    ].isna()
    if n := int(resolved_no_time.sum()):
        report.warnings.append(f"{n} resolved tickets have no resolution_time_hrs")

    unresolved_with_time = df["is_unresolved"] & df[S.RESOLUTION_TIME].notna()
    if n := int(unresolved_with_time.sum()):
        report.warnings.append(f"{n} unresolved tickets have a resolution_time_hrs")

    impossible = df[S.RESOLUTION_TIME].notna() & (
        df[S.RESOLUTION_TIME] < df[S.RESPONSE_TIME]
    )
    if n := int(impossible.sum()):
        report.warnings.append(
            f"{n} tickets resolve before first response "
            "(surfaced by the data_integrity detector)"
        )


def load_tickets(
    csv_path: str | Path, strict: bool = False
) -> tuple[pd.DataFrame, IngestReport]:
    """Load, validate and normalise the ticket CSV.

    Args:
        csv_path: Path to the CSV file.
        strict: If True, raise when any row is rejected.

    Returns:
        A validated DataFrame and a report describing what was accepted.

    Raises:
        IngestError: File missing, unreadable, empty, or missing columns. Also
            raised on any rejected row when ``strict`` is True.
    """
    path = Path(csv_path)
    if not path.exists():
        raise IngestError(
            f"Dataset not found at {path}. Place support_tickets.csv there or set "
            "CSV_PATH in your environment."
        )

    report = IngestReport(source=str(path))

    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=True, skipinitialspace=True)
    except pd.errors.EmptyDataError as exc:
        raise IngestError(f"{path} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise IngestError(f"{path} is not valid CSV: {exc}") from exc

    report.rows_read = len(raw)
    if report.rows_read == 0:
        raise IngestError(f"{path} contains a header but no data rows.")

    df = _normalise_headers(raw, report)
    df, rejected = _coerce_and_validate(df)
    report.rejected = rejected

    if rejected and strict:
        detail = "; ".join(f"row {r.row_number}: {r.reason}" for r in rejected[:10])
        raise IngestError(f"{len(rejected)} invalid row(s) with strict ingest on: {detail}")

    if df.empty:
        raise IngestError(
            f"Every row in {path} was rejected. First reason: "
            f"{rejected[0].reason if rejected else 'unknown'}"
        )

    df = _add_derived_columns(df)
    report.rows_accepted = len(df)
    _consistency_warnings(df, report)

    logger.info(
        "Loaded %d/%d rows from %s (%d rejected)",
        report.rows_accepted,
        report.rows_read,
        path,
        report.rows_rejected,
    )
    return df, report
