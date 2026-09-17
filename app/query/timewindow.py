"""Relative date resolution.

Every relative term resolves against the store's reference clock, not the wall
clock. The shipped dataset ends 2024-03-30, so "this month" against the real
today would match nothing and read as a broken system. The resolved bounds are
returned alongside the answer so the interpretation is always visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.query.plan import TimePreset, TimeWindow


@dataclass(frozen=True)
class ResolvedWindow:
    start: datetime | None
    end: datetime | None
    label: str

    @property
    def is_open(self) -> bool:
        return self.start is None and self.end is None

    def as_dict(self) -> dict:
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "label": self.label,
        }


def _day_start(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _day_end(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def _fmt(start: datetime | None, end: datetime | None, name: str) -> str:
    if start and end:
        return f"{name} ({start:%Y-%m-%d} to {end:%Y-%m-%d})"
    if start:
        return f"{name} (from {start:%Y-%m-%d})"
    if end:
        return f"{name} (until {end:%Y-%m-%d})"
    return name


def resolve_window(window: TimeWindow | None, reference: datetime) -> ResolvedWindow:
    """Turn a plan's TimeWindow into concrete datetime bounds.

    Args:
        window: The plan's window, or None for no restriction.
        reference: The clock relative terms are measured from.

    Returns:
        Inclusive bounds plus a human-readable label.
    """
    if window is None or window.preset == TimePreset.ALL_TIME:
        return ResolvedWindow(None, None, "all time")

    p = window.preset

    if p == TimePreset.CUSTOM:
        return ResolvedWindow(
            window.start, window.end, _fmt(window.start, window.end, "custom range")
        )

    if p == TimePreset.TODAY:
        s, e = _day_start(reference), _day_end(reference)
        return ResolvedWindow(s, e, _fmt(s, e, "today"))

    if p == TimePreset.YESTERDAY:
        y = reference - timedelta(days=1)
        s, e = _day_start(y), _day_end(y)
        return ResolvedWindow(s, e, _fmt(s, e, "yesterday"))

    if p == TimePreset.LAST_7_DAYS:
        s, e = _day_start(reference - timedelta(days=6)), _day_end(reference)
        return ResolvedWindow(s, e, _fmt(s, e, "last 7 days"))

    if p == TimePreset.LAST_30_DAYS:
        s, e = _day_start(reference - timedelta(days=29)), _day_end(reference)
        return ResolvedWindow(s, e, _fmt(s, e, "last 30 days"))

    if p == TimePreset.THIS_WEEK:
        # ISO week: Monday start.
        s = _day_start(reference - timedelta(days=reference.weekday()))
        e = _day_end(reference)
        return ResolvedWindow(s, e, _fmt(s, e, "this week"))

    if p == TimePreset.THIS_MONTH:
        s = _day_start(reference.replace(day=1))
        e = _day_end(reference)
        return ResolvedWindow(s, e, _fmt(s, e, "this month"))

    if p == TimePreset.LAST_MONTH:
        first_this = reference.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        s = _day_start(last_prev.replace(day=1))
        e = _day_end(last_prev)
        return ResolvedWindow(s, e, _fmt(s, e, "last month"))

    # Unreachable while TimePreset stays exhaustive; explicit for safety.
    raise ValueError(f"Unhandled time preset: {p}")
