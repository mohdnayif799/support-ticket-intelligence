"""Deterministic anomaly detection.

No model involvement. Each detector states the rule it applied and the exact
threshold it used, so every flag is reproducible and arguable. Detectors are
scoped by an optional time window so "any anomalies in resolution times this
week?" narrows the population before the statistics are computed.

Thresholds are calibrated against the shipped dataset; see app/config.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from app.config import Settings
from app.domain import schema as S
from app.query.plan import AnomalyType
from app.query.timewindow import ResolvedWindow

logger = logging.getLogger(__name__)

#: Columns shown on a flagged ticket.
_DISPLAY = [
    S.TICKET_ID,
    S.CREATED_AT,
    S.CATEGORY,
    S.PRIORITY,
    S.STATUS,
    S.RESPONSE_TIME,
    S.RESOLUTION_TIME,
    S.AGENT_ID,
    S.CUSTOMER_RATING,
]


@dataclass
class AnomalyGroup:
    """One detector's findings."""

    type: str
    title: str
    severity: str  # high | medium | low
    rule: str
    count: int
    tickets: list[dict] = field(default_factory=list)
    truncated: bool = False
    stats: dict = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "title": self.title,
            "severity": self.severity,
            "rule": self.rule,
            "count": self.count,
            "tickets": self.tickets,
            "truncated": self.truncated,
            "stats": self.stats,
            "note": self.note,
        }


def _rows(
    frame: pd.DataFrame, limit: int, extra: list[str] | None = None
) -> tuple[list[dict], bool]:
    cols = [c for c in _DISPLAY + (extra or []) if c in frame.columns]
    truncated = len(frame) > limit
    out: list[dict] = []
    for rec in frame[cols].head(limit).to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, pd.Timestamp):
                clean[k] = v.isoformat()
            elif pd.isna(v):
                clean[k] = None
            elif hasattr(v, "item"):
                clean[k] = v.item()
            elif isinstance(v, float):
                clean[k] = round(v, 4)
            else:
                clean[k] = v
        out.append(clean)
    return out, truncated


def _fences(series: pd.Series, settings: Settings) -> tuple[float, dict]:
    """Upper cutoff for a numeric series, using the configured method."""
    if settings.outlier_method == "mad":
        median = float(series.median())
        mad = float((series - median).abs().median())
        if mad == 0:
            return float("inf"), {"method": "mad", "median": median, "mad": 0.0}
        cutoff = median + settings.mad_z_threshold * mad / 0.6745
        return cutoff, {
            "method": "mad",
            "median": round(median, 2),
            "mad": round(mad, 2),
            "z_threshold": settings.mad_z_threshold,
        }
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    cutoff = float(q3) + settings.iqr_multiplier * iqr
    return cutoff, {
        "method": "iqr",
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
        "iqr": round(iqr, 2),
        "multiplier": settings.iqr_multiplier,
    }


def detect_resolution_outliers(frame: pd.DataFrame, settings: Settings) -> AnomalyGroup:
    """Resolution times far above normal *for comparable tickets*.

    Grouped by priority rather than category. Measured on the shipped data:
    category medians are 11.4 / 12.1 / 13.2 hrs, so category carries no signal,
    while priority medians run 3.7 / 6.9 / 14.5 / 26.6 hrs. A single global fence
    is dominated by Low-priority tickets that are perfectly normal for their
    class, and misses Critical tickets running five times their class median.
    """
    resolved = frame[frame[S.RESOLUTION_TIME].notna()].copy()
    group_by = settings.outlier_group_by

    if resolved.empty:
        return AnomalyGroup(
            type=AnomalyType.RESOLUTION_OUTLIER.value,
            title="Unusually long resolution times",
            severity="medium",
            rule="no resolved tickets in scope",
            count=0,
        )

    flagged: list[pd.DataFrame] = []
    per_group: dict[str, dict] = {}

    if group_by == "none":
        cutoff, stats = _fences(resolved[S.RESOLUTION_TIME], settings)
        hit = resolved[resolved[S.RESOLUTION_TIME] > cutoff].copy()
        hit["threshold_hrs"] = round(cutoff, 2)
        flagged.append(hit)
        per_group["all"] = {**stats, "cutoff_hrs": round(cutoff, 2), "n": len(resolved)}
    else:
        key = S.PRIORITY if group_by == "priority" else S.CATEGORY
        for name, group in resolved.groupby(key, observed=True):
            if len(group) < settings.outlier_min_group_size:
                per_group[str(name)] = {"n": len(group), "skipped": "group too small"}
                continue
            cutoff, stats = _fences(group[S.RESOLUTION_TIME], settings)
            hit = group[group[S.RESOLUTION_TIME] > cutoff].copy()
            hit["threshold_hrs"] = round(cutoff, 2)
            flagged.append(hit)
            per_group[str(name)] = {
                **stats,
                "cutoff_hrs": round(cutoff, 2),
                "n": len(group),
                "median_hrs": round(float(group[S.RESOLUTION_TIME].median()), 2),
                "flagged": len(hit),
            }

    combined = (
        pd.concat(flagged).sort_values(S.RESOLUTION_TIME, ascending=False)
        if flagged
        else resolved.iloc[0:0]
    )
    rows, truncated = _rows(combined, settings.max_anomalies_per_detector, ["threshold_hrs"])

    scope = "all resolved tickets" if group_by == "none" else f"each {group_by}"
    method = settings.outlier_method.upper()
    rule = (
        f"{method} upper fence on resolution_time_hrs, computed within {scope}"
        + (f" (multiplier {settings.iqr_multiplier})" if settings.outlier_method == "iqr" else "")
    )

    return AnomalyGroup(
        type=AnomalyType.RESOLUTION_OUTLIER.value,
        title="Unusually long resolution times",
        severity="medium",
        rule=rule,
        count=len(combined),
        tickets=rows,
        truncated=truncated,
        stats={"groups": per_group, "resolved_in_scope": len(resolved)},
        note=(
            "Thresholds are computed per priority because priority, not category, "
            "drives resolution time in this dataset."
            if group_by == "priority"
            else ""
        ),
    )


def detect_aging_unresolved(
    frame: pd.DataFrame, settings: Settings, reference: datetime
) -> AnomalyGroup:
    """High and Critical tickets still open past the aging threshold.

    This is the rule named in the brief. On the shipped data it matches 80 of the
    80 unresolved High/Critical tickets, because the file spans three months and
    nothing was backfilled as resolved. The count is exact; the ticket list is
    capped and sorted oldest first so the worst cases surface.
    """
    unresolved = frame[frame[S.STATUS].isin(S.UNRESOLVED_STATUSES)].copy()
    urgent = unresolved[unresolved[S.PRIORITY].astype("string").isin(S.HIGH_URGENCY)].copy()

    if urgent.empty:
        return AnomalyGroup(
            type=AnomalyType.AGING_UNRESOLVED.value,
            title="Aging unresolved high-priority tickets",
            severity="high",
            rule=f"status in {sorted(S.UNRESOLVED_STATUSES)} and priority in "
            f"{sorted(S.HIGH_URGENCY)} and age > {settings.aging_hours}h",
            count=0,
        )

    age = (pd.Timestamp(reference) - urgent[S.CREATED_AT]).dt.total_seconds() / 3600.0
    urgent["age_hours"] = age.round(1)
    hit = urgent[urgent["age_hours"] > settings.aging_hours].sort_values(
        "age_hours", ascending=False
    )
    rows, truncated = _rows(hit, settings.max_anomalies_per_detector, ["age_hours"])

    return AnomalyGroup(
        type=AnomalyType.AGING_UNRESOLVED.value,
        title="Aging unresolved high-priority tickets",
        severity="high",
        rule=(
            f"status in {sorted(S.UNRESOLVED_STATUSES)} and priority in "
            f"{sorted(S.HIGH_URGENCY)} and age > {settings.aging_hours}h, "
            f"measured against {reference:%Y-%m-%d %H:%M}"
        ),
        count=len(hit),
        tickets=rows,
        truncated=truncated,
        stats={
            "urgent_unresolved_in_scope": len(urgent),
            "oldest_age_hours": round(float(hit["age_hours"].max()), 1) if len(hit) else None,
            "median_age_hours": round(float(hit["age_hours"].median()), 1) if len(hit) else None,
        },
        note=(
            "Nearly every unresolved urgent ticket clears this bar because the "
            "dataset spans three months. Raise AGING_HOURS to narrow it."
            if len(hit) > 0.8 * len(urgent) and len(urgent) > 10
            else ""
        ),
    )


def detect_response_sla_breach(frame: pd.DataFrame, settings: Settings) -> AnomalyGroup:
    """First responses slower than the per-priority target.

    Fixed targets, not statistics: response_time_hrs is uniform on [0.2, 5.0] in
    this dataset, so its IQR fence lands at 7.65 and a statistical detector would
    never fire. Priority and response time are uncorrelated here, which is itself
    the finding.
    """
    sla = settings.response_sla_hours
    priority = frame[S.PRIORITY].astype("string")
    target = priority.map(sla)
    breach = frame[frame[S.RESPONSE_TIME] > target].copy()
    breach["sla_hours"] = priority.map(sla)
    breach["over_by_hours"] = (breach[S.RESPONSE_TIME] - breach["sla_hours"]).round(2)
    breach = breach.sort_values("over_by_hours", ascending=False)

    rows, truncated = _rows(
        breach, settings.max_anomalies_per_detector, ["sla_hours", "over_by_hours"]
    )
    by_priority = (
        breach[S.PRIORITY].astype("string").value_counts().to_dict() if len(breach) else {}
    )

    return AnomalyGroup(
        type=AnomalyType.RESPONSE_SLA_BREACH.value,
        title="First response slower than target",
        severity="medium",
        rule="response_time_hrs > per-priority target "
        + ", ".join(f"{k}:{v}h" for k, v in sla.items()),
        count=len(breach),
        tickets=rows,
        truncated=truncated,
        stats={"by_priority": by_priority, "tickets_in_scope": len(frame)},
    )


def detect_low_ratings(frame: pd.DataFrame, settings: Settings) -> AnomalyGroup:
    """Resolved tickets the customer rated poorly."""
    rated = frame[frame[S.CUSTOMER_RATING].notna()]
    hit = rated[rated[S.CUSTOMER_RATING] <= settings.low_rating_threshold].sort_values(
        [S.CUSTOMER_RATING, S.CREATED_AT]
    )
    rows, truncated = _rows(hit, settings.max_anomalies_per_detector)

    by_agent = hit[S.AGENT_ID].value_counts().head(5).to_dict() if len(hit) else {}
    return AnomalyGroup(
        type=AnomalyType.LOW_RATING.value,
        title="Poorly rated resolutions",
        severity="low",
        rule=f"customer_rating <= {settings.low_rating_threshold}",
        count=len(hit),
        tickets=rows,
        truncated=truncated,
        stats={
            "rated_in_scope": len(rated),
            "share_of_rated": round(len(hit) / len(rated), 3) if len(rated) else None,
            "most_affected_agents": by_agent,
        },
    )


def detect_data_integrity(frame: pd.DataFrame, settings: Settings) -> AnomalyGroup:
    """Rows that contradict the schema's own rules.

    The dominant case in the shipped data is 28 tickets whose resolution_time_hrs
    is lower than their response_time_hrs, which is chronologically impossible.
    These rows are kept in the dataset rather than dropped, since removing them
    would silently shift every aggregate; they are surfaced here instead.
    """
    issues: list[pd.DataFrame] = []

    impossible = frame[
        frame[S.RESOLUTION_TIME].notna()
        & (frame[S.RESOLUTION_TIME] < frame[S.RESPONSE_TIME])
    ].copy()
    if len(impossible):
        impossible["issue"] = "resolved before first response"
        issues.append(impossible)

    resolved_no_time = frame[
        (frame[S.STATUS] == S.Status.RESOLVED.value) & frame[S.RESOLUTION_TIME].isna()
    ].copy()
    if len(resolved_no_time):
        resolved_no_time["issue"] = "resolved without a resolution time"
        issues.append(resolved_no_time)

    unresolved_with_time = frame[
        frame[S.STATUS].isin(S.UNRESOLVED_STATUSES) & frame[S.RESOLUTION_TIME].notna()
    ].copy()
    if len(unresolved_with_time):
        unresolved_with_time["issue"] = "unresolved but has a resolution time"
        issues.append(unresolved_with_time)

    resolved_no_rating = frame[
        (frame[S.STATUS] == S.Status.RESOLVED.value) & frame[S.CUSTOMER_RATING].isna()
    ].copy()
    if len(resolved_no_rating):
        resolved_no_rating["issue"] = "resolved without a customer rating"
        issues.append(resolved_no_rating)

    combined = pd.concat(issues) if issues else frame.iloc[0:0].assign(issue=None)
    rows, truncated = _rows(combined, settings.max_anomalies_per_detector, ["issue"])
    breakdown = combined["issue"].value_counts().to_dict() if len(combined) else {}

    return AnomalyGroup(
        type=AnomalyType.DATA_INTEGRITY.value,
        title="Internally inconsistent records",
        severity="high" if len(combined) else "low",
        rule=(
            "resolution_time_hrs < response_time_hrs, or resolution/rating fields "
            "disagreeing with status"
        ),
        count=len(combined),
        tickets=rows,
        truncated=truncated,
        stats={"by_issue": breakdown},
        note=(
            "These rows are counted in all other results. They are reported, not "
            "removed, so aggregates stay faithful to the source file."
            if len(combined)
            else ""
        ),
    )


def run_detectors(
    frame: pd.DataFrame,
    settings: Settings,
    reference: datetime,
    types: list[AnomalyType] | None = None,
    window: ResolvedWindow | None = None,
) -> dict:
    """Run the requested detectors over the in-scope tickets.

    Args:
        frame: Ticket data, already narrowed to the window if one applies.
        settings: Threshold configuration.
        reference: Clock for age calculations.
        types: Which detectors to run. None or [ALL] runs everything.
        window: The window used, echoed back for provenance.

    Returns:
        A dict with per-detector groups and a summary.
    """
    wanted = set(types or [AnomalyType.ALL])
    run_all = AnomalyType.ALL in wanted

    groups: list[AnomalyGroup] = []
    if run_all or AnomalyType.RESOLUTION_OUTLIER in wanted:
        groups.append(detect_resolution_outliers(frame, settings))
    if run_all or AnomalyType.AGING_UNRESOLVED in wanted:
        groups.append(detect_aging_unresolved(frame, settings, reference))
    if run_all or AnomalyType.RESPONSE_SLA_BREACH in wanted:
        groups.append(detect_response_sla_breach(frame, settings))
    if run_all or AnomalyType.LOW_RATING in wanted:
        groups.append(detect_low_ratings(frame, settings))
    if run_all or AnomalyType.DATA_INTEGRITY in wanted:
        groups.append(detect_data_integrity(frame, settings))

    flagged_ids: set[str] = set()
    for g in groups:
        flagged_ids.update(t[S.TICKET_ID] for t in g.tickets if t.get(S.TICKET_ID))

    return {
        "window": window.as_dict() if window else None,
        "tickets_in_scope": len(frame),
        "total_flags": sum(g.count for g in groups),
        "distinct_tickets_shown": len(flagged_ids),
        "detectors": [g.as_dict() for g in groups],
    }
