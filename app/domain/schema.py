"""Canonical dataset schema.

Single source of truth for column names, allowed values, and types. Every other
module imports from here rather than hardcoding strings, so a schema change is a
one-file edit.

The assessment brief spells three columns two different ways (section 3.2's table
header vs section 3.3's descriptions). The shipped CSV uses the section 3.3
spelling, but ALIASES accepts either so the loader does not break if the other
variant ever shows up.
"""

from __future__ import annotations

from enum import Enum

# --- Canonical column names -------------------------------------------------

TICKET_ID = "ticket_id"
CREATED_AT = "created_at"
CATEGORY = "category"
PRIORITY = "priority"
STATUS = "status"
RESPONSE_TIME = "response_time_hrs"
RESOLUTION_TIME = "resolution_time_hrs"
AGENT_ID = "agent_id"
CUSTOMER_RATING = "customer_rating"
ISSUE_SUMMARY = "issue_summary"

ALL_COLUMNS: tuple[str, ...] = (
    TICKET_ID,
    CREATED_AT,
    CATEGORY,
    PRIORITY,
    STATUS,
    RESPONSE_TIME,
    RESOLUTION_TIME,
    AGENT_ID,
    CUSTOMER_RATING,
    ISSUE_SUMMARY,
)

REQUIRED_COLUMNS: frozenset[str] = frozenset(ALL_COLUMNS)

#: Columns that hold numbers and can therefore be averaged, summed, etc.
NUMERIC_COLUMNS: frozenset[str] = frozenset(
    {RESPONSE_TIME, RESOLUTION_TIME, CUSTOMER_RATING}
)

#: Columns safe to group by. Free text and unique IDs are excluded on purpose:
#: grouping by ticket_id or issue_summary produces 500 one-row groups.
GROUPABLE_COLUMNS: frozenset[str] = frozenset({CATEGORY, PRIORITY, STATUS, AGENT_ID})

#: Alternate spellings seen in the brief, mapped to canonical names.
#: Keys are compared after lowercasing and stripping.
ALIASES: dict[str, str] = {
    "resp_time_hrs": RESPONSE_TIME,
    "response_time": RESPONSE_TIME,
    "resol_time_hrs": RESOLUTION_TIME,
    "resolution_time": RESOLUTION_TIME,
    "cust_rating": CUSTOMER_RATING,
    "rating": CUSTOMER_RATING,
    "ticket": TICKET_ID,
    "id": TICKET_ID,
    "agent": AGENT_ID,
    "created": CREATED_AT,
    "summary": ISSUE_SUMMARY,
}


# --- Controlled vocabularies ------------------------------------------------


class Category(str, Enum):
    BILLING = "Billing"
    TECHNICAL = "Technical"
    GENERAL = "General"


class Priority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class Status(str, Enum):
    OPEN = "Open"
    RESOLVED = "Resolved"
    ESCALATED = "Escalated"


#: Priority ordered least to most urgent, for sorting and SLA lookups.
PRIORITY_ORDER: tuple[str, ...] = (
    Priority.LOW.value,
    Priority.MEDIUM.value,
    Priority.HIGH.value,
    Priority.CRITICAL.value,
)

#: A ticket is "unresolved" if it is in any of these states.
UNRESOLVED_STATUSES: frozenset[str] = frozenset({Status.OPEN.value, Status.ESCALATED.value})

#: Priorities treated as high urgency by the aging anomaly detector.
HIGH_URGENCY: frozenset[str] = frozenset({Priority.HIGH.value, Priority.CRITICAL.value})

RATING_MIN = 1
RATING_MAX = 5


def canonical_column(raw: str) -> str | None:
    """Map a raw CSV header to its canonical name, or None if unrecognised."""
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if key in REQUIRED_COLUMNS:
        return key
    return ALIASES.get(key)
