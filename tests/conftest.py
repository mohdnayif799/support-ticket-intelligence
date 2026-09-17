"""Shared fixtures.

Unit tests run against a small hand-built frame with known values, so assertions
state exact expected numbers. Integration tests use the real shipped CSV.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from app.config import Settings
from app.data.loader import load_tickets
from app.data.store import TicketStore
from app.domain import schema as S

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_CSV = PROJECT_ROOT / "data" / "support_tickets.csv"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(llm_provider="stub", csv_path=REAL_CSV)


@pytest.fixture
def real_store(settings) -> TicketStore:
    return TicketStore.from_csv(settings)


@pytest.fixture
def tiny_csv(tmp_path) -> Path:
    """Twelve rows with deliberately known properties.

    Includes an unresolved Critical ticket, a resolution-before-response row, a
    clear resolution-time outlier, and a low rating.
    """
    rows = [
        # id, created, category, priority, status, resp, resol, agent, rating, summary
        ("TKT-001", "2024-03-01 09:00", "Billing", "High", "Resolved", 0.5, 2.0, "AGT-01", 5, "Invoice wrong"),
        ("TKT-002", "2024-03-01 10:00", "Billing", "High", "Resolved", 1.0, 3.0, "AGT-01", 4, "Refund late"),
        ("TKT-003", "2024-03-02 11:00", "Billing", "High", "Resolved", 1.5, 2.5, "AGT-02", 4, "Double charge"),
        ("TKT-004", "2024-03-02 12:00", "Billing", "High", "Resolved", 1.0, 3.5, "AGT-02", 3, "Fee query"),
        ("TKT-005", "2024-03-03 13:00", "Billing", "High", "Resolved", 0.5, 2.0, "AGT-03", 5, "Plan change"),
        ("TKT-006", "2024-03-03 14:00", "Billing", "High", "Resolved", 1.2, 3.0, "AGT-03", 4, "Tax question"),
        ("TKT-007", "2024-03-04 15:00", "Billing", "High", "Resolved", 0.8, 2.2, "AGT-01", 5, "Card declined"),
        ("TKT-008", "2024-03-04 16:00", "Billing", "High", "Resolved", 1.1, 2.8, "AGT-02", 4, "Receipt copy"),
        # Outlier: 90h against a High-priority cluster sitting near 2-3.5h.
        ("TKT-009", "2024-03-05 08:00", "Billing", "High", "Resolved", 1.0, 90.0, "AGT-03", 1, "Stuck migration"),
        # Impossible: resolved before first response.
        ("TKT-010", "2024-03-05 09:00", "Technical", "Critical", "Resolved", 4.0, 1.0, "AGT-01", 2, "Outage"),
        # Unresolved Critical, old relative to the newest ticket.
        ("TKT-011", "2024-03-01 08:00", "Technical", "Critical", "Open", 3.0, None, "AGT-02", None, "Login broken"),
        ("TKT-012", "2024-03-06 10:00", "General", "Low", "Escalated", 2.0, None, "AGT-03", None, "Docs missing"),
    ]
    frame = pd.DataFrame(rows, columns=list(S.ALL_COLUMNS))
    path = tmp_path / "tiny.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def tiny_frame(tiny_csv):
    frame, _ = load_tickets(tiny_csv)
    return frame


@pytest.fixture
def tiny_reference() -> datetime:
    """Newest ticket in the tiny fixture."""
    return datetime(2024, 3, 6, 10, 0)
