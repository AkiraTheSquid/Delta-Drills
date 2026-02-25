from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Session:
    area_id: int
    timestamp: str  # ISO-8601 string
    percent_correct: float


@dataclass(frozen=True)
class Area:
    id: int
    name: str
    weight: float  # exam blueprint weight (sums to 1.0)


@dataclass(frozen=True)
class RecommendationRow:
    area_id: int
    area_name: str
    weight: float
    learning_rate: float
    gradient: float
    recent_sessions: int
    current_performance: Optional[float]


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def estimate_learning_rate(sessions: Iterable[Session]) -> float:
    """
    EWMA learning-rate estimate.
    s_t = alpha * x_t + (1 - alpha) * s_{t-1}
    alpha = 1 - exp(-lambda)
    Returns percentage points improved per hour.
    """
    ordered = sorted(sessions, key=lambda s: _parse_ts(s.timestamp))
    if len(ordered) < 2:
        return 0.5

    lambda_ = 0.3
    alpha = 1 - exp(-lambda_)

    rates: List[float] = []
    for i in range(1, len(ordered)):
        current = ordered[i]
        previous = ordered[i - 1]
        delta_percent = current.percent_correct - previous.percent_correct
        delta_hours = (
            _parse_ts(current.timestamp) - _parse_ts(previous.timestamp)
        ).total_seconds() / 3600.0
        if delta_hours > 0:
            rates.append(delta_percent / delta_hours)

    if not rates:
        return 0.5

    s = rates[0]
    for t in range(1, len(rates)):
        s = alpha * rates[t] + (1 - alpha) * s

    return max(0.0, s)


def compute_recommendations(areas: Iterable[Area], all_sessions: Iterable[Session]) -> List[RecommendationRow]:
    two_weeks_ago = datetime.now(timezone.utc).timestamp() - (14 * 24 * 60 * 60)

    rows: List[RecommendationRow] = []
    for area in areas:
        area_sessions = [
            s
            for s in all_sessions
            if s.area_id == area.id and _parse_ts(s.timestamp).timestamp() >= two_weeks_ago
        ]
        area_sessions.sort(key=lambda s: _parse_ts(s.timestamp), reverse=True)
        area_sessions = area_sessions[:10]

        learning_rate = estimate_learning_rate(area_sessions)
        gradient = area.weight * learning_rate
        current_performance = area_sessions[0].percent_correct if area_sessions else None

        rows.append(
            RecommendationRow(
                area_id=area.id,
                area_name=area.name,
                weight=area.weight,
                learning_rate=learning_rate,
                gradient=gradient,
                recent_sessions=len(area_sessions),
                current_performance=current_performance,
            )
        )

    return sorted(rows, key=lambda r: r.gradient, reverse=True)


AREAS: List[Area] = [
    Area(1, "General Principles of Foundational Science", 0.020),
    Area(2, "Immune System, Blood & Lymphoreticular System, and Multisystem Processes/Disorders", 0.070),
    Area(3, "Behavioral Health", 0.050),
    Area(4, "Nervous System & Special Senses", 0.090),
    Area(5, "Skin & Subcutaneous Tissue", 0.050),
    Area(6, "Musculoskeletal System", 0.060),
    Area(7, "Cardiovascular System", 0.100),
    Area(8, "Respiratory System", 0.090),
    Area(9, "Gastrointestinal System", 0.070),
    Area(10, "Renal/Urinary & Male Reproductive Systems", 0.050),
    Area(11, "Pregnancy/Childbirth & Female Reproductive System & Breast", 0.080),
    Area(12, "Endocrine System", 0.060),
    Area(13, "Biostatistics & Epidemiology/Population Health & Interpretation of Medical Literature", 0.120),
    Area(14, "Social Sciences: Communication Skills/Ethics/Patient Safety", 0.090),
]
