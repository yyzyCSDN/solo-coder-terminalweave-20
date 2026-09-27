"""Transshipment misconnection early warning.

The monitor continuously re-evaluates a confirmed connection against the
latest operational picture (inbound arrival deviation, remaining discharge
work, yard dwell and fallback sailing capacity) and produces versioned
alerts carrying ranked rebooking candidates.

Alerts are advisory only: the monitor never receives, and therefore never
mutates, the confirmed ConnectionGraph. Applying a candidate is an explicit
decision for the embedding application. Updates are guarded by prediction
vintage so a stale forecast can never overwrite a newer assessment, and
capacity contention between alerts is resolved by a deterministic
allocation pass.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError
from terminalweave.transshipment.connections import Connection, ServiceCall
from terminalweave.transshipment.rebooking import Sailing

class RiskLevel(str, Enum):
    ON_TIME = 'on_time'
    WATCH = 'watch'
    AT_RISK = 'at_risk'
    MISSED = 'missed'

_SEVERITY = {RiskLevel.ON_TIME: 0, RiskLevel.WATCH: 1, RiskLevel.AT_RISK: 2, RiskLevel.MISSED: 3}

def _escalate(level: RiskLevel) -> RiskLevel:
    # dwell pressure can raise concern but never declares a miss on its own
    return {RiskLevel.ON_TIME: RiskLevel.WATCH, RiskLevel.WATCH: RiskLevel.AT_RISK}.get(level, level)

@dataclass(frozen=True)
class MisconnectPolicy:
    watch_margin: timedelta
    risk_margin: timedelta
    dwell_limit: timedelta
    max_wait: timedelta = timedelta(days=7)

    def __post_init__(self) -> None:
        if not timedelta() <= self.risk_margin <= self.watch_margin:
            raise ValidationError('risk margin must be between zero and watch margin')
        if self.dwell_limit.total_seconds() < 0 or self.max_wait.total_seconds() <= 0:
            raise ValidationError('invalid misconnect policy')

@dataclass(frozen=True)
class OperationSnapshot:
    predicted_inbound_arrival: datetime
    remaining_moves: int
    moves_per_hour: float
    dwell_elapsed: timedelta = timedelta()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'predicted_inbound_arrival', ensure_utc(self.predicted_inbound_arrival))
        if self.remaining_moves < 0 or self.moves_per_hour <= 0 or self.dwell_elapsed.total_seconds() < 0:
            raise ValidationError('invalid operation snapshot')

@dataclass(frozen=True)
class RebookingCandidate:
    sailing_id: str
    service: str
    departure: datetime
    remaining_teu: int
    rank: int

@dataclass(frozen=True)
class MisconnectAlert:
    container_id: str
    outbound_call: str
    revision: int
    assessed_at: datetime
    level: RiskLevel
    ready_at: datetime
    slack: timedelta
    teu: int
    recoverable: bool
    candidates: tuple[RebookingCandidate, ...]

class MisconnectMonitor:

    def __init__(self, policy: MisconnectPolicy):
        self.policy = policy
        self._alerts: dict[str, MisconnectAlert] = {}

    def current(self, container_id: str) -> MisconnectAlert | None:
        return self._alerts.get(container_id)

    def _ready_at(self, connection: Connection, snapshot: OperationSnapshot) -> datetime:
        work = timedelta(hours=snapshot.remaining_moves / snapshot.moves_per_hour)
        return snapshot.predicted_inbound_arrival + work + connection.minimum_transfer + connection.yard_buffer

    def _level(self, slack: timedelta, snapshot: OperationSnapshot) -> RiskLevel:
        if slack < timedelta():
            return RiskLevel.MISSED
        if slack <= self.policy.risk_margin:
            level = RiskLevel.AT_RISK
        elif slack <= self.policy.watch_margin:
            level = RiskLevel.WATCH
        else:
            level = RiskLevel.ON_TIME
        if snapshot.dwell_elapsed > self.policy.dwell_limit:
            level = _escalate(level)
        return level

    def _candidates(self, fallbacks: list[Sailing], ready_at: datetime, teu: int) -> tuple[RebookingCandidate, ...]:
        eligible = [s for s in fallbacks if s.departure >= ready_at and s.departure - ready_at <= self.policy.max_wait and (s.remaining_teu >= teu)]
        ordered = sorted(eligible, key=lambda s: (s.departure, -s.remaining_teu, s.sailing_id))
        return tuple((RebookingCandidate(s.sailing_id, s.service, s.departure, s.remaining_teu, i + 1) for i, s in enumerate(ordered)))

    def update(self, connection: Connection, outbound: ServiceCall, snapshot: OperationSnapshot, fallbacks: list[Sailing], teu: int, assessed_at: datetime) -> MisconnectAlert:
        assessed_at = ensure_utc(assessed_at)
        if teu <= 0:
            raise ValidationError('TEU must be positive')
        existing = self._alerts.get(connection.container_id)
        if existing and assessed_at <= existing.assessed_at:
            return existing
        ready_at = self._ready_at(connection, snapshot)
        slack = outbound.departure - ready_at
        candidates = self._candidates(fallbacks, ready_at, teu)
        alert = MisconnectAlert(container_id=connection.container_id, outbound_call=connection.outbound_call, revision=existing.revision + 1 if existing else 1, assessed_at=assessed_at, level=self._level(slack, snapshot), ready_at=ready_at, slack=slack, teu=teu, recoverable=bool(candidates), candidates=candidates)
        self._alerts[connection.container_id] = alert
        return alert

def allocate_candidates(alerts: list[MisconnectAlert], sailings: list[Sailing]) -> dict[str, RebookingCandidate | None]:
    """Resolve berth-on-sailing contention deterministically.

    Alerts are served highest-severity first (then tightest slack, then
    container id); each takes its best-ranked candidate that still has
    capacity in the ledger. Identical inputs always yield the identical
    assignment, and a slot is never committed twice.
    """
    ids = [a.container_id for a in alerts]
    if len(set(ids)) != len(ids):
        raise ValidationError('duplicate alert for container')
    remaining = {s.sailing_id: s.remaining_teu for s in sailings}
    ordered = sorted(alerts, key=lambda a: (-_SEVERITY[a.level], a.slack, a.container_id))
    result: dict[str, RebookingCandidate | None] = {}
    for alert in ordered:
        chosen = None
        for cand in alert.candidates:
            if remaining.get(cand.sailing_id, 0) >= alert.teu:
                remaining[cand.sailing_id] -= alert.teu
                chosen = cand
                break
        result[alert.container_id] = chosen
    return result
