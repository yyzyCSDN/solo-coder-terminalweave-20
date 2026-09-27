"""Missed-connection early warning for transshipment containers.

The monitor continuously evaluates whether a container will make its confirmed
outbound call using four live signals:

* arrival deviation of the inbound call;
* work remaining before the container is discharge-ready;
* dwell already spent in the yard;
* remaining capacity on the connecting (and downstream) voyages.

Warnings are *advisory*: evaluating or re-evaluating a forecast never mutates
the confirmed :class:`ConnectionGraph`. Candidate rebookings are proposed and
contested slots are allocated consistently so two candidates can never reserve
the same TEU. Every observation carries a timestamp; stale telemetry and stale
sailing catalogs are rejected rather than allowed to overwrite newer results.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum
from typing import Iterable

from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, ConflictError, NotFoundError
from terminalweave.transshipment.connections import ConnectionGraph
from terminalweave.transshipment.rebooking import RebookingPlanner, Sailing


class RiskLevel(IntEnum):
    OK = 0
    WATCH = 1
    HIGH = 2
    CRITICAL = 3


# Risk factor labels, ordered worst-first so evaluation stays deterministic.
FACTOR_ARRIVAL = 'arrival_deviation'
FACTOR_WORK = 'work_remaining'
FACTOR_DWELL = 'dwell'
FACTOR_HEADROOM = 'voyage_headroom'
FACTOR_ORDER = (FACTOR_ARRIVAL, FACTOR_WORK, FACTOR_DWELL, FACTOR_HEADROOM)


@dataclass(frozen=True)
class RiskThresholds:
    """Slack floors that turn projected slippage into risk scores.

    ``arrival_tolerance`` / ``work_tolerance`` / ``dwell_allowance`` are the
    amounts of slack each factor may consume before risk starts. ``headroom``
    is the free-TEU floor on the connecting voyage. ``watch_slack`` and
    ``critical_slack`` layer an absolute floor on projected schedule slack.
    """
    arrival_tolerance: timedelta = timedelta(hours=2)
    work_tolerance: timedelta = timedelta(hours=1)
    dwell_allowance: timedelta = timedelta(hours=12)
    headroom_teu: int = 20
    watch_slack: timedelta = timedelta(hours=6)
    critical_slack: timedelta = timedelta(hours=1)

    def __post_init__(self) -> None:
        for name in ('arrival_tolerance', 'work_tolerance', 'dwell_allowance', 'watch_slack', 'critical_slack'):
            if getattr(self, name).total_seconds() < 0:
                raise ValidationError(f'{name} cannot be negative')
        if self.headroom_teu < 0 or self.critical_slack > self.watch_slack:
            raise ValidationError('invalid risk thresholds')


@dataclass(frozen=True)
class ConnectionTelemetry:
    """Live observation for one container's confirmed connection."""
    container_id: str
    observed_at: datetime
    arrival_deviation: timedelta = timedelta()
    work_remaining: timedelta = timedelta()
    dwell: timedelta = timedelta()
    destination: str = ''
    teu: int = 1
    connecting_remaining_teu: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'observed_at', ensure_utc(self.observed_at))
        for name in ('arrival_deviation', 'work_remaining', 'dwell'):
            if getattr(self, name).total_seconds() < 0:
                raise ValidationError(f'{name} cannot be negative')
        if self.teu <= 0:
            raise ValidationError('TEU must be positive')
        if self.connecting_remaining_teu is not None and self.connecting_remaining_teu < 0:
            raise ValidationError('connecting capacity cannot be negative')


@dataclass(frozen=True)
class ConnectionWarning:
    container_id: str
    observed_at: datetime
    projected_ready: datetime
    outbound_departure: datetime
    projected_slack: timedelta
    score: float
    level: RiskLevel
    factors: tuple[tuple[str, float], ...]
    misses_connection: bool

    @property
    def at_risk(self) -> bool:
        return self.level >= RiskLevel.WATCH


@dataclass(frozen=True)
class RebookingOption:
    sailing_id: str
    service: str
    departure: datetime
    remaining_teu: int


@dataclass(frozen=True)
class RebookingProposal:
    container_id: str
    observed_at: datetime
    options: tuple[RebookingOption, ...]
    catalog_version: int


@dataclass(frozen=True)
class Allocation:
    """Result of consistent candidate allocation against one sailing."""
    sailing_id: str
    assignments: tuple[str, ...]
    allocated_teu: int
    denied: tuple[str, ...]


class MissedConnectionMonitor:
    """Evaluate connection risk and propose/rebook alternative sailings.

    The monitor keeps the latest observation per container plus the latest
    warning and proposal. Re-evaluation with an older ``observed_at`` is
    rejected, so a late-arriving prediction can never overwrite a newer one.
    """

    def __init__(self, graph: ConnectionGraph, thresholds: RiskThresholds | None = None,
                 planner: RebookingPlanner | None = None,
                 candidate_level: RiskLevel = RiskLevel.WATCH,
                 max_wait: timedelta = timedelta(days=7)):
        self.graph = graph
        self.thresholds = thresholds or RiskThresholds()
        self.planner = planner or RebookingPlanner()
        if candidate_level < RiskLevel.OK:
            raise ValidationError('candidate level must be at least OK')
        self.candidate_level = candidate_level
        self.max_wait = max_wait
        self._telemetry: dict[str, ConnectionTelemetry] = {}
        self._warnings: dict[str, ConnectionWarning] = {}
        self._proposals: dict[str, RebookingProposal] = {}
        self._catalog: tuple[Sailing, ...] = ()
        self._catalog_version: int = 0
        self._catalog_as_of: datetime | None = None
        # Committed allocations: sailing_id -> {container_id: teu}. Assignments
        # are held against the catalog snapshot until released or re-catalogued.
        self._allocations: dict[str, dict[str, int]] = {}

    # -- observation intake -------------------------------------------------

    def observe(self, telemetry: ConnectionTelemetry) -> None:
        """Accept a fresh observation; reject anything older than what we hold."""
        current = self._telemetry.get(telemetry.container_id)
        if current is not None and telemetry.observed_at < current.observed_at:
            raise ConflictError(
                f'stale telemetry for {telemetry.container_id}: '
                f'{telemetry.observed_at.isoformat()} precedes {current.observed_at.isoformat()}'
            )
        if not any(c.container_id == telemetry.container_id for c in self.graph.connections):
            raise ValidationError('no confirmed connection for container')
        self._telemetry[telemetry.container_id] = telemetry

    def observe_many(self, observations: Iterable[ConnectionTelemetry]) -> int:
        accepted = 0
        for telemetry in observations:
            self.observe(telemetry)
            accepted += 1
        return accepted

    #-- risk evaluation ----------------------------------------------------

    def projected_ready(self, telemetry: ConnectionTelemetry) -> datetime:
        connection = self._connection(telemetry.container_id)
        inbound = self.graph.calls[connection.inbound_call]
        # Operational readiness: discharge work must finish, accounting for the
        # inbound call deviation.
        discharge_ready = inbound.arrival + telemetry.arrival_deviation + telemetry.work_remaining
        # Yard readiness: the transfer window starts when the container actually
        # entered the yard (observed_at - dwell) and needs min_transfer + buffer.
        yard_ready = telemetry.observed_at - telemetry.dwell + connection.minimum_transfer + connection.yard_buffer
        return max(discharge_ready, yard_ready, inbound.arrival)

    def evaluate(self, container_id: str, now: datetime | None = None) -> ConnectionWarning:
        """(Re)compute the warning for one container from its latest telemetry."""
        now = ensure_utc(now) if now else None
        telemetry = self._telemetry.get(container_id)
        if telemetry is None:
            raise ValidationError('no telemetry observed for container')
        if now is not None and telemetry.observed_at > now:
            raise ValidationError('telemetry observed in the future')
        connection = self._connection(container_id)
        outbound = self.graph.calls[connection.outbound_call]
        ready = self.projected_ready(telemetry)
        slack = outbound.departure - ready

        thresholds = self.thresholds
        arrival_factor = self._linear_factor(
            telemetry.arrival_deviation,
            thresholds.arrival_tolerance,
            thresholds.watch_slack,
        )
        work_factor = self._linear_factor(
            telemetry.work_remaining,
            thresholds.work_tolerance,
            thresholds.watch_slack,
        )
        dwell_excess = max(timedelta(), telemetry.dwell - thresholds.dwell_allowance)
        dwell_factor = self._linear_factor(
            dwell_excess,
            timedelta(),
            thresholds.watch_slack,
        )
        remaining = telemetry.connecting_remaining_teu
        if remaining is None:
            headroom_factor = 0.0
        elif remaining < telemetry.teu:
            headroom_factor = 1.0
        elif remaining >= thresholds.headroom_teu:
            headroom_factor = 0.0
        else:
            headroom_factor = self._linear_scalar(
                thresholds.headroom_teu - remaining,
                0,
                max(1, thresholds.headroom_teu - telemetry.teu),
            )

        raw = {
            FACTOR_ARRIVAL: arrival_factor,
            FACTOR_WORK: work_factor,
            FACTOR_DWELL: dwell_factor,
            FACTOR_HEADROOM: headroom_factor,
        }
        factors = tuple((label, round(raw[label], 6)) for label in FACTOR_ORDER if raw[label] > 0.0)
        score = round(max(raw.values()), 6) if raw else 0.0

        if slack < thresholds.critical_slack or (remaining is not None and remaining < telemetry.teu):
            level = RiskLevel.CRITICAL
        elif slack < thresholds.watch_slack or score >= 0.8:
            level = RiskLevel.HIGH
        elif score > 0.0:
            level = RiskLevel.WATCH
        else:
            level = RiskLevel.OK

        warning = ConnectionWarning(
            container_id=container_id,
            observed_at=telemetry.observed_at,
            projected_ready=ready,
            outbound_departure=outbound.departure,
            projected_slack=slack,
            score=score,
            level=level,
            factors=factors,
            misses_connection=ready > outbound.departure,
        )
        previous = self._warnings.get(container_id)
        if previous is not None and warning.observed_at < previous.observed_at:
            raise ConflictError('stale evaluation cannot replace newer warning')
        self._warnings[container_id] = warning
        if warning.at_risk:
            self._refresh_proposal(telemetry, warning)
        else:
            self._proposals.pop(container_id, None)
        return warning

    def evaluate_all(self, now: datetime | None = None) -> list[ConnectionWarning]:
        """Re-evaluate every observed container, most recent observation first."""
        order = sorted(self._telemetry, key=lambda cid: self._telemetry[cid].observed_at)
        return [self.evaluate(cid, now) for cid in order]

    #-- candidate rebooking -------------------------------------------------

    def load_sailing_catalog(self, sailings: Iterable[Sailing], as_of: datetime, version: int) -> None:
        """Install a fresh sailing catalog; older versions are refused.

        Capacity is snapshotted at load time so allocations run against one
        consistent view; a later catalog must carry a strictly higher version.
        """
        as_of = ensure_utc(as_of)
        if self._catalog_as_of is not None and as_of < self._catalog_as_of:
            raise ConflictError('stale sailing catalog cannot replace newer catalog')
        if version < self._catalog_version:
            raise ConflictError('stale catalog version cannot replace newer catalog')
        catalog = tuple(sorted(sailings, key=lambda s: s.sailing_id))
        ids = [s.sailing_id for s in catalog]
        if len(ids) != len(set(ids)):
            raise ValidationError('duplicate sailing id in catalog')
        self._catalog = catalog
        self._catalog_version = version
        self._catalog_as_of = as_of
        # Options priced against an earlier catalog cannot contest new slots,
        # and holds against the old capacity picture are void.
        self._proposals.clear()
        self._allocations.clear()

    def proposal(self, container_id: str) -> RebookingProposal | None:
        return self._proposals.get(container_id)

    def proposals(self) -> list[RebookingProposal]:
        return [self._proposals[cid] for cid in sorted(self._proposals)]

    def _live_candidates(self, sailing_id: str) -> list[tuple[ConnectionWarning, RebookingProposal, ConnectionTelemetry]]:
        candidates: list[tuple[ConnectionWarning, RebookingProposal, ConnectionTelemetry]] = []
        for cid, proposal in self._proposals.items():
            # Candidates priced against an outdated catalog cannot contest slots.
            if proposal.catalog_version != self._catalog_version:
                continue
            if any(o.sailing_id == sailing_id for o in proposal.options):
                candidates.append((self._warnings[cid], proposal, self._telemetry[cid]))
        candidates.sort(key=lambda item: (
            -int(item[0].level),
            item[0].projected_slack,
            item[1].observed_at,
            item[0].container_id,
        ))
        return candidates

    def preview_allocation(self, sailing_id: str) -> Allocation:
        """Show how slots would resolve against current holds without committing."""
        return self._resolve(sailing_id, commit=False)

    def allocate_candidates(self, sailing_id: str) -> Allocation:
        """Commit contested TEU on one sailing, consistently across candidates.

        Candidates are processed in a deterministic priority order (highest
        risk, tightest projected slack, earliest observation, container id).
        Accepted assignments are held in a ledger against the catalog
        snapshot, so two candidates can never be sold the same TEU and
        repeated calls are idempotent.
        """
        return self._resolve(sailing_id, commit=True)

    def _resolve(self, sailing_id: str, commit: bool) -> Allocation:
        sailing = next((s for s in self._catalog if s.sailing_id == sailing_id), None)
        if sailing is None:
            raise ValidationError('unknown sailing in catalog')
        candidates = self._live_candidates(sailing_id)
        live_ids = {warning.container_id for warning, _p, _t in candidates}
        ledger = self._allocations.setdefault(sailing_id, {})
        if commit:
            # Drop holds belonging to containers whose latest proposal moved away.
            for cid in [c for c in ledger if c not in live_ids]:
                del ledger[cid]
        held = dict(ledger)
        remaining = sailing.remaining_teu - sum(held.values())
        if remaining < 0:
            raise ConflictError(f'allocation ledger overbooked on sailing {sailing_id}')
        denied: list[str] = []
        for warning, _proposal, telemetry in candidates:
            cid = warning.container_id
            if cid in held:
                continue
            if telemetry.teu <= remaining:
                held[cid] = telemetry.teu
                remaining -= telemetry.teu
                if commit:
                    ledger[cid] = telemetry.teu
            else:
                denied.append(cid)
        ordered = tuple(warning.container_id for warning, _p, _t in candidates if warning.container_id in held)
        return Allocation(
            sailing_id=sailing_id,
            assignments=ordered,
            allocated_teu=sum(held.values()),
            denied=tuple(denied),
        )

    def committed_teu(self, sailing_id: str) -> int:
        return sum(self._allocations.get(sailing_id, {}).values())

    def release_assignment(self, container_id: str, sailing_id: str) -> None:
        """Return a committed container's slots to the sailing pool."""
        ledger = self._allocations.get(sailing_id, {})
        if container_id not in ledger:
            raise NotFoundError('no committed assignment for container on sailing')
        del ledger[container_id]

    def allocation(self, sailing_id: str) -> Allocation:
        """Re-read the current committed state without changing it."""
        sailing = next((s for s in self._catalog if s.sailing_id == sailing_id), None)
        if sailing is None:
            raise ValidationError('unknown sailing in catalog')
        candidates = self._live_candidates(sailing_id)
        ledger = self._allocations.get(sailing_id, {})
        ordered = tuple(w.container_id for w, _p, _t in candidates if w.container_id in ledger)
        denied = tuple(w.container_id for w, _p, _t in candidates if w.container_id not in ledger)
        return Allocation(sailing_id, ordered, sum(ledger.values()), denied)

    def allocate_all(self) -> list[Allocation]:
        """Run consistent allocation across every sailing offered as an option."""
        sailing_ids = {o.sailing_id for proposal in self._proposals.values() for o in proposal.options}
        return [self.allocate_candidates(sid) for sid in sorted(sailing_ids)]

    #-- helpers -------------------------------------------------------------

    def _connection(self, container_id: str):
        connection = next((c for c in self.graph.connections if c.container_id == container_id), None)
        if connection is None:
            raise ValidationError('no confirmed connection for container')
        return connection

    @staticmethod
    def _linear_factor(value: timedelta, floor: timedelta, full: timedelta) -> float:
        if value <= floor:
            return 0.0
        if full <= floor:
            return 1.0
        ratio = (value - floor).total_seconds() / (full - floor).total_seconds()
        return min(1.0, max(0.0, ratio))

    @staticmethod
    def _linear_scalar(value: float, floor: float, full: float) -> float:
        if value <= floor:
            return 0.0
        if full <= floor:
            return 1.0
        return min(1.0, max(0.0, (value - floor) / (full - floor)))

    def _refresh_proposal(self, telemetry: ConnectionTelemetry, warning: ConnectionWarning) -> None:
        if int(warning.level) < int(self.candidate_level):
            self._proposals.pop(telemetry.container_id, None)
            return
        if not self._catalog or self._catalog_as_of is None:
            self._proposals.pop(telemetry.container_id, None)
            return
        if warning.observed_at < self._catalog_as_of:
            raise ConflictError('telemetry older than catalog; refresh observations before proposing')
        connection = self._connection(telemetry.container_id)
        preferred = self.graph.calls[connection.outbound_call].service
        ranked = self.planner.rank(
            destination=telemetry.destination,
            ready_at=warning.projected_ready,
            teu=telemetry.teu,
            sailings=list(self._catalog),
            preferred_service=preferred,
            max_wait=self.max_wait,
        )
        options = tuple(
            RebookingOption(s.sailing_id, s.service, s.departure, s.remaining_teu)
            for s in ranked
        )
        proposal = RebookingProposal(
            container_id=telemetry.container_id,
            observed_at=telemetry.observed_at,
            options=options,
            catalog_version=self._catalog_version,
        )
        previous = self._proposals.get(telemetry.container_id)
        if previous is not None and proposal.observed_at < previous.observed_at:
            raise ConflictError('stale proposal cannot replace newer proposal')
        self._proposals[telemetry.container_id] = proposal

    #-- read-only views -----------------------------------------------------

    @property
    def warnings(self) -> dict[str, ConnectionWarning]:
        return dict(self._warnings)

    def at_risk(self, min_level: RiskLevel = RiskLevel.WATCH) -> list[ConnectionWarning]:
        return sorted(
            (w for w in self._warnings.values() if w.level >= min_level),
            key=lambda w: (-int(w.level), w.projected_slack, w.container_id),
        )
