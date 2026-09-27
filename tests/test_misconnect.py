from datetime import datetime, timedelta, timezone

import pytest

from terminalweave.transshipment.connections import ServiceCall, Connection, ConnectionGraph
from terminalweave.transshipment.rebooking import Sailing
from terminalweave.transshipment.misconnect import (
    MisconnectPolicy, OperationSnapshot, MisconnectMonitor, RiskLevel,
    RebookingCandidate, allocate_candidates,
)

UTC = timezone.utc
D = datetime(2026, 1, 1, tzinfo=UTC)

POLICY = MisconnectPolicy(
    watch_margin=timedelta(hours=6),
    risk_margin=timedelta(hours=2),
    dwell_limit=timedelta(hours=24),
)


def connection(container_id='C1', transfer=timedelta(hours=2)):
    return Connection(container_id, 'IN', 'OUT', transfer)


def outbound(departure):
    return ServiceCall('OUT', 'S2', departure - timedelta(hours=3), departure)


def monitor():
    return MisconnectMonitor(POLICY)


def sailing(sid, dep, remaining, service='FB', booked=0, capacity=None):
    return Sailing(sid, service, 'SZX', dep, capacity if capacity is not None else booked + remaining, booked)


def test_on_time_when_slack_absorbs_deviation():
    m = monitor()
    snap = OperationSnapshot(D + timedelta(hours=1), remaining_moves=0, moves_per_hour=30)
    alert = m.update(connection(), outbound(D + timedelta(hours=12)), snap, [], 1, D)
    assert alert.level == RiskLevel.ON_TIME
    assert alert.slack == timedelta(hours=9)
    assert alert.revision == 1


def test_arrival_deviation_and_remaining_work_escalate_risk():
    m = monitor()
    # +7h deviation, 2h of discharge work left, 2h transfer -> ready D+11, slack 1h
    snap = OperationSnapshot(D + timedelta(hours=7), remaining_moves=60, moves_per_hour=30)
    alert = m.update(connection(), outbound(D + timedelta(hours=12)), snap, [], 1, D)
    assert alert.level == RiskLevel.AT_RISK
    assert alert.ready_at == D + timedelta(hours=11)

    snap_late = OperationSnapshot(D + timedelta(hours=11), remaining_moves=60, moves_per_hour=30)
    missed = m.update(connection(), outbound(D + timedelta(hours=12)), snap_late, [], 1, D + timedelta(hours=1))
    assert missed.level == RiskLevel.MISSED
    assert missed.revision == 2


def test_dwell_beyond_limit_escalates_without_declaring_miss():
    m = monitor()
    # slack 4h is WATCH; long dwell pushes it to AT_RISK
    snap = OperationSnapshot(D + timedelta(hours=6), 0, 30, dwell_elapsed=timedelta(hours=30))
    alert = m.update(connection(), outbound(D + timedelta(hours=12)), snap, [], 1, D)
    assert alert.level == RiskLevel.AT_RISK


def test_candidate_rebookings_generated_ranked_and_capacity_filtered():
    m = monitor()
    early = sailing('FB1', D + timedelta(hours=20), remaining=2)
    full = sailing('FB0', D + timedelta(hours=18), remaining=0, booked=10, capacity=10)
    later = sailing('FB2', D + timedelta(days=2), remaining=5)
    snap = OperationSnapshot(D + timedelta(hours=7), 60, 30)
    alert = m.update(connection(), outbound(D + timedelta(hours=12)), snap,
                     [later, full, early], 2, D)
    assert alert.recoverable
    assert [c.sailing_id for c in alert.candidates] == ['FB1', 'FB2']
    assert [c.rank for c in alert.candidates] == [1, 2]
    assert all(c.remaining_teu >= 2 for c in alert.candidates)


def test_not_recoverable_when_no_fallback_capacity():
    m = monitor()
    full = sailing('FB0', D + timedelta(days=1), remaining=0, booked=10, capacity=10)
    snap = OperationSnapshot(D + timedelta(hours=11), 60, 30)
    alert = m.update(connection(), outbound(D + timedelta(hours=12)), snap, [full], 1, D)
    assert alert.level == RiskLevel.MISSED
    assert not alert.recoverable
    assert alert.candidates == ()


def test_alert_update_never_mutates_confirmed_route():
    g = ConnectionGraph()
    g.add_call(ServiceCall('IN', 'S1', D, D + timedelta(hours=2)))
    g.add_call(ServiceCall('OUT', 'S2', D + timedelta(hours=5), D + timedelta(hours=8)))
    c = Connection('C1', 'IN', 'OUT', timedelta(hours=1))
    g.add_connection(c)
    before = g.slack(c)
    before_edges = g.service_edges()

    m = monitor()
    for hours in (1, 4, 8):
        snap = OperationSnapshot(D + timedelta(hours=hours), 30, 30)
        m.update(c, g.calls['OUT'], snap,
                 [sailing('FB1', D + timedelta(days=1), 2)], 1, D + timedelta(hours=hours))

    assert g.connections == [c]
    assert g.slack(c) == before
    assert g.service_edges() == before_edges


def test_stale_prediction_cannot_overwrite_newer_result():
    m = monitor()
    bad = OperationSnapshot(D + timedelta(hours=11), 60, 30)
    newer = m.update(connection(), outbound(D + timedelta(hours=12)), bad, [], 1,
                     assessed_at=D + timedelta(hours=5))
    assert newer.level == RiskLevel.MISSED

    good = OperationSnapshot(D, 0, 30)
    stale = m.update(connection(), outbound(D + timedelta(hours=12)), good, [], 1,
                     assessed_at=D + timedelta(hours=2))
    assert stale is newer
    current = m.current('C1')
    assert current.level == RiskLevel.MISSED
    assert current.revision == 1
    assert current.assessed_at == D + timedelta(hours=5)

    # same vintage as stored is also discarded; newer vintage advances revision
    m.update(connection(), outbound(D + timedelta(hours=12)), good, [], 1,
             assessed_at=D + timedelta(hours=5))
    assert m.current('C1').revision == 1
    m.update(connection(), outbound(D + timedelta(hours=12)), good, [], 1,
             assessed_at=D + timedelta(hours=6))
    assert m.current('C1').revision == 2
    assert m.current('C1').level == RiskLevel.ON_TIME


def test_contention_allocates_same_slot_consistently():
    out = outbound(D + timedelta(hours=12))
    c1, c2 = connection('C1'), connection('C2')
    fb1 = sailing('FB1', D + timedelta(days=1), remaining=1)
    fb2 = sailing('FB2', D + timedelta(days=2), remaining=1)

    m = monitor()
    missed = m.update(c1, out, OperationSnapshot(D + timedelta(hours=11), 60, 30),
                      [fb1, fb2], 1, D)
    risky = m.update(c2, out, OperationSnapshot(D + timedelta(hours=7), 60, 30),
                     [fb1, fb2], 1, D)
    assert missed.level == RiskLevel.MISSED
    assert risky.level == RiskLevel.AT_RISK

    first = allocate_candidates([risky, missed], [fb1, fb2])
    second = allocate_candidates([missed, risky], [fb1, fb2])
    assert first == second  # input order irrelevant
    assert first['C1'].sailing_id == 'FB1'  # higher severity wins contested slot
    assert first['C2'].sailing_id == 'FB2'


def test_contention_lower_severity_gets_none_when_no_space():
    out = outbound(D + timedelta(hours=12))
    c1, c2 = connection('C1'), connection('C2')
    fb1 = sailing('FB1', D + timedelta(days=1), remaining=1)

    m = monitor()
    missed = m.update(c1, out, OperationSnapshot(D + timedelta(hours=11), 60, 30), [fb1], 1, D)
    risky = m.update(c2, out, OperationSnapshot(D + timedelta(hours=7), 60, 30), [fb1], 1, D)

    assigned = allocate_candidates([risky, missed], [fb1])
    assert assigned['C1'].sailing_id == 'FB1'
    assert assigned['C2'] is None


def test_invalid_policy_and_snapshot_rejected():
    with pytest.raises(Exception):
        MisconnectPolicy(watch_margin=timedelta(hours=1), risk_margin=timedelta(hours=2),
                         dwell_limit=timedelta(hours=24))
    with pytest.raises(Exception):
        OperationSnapshot(D, remaining_moves=-1, moves_per_hour=30)
    with pytest.raises(Exception):
        OperationSnapshot(D, remaining_moves=10, moves_per_hour=0)
