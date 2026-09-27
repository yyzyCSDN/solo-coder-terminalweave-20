from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from terminalweave.core.errors import ConflictError, ValidationError, NotFoundError
from terminalweave.transshipment.connections import ServiceCall, Connection, ConnectionGraph
from terminalweave.transshipment.rebooking import Sailing
from terminalweave.transshipment.alerts import (
    ConnectionTelemetry,
    MissedConnectionMonitor,
    RiskLevel,
    RiskThresholds,
)

UTC = timezone.utc
D = datetime(2026, 6, 1, tzinfo=UTC)


def build_graph(min_transfer=timedelta(hours=4)):
    g = ConnectionGraph()
    g.add_call(ServiceCall('IN', 'FE1', D, D + timedelta(hours=2)))
    g.add_call(ServiceCall('OUT', 'FE2', D + timedelta(hours=10), D + timedelta(hours=12)))
    g.add_call(ServiceCall('OUT2', 'FE3', D + timedelta(days=2), D + timedelta(days=2, hours=2)))
    conn = Connection('C1', 'IN', 'OUT', min_transfer)
    g.add_connection(conn)
    return g, conn


def catalog():
    return [
        Sailing('S1', 'FE2', 'POD', D + timedelta(days=1), 100, 0),
        Sailing('S2', 'FE3', 'POD', D + timedelta(days=2), 10, 0),
    ]


def monitor(graph=None, thresholds=None):
    g, _ = graph or build_graph()
    return MissedConnectionMonitor(g, thresholds=thresholds)


# -- risk evaluation --------------------------------------------------------

def test_healthy_connection_is_ok():
    m = monitor()
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=1), teu=1, destination='POD'))
    w = m.evaluate('C1')
    assert w.level == RiskLevel.OK
    assert not w.at_risk and not w.misses_connection
    # discharge ready D+4, yard ready (D+1)-dwell0+4 = D+5 -> projected D+5
    assert w.projected_ready == D + timedelta(hours=5)
    assert w.projected_slack == timedelta(hours=7)
    assert m.proposal('C1') is None


def test_arrival_deviation_drives_risk_levels():
    m = monitor()
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=1),
                                  arrival_deviation=timedelta(hours=5),
                                  destination='POD'))
    w = m.evaluate('C1')
    # discharge ready D+5h (0+5 dev); yard ready D+1+4 = D+5 -> projected D+5
    assert w.projected_ready == D + timedelta(hours=5)
    assert w.projected_slack == timedelta(hours=7)
    assert w.level == RiskLevel.WATCH
    assert w.misses_connection is False
    assert dict(w.factors)['arrival_deviation'] == pytest.approx(0.75)

    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=2),
                                  arrival_deviation=timedelta(hours=13),
                                  dwell=timedelta(hours=1),
                                  destination='POD'))
    w2 = m.evaluate('C1')
    assert w2.misses_connection is True
    assert w2.level == RiskLevel.CRITICAL
    assert w2.projected_slack == timedelta(hours=-1)


def test_work_remaining_and_dwell_are_separate_factors():
    thresholds = RiskThresholds(
        arrival_tolerance=timedelta(hours=2),
        work_tolerance=timedelta(hours=1),
        dwell_allowance=timedelta(hours=12),
        watch_slack=timedelta(hours=6),
    )
    m = monitor(thresholds=thresholds)
    m.observe(ConnectionTelemetry(
        'C1', D + timedelta(days=1),
        work_remaining=timedelta(hours=4),
        dwell=timedelta(hours=20),
        destination='POD',
    ))
    w = m.evaluate('C1')
    labels = dict(w.factors)
    assert 'work_remaining' in labels and 'dwell' in labels
    # work: (4-1)/(6-1)=0.6 ; dwell excess 8h / 6h -> capped 1.0
    assert labels['work_remaining'] == pytest.approx(0.6)
    assert labels['dwell'] == pytest.approx(1.0)
    assert w.score == pytest.approx(1.0)


def test_connecting_voyage_headroom_flags_capacity():
    m = monitor()
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=1),
                                  connecting_remaining_teu=0,
                                  destination='POD'))
    w = m.evaluate('C1')
    assert w.level == RiskLevel.CRITICAL
    assert dict(w.factors)['voyage_headroom'] == 1.0

    m2 = monitor()
    m2.observe(ConnectionTelemetry('C1', D + timedelta(hours=1),
                                   connecting_remaining_teu=10,
                                   destination='POD'))
    w2 = m2.evaluate('C1')
    # between teu (1) and floor (20): partial factor, watch-level
    assert 0.0 < dict(w2.factors)['voyage_headroom'] < 1.0
    assert w2.level >= RiskLevel.WATCH


def test_evaluate_all_returns_one_warning_per_container():
    g, _ = build_graph()
    g.add_call(ServiceCall('IN2', 'FE1', D + timedelta(hours=1), D + timedelta(hours=3)))
    g.add_connection(Connection('C2', 'IN2', 'OUT2', timedelta(hours=4)))
    m = MissedConnectionMonitor(g)
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=1), destination='POD'))
    m.observe(ConnectionTelemetry('C2', D + timedelta(hours=2),
                                  arrival_deviation=timedelta(hours=6),
                                  destination='POD'))
    warnings = m.evaluate_all()
    assert {w.container_id for w in warnings} == {'C1', 'C2'}
    assert {w.container_id for w in m.at_risk()} == {'C2'}


# -- warning updates never touch confirmed routes ---------------------------

def test_warning_updates_do_not_mutate_confirmed_connections():
    g, conn = build_graph()
    m = MissedConnectionMonitor(g)
    snapshot = list(g.connections)
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=1),
                                  arrival_deviation=timedelta(hours=20),
                                  destination='POD'))
    m.load_sailing_catalog(catalog(), D + timedelta(hours=1), version=1)
    m.evaluate('C1')
    m.allocate_all()
    assert g.connections == snapshot
    assert g.connections[0].outbound_call == 'OUT'


# -- stale predictions ------------------------------------------------------

def test_stale_telemetry_cannot_overwrite_newer_result():
    m = monitor()
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=5),
                                  arrival_deviation=timedelta(hours=5),
                                  destination='POD'))
    m.evaluate('C1')
    with pytest.raises(ConflictError):
        m.observe(ConnectionTelemetry('C1', D + timedelta(hours=2),
                                     arrival_deviation=timedelta(),
                                     destination='POD'))
    # latest observation (deviation 5h) still stands
    assert m.warnings['C1'].observed_at == D + timedelta(hours=5)


def test_stale_sailing_catalog_is_rejected():
    m = monitor()
    m.load_sailing_catalog(catalog(), D + timedelta(hours=4), version=2)
    with pytest.raises(ConflictError):
        m.load_sailing_catalog(catalog(), D + timedelta(hours=2), version=1)
    assert m._catalog_version == 2


def test_old_proposals_do_not_contest_new_catalog_slots():
    m = monitor()
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=1),
                                  arrival_deviation=timedelta(hours=5),
                                  destination='POD'))
    m.load_sailing_catalog(catalog(), D + timedelta(hours=1), version=1)
    m.evaluate('C1')
    assert m.proposal('C1') is not None
    # new catalog publication invalidates priced candidates
    m.load_sailing_catalog(catalog(), D + timedelta(hours=2), version=2)
    assert m.proposal('C1') is None
    # a fresh observation paired with the new catalog regenerates candidates
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=2),
                                  arrival_deviation=timedelta(hours=5),
                                  destination='POD'))
    m.evaluate('C1')
    assert m.proposal('C1').catalog_version == 2


# -- candidate generation ---------------------------------------------------

def test_candidates_are_generated_for_at_risk_and_ranked():
    m = monitor()
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=1), teu=2,
                                  arrival_deviation=timedelta(hours=5),
                                  destination='POD'))
    with pytest.raises(ValidationError):
        m.allocate_candidates('S1')  # no catalog loaded
    m.load_sailing_catalog(catalog(), D + timedelta(hours=1), version=1)
    m.evaluate('C1')
    proposal = m.proposal('C1')
    assert [o.sailing_id for o in proposal.options] == ['S1', 'S2']
    # option capacity is a snapshot from the catalog
    assert proposal.options[0].remaining_teu == 100


def test_recovered_connection_withdraws_candidates():
    m = monitor()
    m.load_sailing_catalog(catalog(), D + timedelta(hours=1), version=1)
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=1),
                                  arrival_deviation=timedelta(hours=5),
                                  destination='POD'))
    m.evaluate('C1')
    assert m.proposal('C1') is not None
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=2),
                                  arrival_deviation=timedelta(),
                                  dwell=timedelta(hours=1),
                                  destination='POD'))
    m.evaluate('C1')
    assert m.proposal('C1') is None


# -- consistent allocation ---------------------------------------------------

def test_contested_capacity_is_allocated_consistently():
    g, _ = build_graph()
    g.add_call(ServiceCall('IN2', 'FE1', D + timedelta(hours=1), D + timedelta(hours=3)))
    g.add_connection(Connection('C2', 'IN2', 'OUT', timedelta(hours=4)))
    m = MissedConnectionMonitor(g)
    tight = [Sailing('S1', 'FE3', 'POD', D + timedelta(days=1), 3, 0)]
    m.load_sailing_catalog(tight, D + timedelta(hours=1), version=1)
    # C2 critical with negative slack; C1 critical with zero slack
    m.observe(ConnectionTelemetry('C2', D + timedelta(hours=1), teu=2,
                                  arrival_deviation=timedelta(hours=12),
                                  destination='POD'))
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=1), teu=2,
                                  arrival_deviation=timedelta(hours=12),
                                  destination='POD'))
    m.evaluate_all()
    allocation = m.allocate_candidates('S1')
    assert allocation.assignments == ('C2',)
    assert allocation.allocated_teu == 2
    assert allocation.denied == ('C1',)
    # same result on repeat — allocation is a pure read of the snapshot
    assert m.allocate_candidates('S1') == allocation


def test_allocation_never_sells_same_slot_twice_across_runs():
    m = monitor()
    m.load_sailing_catalog([Sailing('S1', 'FE3', 'POD', D + timedelta(days=1), 1, 0)],
                           D + timedelta(hours=1), version=1)
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=1),
                                  arrival_deviation=timedelta(hours=10),
                                  destination='POD'))
    m.evaluate('C1')
    a1 = m.allocate_candidates('S1')
    a2 = m.allocate_candidates('S1')
    assert a1.assignments == a2.assignments == ('C1',)
    assert a1.allocated_teu == a2.allocated_teu == 1
    assert m.committed_teu('S1') == 1


def test_preview_does_not_commit_and_holds_block_later_candidates():
    g, _ = build_graph()
    g.add_call(ServiceCall('IN2', 'FE1', D + timedelta(hours=1), D + timedelta(hours=3)))
    g.add_connection(Connection('C2', 'IN2', 'OUT', timedelta(hours=4)))
    m = MissedConnectionMonitor(g)
    m.load_sailing_catalog([Sailing('S1', 'FE3', 'POD', D + timedelta(days=1), 2, 0)],
                           D + timedelta(hours=1), version=1)
    m.observe(ConnectionTelemetry('C2', D + timedelta(hours=1), teu=2,
                                  arrival_deviation=timedelta(hours=12),
                                  destination='POD'))
    m.evaluate('C2')
    preview = m.preview_allocation('S1')
    assert preview.assignments == ('C2',)
    assert m.committed_teu('S1') == 0  # preview committed nothing
    m.allocate_candidates('S1')
    # a later, lower-priority candidate cannot take the held TEU
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=2), teu=2,
                                  arrival_deviation=timedelta(hours=8),
                                  destination='POD'))
    m.evaluate('C1')
    after = m.allocate_candidates('S1')
    assert after.assignments == ('C2',)
    assert after.denied == ('C1',)
    # C2's inbound recovers: its candidate is withdrawn and the hold released,
    # freeing the slot for the waiting C1.
    m.observe(ConnectionTelemetry('C2', D + timedelta(hours=3),
                                  arrival_deviation=timedelta(),
                                  dwell=timedelta(hours=1),
                                  destination='POD'))
    m.evaluate('C2')
    m.release_assignment('C2', 'S1')
    refilled = m.allocate_candidates('S1')
    assert refilled.assignments == ('C1',)
    with pytest.raises(NotFoundError):
        m.release_assignment('C1', 'MISSING')


# -- confirmed rebooking is an explicit separate act -------------------------

def test_rebook_connection_applies_only_after_confirmation():
    g, conn = build_graph()
    m = MissedConnectionMonitor(g)
    m.observe(ConnectionTelemetry('C1', D + timedelta(hours=1),
                                  arrival_deviation=timedelta(hours=20),
                                  destination='POD'))
    m.load_sailing_catalog(catalog(), D + timedelta(hours=1), version=1)
    m.evaluate('C1')
    # evaluation alone leaves the confirmed route intact
    assert g.connections[0].outbound_call == 'OUT'
    # operator confirms the proposed leg
    replaced = g.rebook_connection('C1', 'OUT2', timedelta(hours=4))
    assert replaced.outbound_call == 'OUT2'
    # OUT2 departs D+2d02h; ready from arrival D + 4h transfer
    assert g.slack(replaced) == timedelta(days=2) - timedelta(hours=2)

    with pytest.raises(NotFoundError):
        g.rebook_connection('NOPE', 'OUT2', timedelta(hours=4))
    with pytest.raises(ConflictError):
        g.rebook_connection('C1', 'OUT2', timedelta(days=10))
