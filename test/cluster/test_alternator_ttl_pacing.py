#
# Copyright (C) 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
#

"""Tests for the pacing of Alternator's TTL expiration scan, and for the
statistics the scanner publishes about it.

The expiration scanner used to run each pass at full speed and then idle until
the next period began. It now spreads the work evenly instead: before a pass it
counts the tablets this shard owns across all TTL-enabled tables, and gives each
tablet a slot in the period, sleeping between tablets to stay on schedule.

The first test checks the observable effect - deletions arriving spread out
rather than in one burst. The rest check the numbers the scanner reports about
itself, because those numbers are what an operator will trust when something
goes wrong:

    scylla_expiration_last_scan_estimated_work_units  what the count predicted
    scylla_expiration_last_scan_done_work_units       what the scan really did
    scylla_expiration_last_scan_duration_ms           how long the pass took
    scylla_expiration_last_scan_sleep_ms              how much of that was sleep

These are written to be hard to satisfy by accident. Every test asserts exact
counts we control rather than relating one metric to another, and every "should
be zero" assertion is paired with proof that the scanner really ran - otherwise
a scanner that silently stopped would pass all of them.

Pacing applies only to tables using tablets. Vnode tables deliberately count
for nothing and never tick the pacer, which test_vnode_tables_are_neither_
counted_nor_paced pins down.
"""

import asyncio
import logging
import time

import pytest

from test.cluster.test_alternator import alternator_config, get_alternator, unique_table_name
from test.pylib.manager_client import ManagerClient

logger = logging.getLogger(__name__)

# Period for the pacing test. It needs to be long enough that a paced scan is
# obviously different from an unpaced one - unpaced, this much data is deleted
# in well under a second - but short enough to keep the test bearable.
TTL_PERIOD = 60

# Period for the statistics tests. Those only need the numbers to be
# predictable, not finely resolved in time, so they can run much faster.
STATS_PERIOD = 10

# The scanner paces one tablet at a time, so the tablet count is the number of
# steps the deletions are spread over.
TABLETS = 32

# Two tables with *different* tablet counts, so a bug that reports one table's
# count instead of the sum cannot pass by coincidence.
TABLETS_A = 16
TABLETS_B = 8

ITEMS = 400
SAMPLE_INTERVAL = 0.5

# Gauges. These have no skip_when_empty, so they must always be exported - even
# as zero. Their absence is a failure, not a value; see read_expiration_stats.
GAUGES = (
    'last_scan_duration_ms',
    'last_scan_sleep_ms',
    'last_scan_estimated_work_units',
    'last_scan_done_work_units',
)

# Counters. These do use skip_when_empty, so they are legitimately missing
# until they first become non-zero.
COUNTERS = (
    'scan_passes',
    'scan_table',
    'items_deleted',
)

paced_ttl_config = alternator_config | {'alternator_ttl_period_in_seconds': str(TTL_PERIOD)}
stats_ttl_config = alternator_config | {'alternator_ttl_period_in_seconds': str(STATS_PERIOD)}


async def read_expiration_stats(manager: ManagerClient, ip: str) -> dict[str, float]:
    """Read every expiration metric at once.

    A missing gauge is treated as an error rather than as zero. If a metric is
    renamed or its registration is dropped, `or 0` would turn that into a
    plausible-looking zero and quietly satisfy most of the assertions below.
    """
    metrics = await manager.metrics.query(ip)
    stats = {}
    for name in GAUGES:
        value = metrics.get(f'scylla_expiration_{name}')
        assert value is not None, (
            f"metric scylla_expiration_{name} is not exported at all - it was "
            "renamed, unregistered, or given skip_when_empty")
        stats[name] = value
    for name in COUNTERS:
        stats[name] = metrics.get(f'scylla_expiration_{name}') or 0
    return stats


async def read_items_deleted(manager: ManagerClient, ip: str) -> float:
    """Just the deletion counter, without requiring the pacing gauges.

    The pacing test only cares about when items disappear, so it should not
    fail because a gauge it never looks at was renamed. It uses this rather
    than read_expiration_stats for that reason.
    """
    metrics = await manager.metrics.query(ip)
    return metrics.get('scylla_expiration_items_deleted') or 0


def describe(stats: dict[str, float]) -> str:
    """Compact rendering of the stats, for assertion messages."""
    return ", ".join(f"{k}={v:g}" for k, v in sorted(stats.items()))


async def wait_for_pass(manager: ManagerClient, ip: str, timeout: float) -> dict[str, float]:
    """Wait for one more scan pass to finish, and return the stats it left.

    scan_passes is bumped at the end of a pass, immediately before the gauges
    are written, with no suspension point in between - so once the counter has
    moved, the gauges describe the pass that just finished.
    """
    before = await read_expiration_stats(manager, ip)
    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(0.2)
        stats = await read_expiration_stats(manager, ip)
        if stats['scan_passes'] > before['scan_passes']:
            return stats
    pytest.fail(f"no expiration scan pass completed within {timeout}s. "
                f"Last seen: {describe(before)}")


async def wait_for_settled_pass(manager: ManagerClient, ip: str, planned: int,
                                timeout: float) -> dict[str, float]:
    """Skip transient passes until one reports the expected work estimate.

    Enabling TTL part-way through a pass leaves that pass with a stale estimate,
    so we let the scanner settle first. Only the estimate is used as the settling
    condition - everything else is asserted afterwards on an unconditioned pass,
    so the settling cannot mask a real failure.
    """
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = await wait_for_pass(manager, ip, timeout=timeout)
        if last['last_scan_estimated_work_units'] == planned:
            return last
    pytest.fail(f"the work estimate never settled at {planned}. "
                f"Last pass: {describe(last)}")


def create_table(alternator, initial_tablets, key_type: str = 'N'):
    """Create an Alternator table. initial_tablets is a number of tablets, or
       the string 'none' to request vnodes instead. TTL is not enabled here."""
    return alternator.create_table(TableName=unique_table_name(),
        BillingMode='PAY_PER_REQUEST',
        Tags=[{'Key': 'system:initial_tablets', 'Value': str(initial_tablets)}],
        KeySchema=[{'AttributeName': 'p', 'KeyType': 'HASH'}],
        AttributeDefinitions=[{'AttributeName': 'p', 'AttributeType': key_type}])


def enable_ttl(table, attribute: str = 'expiration'):
    table.meta.client.update_time_to_live(TableName=table.name,
        TimeToLiveSpecification={'AttributeName': attribute, 'Enabled': True})


def write_expired_items(table, count: int):
    expiration = int(time.time()) - 3600
    with table.batch_writer() as batch:
        for p in range(count):
            batch.put_item(Item={'p': p, 'expiration': expiration})


async def test_ttl_scan_is_paced_across_the_period(manager: ManagerClient):
    """The scan should spread its deletions across the period instead of
       deleting everything as fast as it can.

       The cluster is a single node with a single shard on purpose. That way
       every tablet of the table belongs to that one shard, so the pacer has
       exactly TABLETS work units and hands out slots ~TTL_PERIOD*0.9/TABLETS
       apart. With more shards the tablets would be divided between them and
       each shard would pace a smaller, less predictable share.
    """
    server = await manager.server_add(config=paced_ttl_config, cmdline=['--smp=1'])
    # Keep the tablet count fixed for the duration of the test. A merge or split
    # part-way through would change how much work the pacer thinks it has.
    await manager.disable_tablet_balancing()

    alternator = get_alternator(server.ip_addr)
    table = create_table(alternator, TABLETS)
    try:
        # Write every item already expired, then enable TTL *afterwards*.
        # Order matters: the scanner ignores tables without the TTL tag, so
        # until we enable it no pass can look at this table. That keeps the
        # (potentially slow) write out of the window we are measuring - the
        # first pass to see the table sees all ITEMS items at once.
        write_expired_items(table, ITEMS)
        assert table.scan(ConsistentRead=True, Select='COUNT')['Count'] == ITEMS

        baseline = await read_items_deleted(manager, server.ip_addr)
        enable_ttl(table)

        # Follow the deletions. Each sample is (seconds since we started
        # watching, items deleted since the baseline). We may wait up to one
        # full period before the next pass even begins, then about 0.9 of a
        # period for it to work through the tablets.
        samples: list[tuple[float, float]] = []
        started = time.time()
        deadline = started + 3 * TTL_PERIOD
        while time.time() < deadline:
            deleted = await read_items_deleted(manager, server.ip_addr) - baseline
            samples.append((time.time() - started, deleted))
            if samples[-1][1] >= ITEMS:
                break
            await asyncio.sleep(SAMPLE_INTERVAL)

        assert samples[-1][1] >= ITEMS, (
            f"only {samples[-1][1]} of {ITEMS} items were deleted within "
            f"{3 * TTL_PERIOD}s (the TTL period is {TTL_PERIOD}s)")

        first_delete = next(t for t, d in samples if d > 0)
        last_delete = next(t for t, d in samples if d >= ITEMS)
        spread = last_delete - first_delete
        logger.info("deletions ran from %.1fs to %.1fs - spread %.1fs, TTL period %ds",
                    first_delete, last_delete, spread, TTL_PERIOD)

        # The point of the whole feature. Unpaced, all TABLETS tablets are
        # scanned back to back and every item disappears within a second.
        assert spread > 0.25 * TTL_PERIOD, (
            f"all {ITEMS} items were deleted within {spread:.1f}s - the scan was "
            f"not paced (a paced scan should take roughly {0.9 * TTL_PERIOD:.0f}s)")

        # But it still has to fit in the period. The pacer aims at 90% of it and
        # stops sleeping once past that deadline, so a large overshoot means
        # either the work estimate or the deadline check is wrong.
        assert spread < 1.5 * TTL_PERIOD, (
            f"deletions took {spread:.1f}s, far longer than the {TTL_PERIOD}s period")

        # A steady ramp, not two bursts with a gap in between: halfway through
        # the window we should be roughly halfway through the items.
        midpoint = first_delete + spread / 2
        deleted_at_midpoint = max(d for t, d in samples if t <= midpoint)
        fraction = deleted_at_midpoint / ITEMS
        assert 0.15 < fraction < 0.85, (
            f"{fraction:.0%} of the items were already gone at the halfway point, "
            f"so the deletions are not evenly spread. samples={samples}")

        # And the feature still has to do its job.
        assert table.scan(ConsistentRead=True, Select='COUNT')['Count'] == 0
    finally:
        table.delete()


async def test_expiration_stats_report_the_tablets_this_shard_scans(manager: ManagerClient):
    """The reported estimate and the reported work done must both equal the
       number of tablets this shard actually owns.

       Two tables with different tablet counts, so a bug that reports one
       table's count instead of the sum cannot pass by coincidence. Exact
       equality against a number we chose, rather than estimate == done, so a
       pass that scanned nothing cannot satisfy it either.
    """
    server = await manager.server_add(config=stats_ttl_config, cmdline=['--smp=1'])
    await manager.disable_tablet_balancing()

    alternator = get_alternator(server.ip_addr)
    table_a = create_table(alternator, TABLETS_A)
    table_b = create_table(alternator, TABLETS_B)
    try:
        enable_ttl(table_a)
        enable_ttl(table_b)
        expected = TABLETS_A + TABLETS_B

        # Let the estimate settle, then judge the pass *after* that one, so
        # nothing we assert below was also the settling condition.
        await wait_for_settled_pass(manager, server.ip_addr, expected,
                                    timeout=6 * STATS_PERIOD)
        stats = await wait_for_pass(manager, server.ip_addr, timeout=3 * STATS_PERIOD)
        logger.info("settled pass: %s", describe(stats))

        assert stats['last_scan_estimated_work_units'] == expected, (
            f"the count expected {stats['last_scan_estimated_work_units']} tablets "
            f"but this shard owns all {expected} of them ({TABLETS_A}+{TABLETS_B}). "
            f"{describe(stats)}")
        assert stats['last_scan_done_work_units'] == expected, (
            f"the scan ticked the pacer {stats['last_scan_done_work_units']} times "
            f"for {expected} tablets - the counting pass and the scan disagree "
            f"about which tablets this shard owns. {describe(stats)}")

        # A paced pass should occupy most of its period. Unpaced, scanning a few
        # dozen empty tablets takes milliseconds.
        assert 0.5 * STATS_PERIOD * 1000 <= stats['last_scan_duration_ms'] <= 1.3 * STATS_PERIOD * 1000, (
            f"a paced pass should take about {0.9 * STATS_PERIOD:.0f}s, not "
            f"{stats['last_scan_duration_ms'] / 1000:.1f}s. {describe(stats)}")

        # Sleep cannot exceed the pass it happened in. This is the assertion
        # that catches a unit mix-up: nanoseconds landing in a field named _ms
        # would overshoot by a factor of a million.
        assert stats['last_scan_sleep_ms'] <= stats['last_scan_duration_ms'], (
            f"the scan reports sleeping longer than the pass lasted, which is "
            f"impossible - check the duration units. {describe(stats)}")

        # And most of the pass really was sleep, not work.
        assert stats['last_scan_sleep_ms'] >= 0.3 * stats['last_scan_duration_ms'], (
            f"only {stats['last_scan_sleep_ms'] / stats['last_scan_duration_ms']:.0%} "
            f"of the pass was spent asleep, so the pacer is barely pacing. "
            f"{describe(stats)}")
    finally:
        table_a.delete()
        table_b.delete()


async def test_expiration_stats_are_zero_when_no_table_has_ttl(manager: ManagerClient):
    """With no TTL-enabled table, the pacer gets zero work units.

       It must switch itself off rather than divide by zero or sleep forever,
       and the scanner has to keep completing passes. This is the common case
       on real clusters - most have no TTL tables at all.

       The zeros alone would also be satisfied by a scanner that had died, so
       they are only checked on a pass we watched complete, and the pass has to
       be fast - proving it really ran and found nothing.
    """
    server = await manager.server_add(config=stats_ttl_config, cmdline=['--smp=1'])
    alternator = get_alternator(server.ip_addr)
    # A table exists, but TTL is never enabled on it.
    table = create_table(alternator, TABLETS)
    try:
        stats = await wait_for_pass(manager, server.ip_addr, timeout=3 * STATS_PERIOD)
        logger.info("pass with no TTL tables: %s", describe(stats))

        assert stats['last_scan_estimated_work_units'] == 0, (
            f"a table without TTL enabled was counted as work. {describe(stats)}")
        assert stats['last_scan_done_work_units'] == 0, (
            f"the pacer was ticked even though nothing should have been scanned. "
            f"{describe(stats)}")
        assert stats['last_scan_sleep_ms'] == 0, (
            f"the scanner slept with no work to pace. {describe(stats)}")
        assert stats['last_scan_duration_ms'] < 0.25 * STATS_PERIOD * 1000, (
            f"a pass with nothing to scan took {stats['last_scan_duration_ms']}ms. "
            f"{describe(stats)}")

        # The loop has to survive the zero-work case, not just report zeros once.
        later = await wait_for_pass(manager, server.ip_addr, timeout=3 * STATS_PERIOD)
        assert later['scan_passes'] > stats['scan_passes'], (
            "the expiration scanner stopped making passes when it had no work")
    finally:
        table.delete()


async def test_vnode_tables_are_neither_counted_nor_paced(manager: ManagerClient):
    """Vnode tables must contribute nothing to the estimate and must never tick
       the pacer.

       This is the rule that keeps the two sides consistent: count only what you
       tick. Counting a vnode table's ranges without ticking would inflate the
       estimate and throttle the tablet tables for work that never arrives;
       ticking without counting would push work_done past the estimate and
       switch pacing off.

       The zeros are paired with a check that the items really do expire, so a
       vnode table being ignored entirely cannot pass this test.
    """
    server = await manager.server_add(config=stats_ttl_config, cmdline=['--smp=1'])
    alternator = get_alternator(server.ip_addr)
    table = create_table(alternator, 'none')   # non-numeric tag value -> vnodes
    try:
        write_expired_items(table, 20)
        enable_ttl(table)

        # Wait for the items to actually go, which proves the vnode path is
        # scanning this table at all.
        deadline = time.time() + 4 * STATS_PERIOD
        while time.time() < deadline:
            if table.scan(ConsistentRead=True, Select='COUNT')['Count'] == 0:
                break
            await asyncio.sleep(SAMPLE_INTERVAL)
        assert table.scan(ConsistentRead=True, Select='COUNT')['Count'] == 0, (
            "the vnode table's expired items were never deleted, so this test "
            "cannot say anything about how it was counted")

        stats = await wait_for_pass(manager, server.ip_addr, timeout=3 * STATS_PERIOD)
        logger.info("pass with a vnode TTL table: %s", describe(stats))

        assert stats['last_scan_estimated_work_units'] == 0, (
            f"a vnode table was counted as pacer work. Its ranges are never "
            f"ticked, so the estimate would throttle the tablet tables for work "
            f"that never arrives. {describe(stats)}")
        assert stats['last_scan_done_work_units'] == 0, (
            f"a vnode table ticked the pacer. It contributes nothing to the "
            f"estimate, so this pushes work_done past work_to_do and turns "
            f"pacing off. {describe(stats)}")
        assert stats['last_scan_sleep_ms'] == 0, (
            f"the scanner paced a vnode table. {describe(stats)}")
        assert stats['last_scan_duration_ms'] < 0.5 * STATS_PERIOD * 1000, (
            f"the vnode scan was stretched out as if it were paced. {describe(stats)}")
    finally:
        table.delete()


async def test_expiration_stats_expose_an_overestimated_scan(manager: ManagerClient):
    """estimated and done must be able to disagree, and the metrics must show it.

       This is the case the pair exists for. The counting pass only checks for
       the TTL tag; scan_table additionally requires the expiration attribute to
       have a usable type, and skips the table when it does not. That gap is
       deliberate - doing the full check while counting buys little, and an
       overestimate is the harmless direction (the pass simply finishes early).

       Enabling TTL on a String hash key triggers it deterministically:
       update_time_to_live does no type validation, so the tag is written and
       every tablet is counted, but the scanner rejects `utf8` and skips the
       table without scanning a single tablet.

       This is also the only test here that a broken implementation reporting
       done as a copy of estimated could not pass.

       If the counting pass is ever changed to do the full column check, this
       test will fail - and the right response is to update it, because the
       behaviour it pins down will have deliberately changed.
    """
    server = await manager.server_add(config=stats_ttl_config, cmdline=['--smp=1'])
    await manager.disable_tablet_balancing()

    alternator = get_alternator(server.ip_addr)
    # 'p' is the hash key and a String, which is not one of the types the
    # scanner accepts for an expiration time.
    table = create_table(alternator, TABLETS, key_type='S')
    try:
        enable_ttl(table, attribute='p')

        await wait_for_settled_pass(manager, server.ip_addr, TABLETS,
                                    timeout=6 * STATS_PERIOD)
        before = await read_expiration_stats(manager, server.ip_addr)
        stats = await wait_for_pass(manager, server.ip_addr, timeout=3 * STATS_PERIOD)
        logger.info("pass over a table with an unusable TTL column: %s", describe(stats))

        assert stats['last_scan_estimated_work_units'] == TABLETS, (
            f"the counting pass should count all {TABLETS} tablets - it only "
            f"looks for the TTL tag. {describe(stats)}")
        assert stats['last_scan_done_work_units'] == 0, (
            f"the scan should have rejected this table's column type and "
            f"scanned nothing. {describe(stats)}")

        # Proof that the skip happened where we think it did: scan_table is
        # incremented only after the column type check passes.
        assert stats['scan_table'] == before['scan_table'], (
            f"scan_table advanced from {before['scan_table']} to "
            f"{stats['scan_table']}, so the table was not rejected at the "
            f"column type check after all. {describe(stats)}")

        # An overestimate makes the pass end early rather than late. This is the
        # safe direction, and worth pinning down: the opposite would mean the
        # pacer sleeps out a whole period waiting for work that never comes.
        assert stats['last_scan_sleep_ms'] == 0, (
            f"the pacer slept for tablets that were never scanned. {describe(stats)}")
        assert stats['last_scan_duration_ms'] < 0.25 * STATS_PERIOD * 1000, (
            f"an overestimated pass took {stats['last_scan_duration_ms']}ms; it "
            f"should finish immediately, not stretch out. {describe(stats)}")
    finally:
        table.delete()


# Seastar publishes each scheduling group's share count as a per-shard gauge.
# With one shard the sum is simply that shard's value.
SHARES_METRIC = 'scylla_scheduler_shares'

# Whatever main.cc creates the group with. If this ever stops matching the
# config default, the first assertion below says so.
DEFAULT_SHARES = 200


async def read_group_shares(manager: ManagerClient, ip: str, group: str):
    """Share count of one scheduling group, or None if the group does not exist."""
    metrics = await manager.metrics.query(ip)
    return metrics.get(SHARES_METRIC, {'group': group})


async def set_config(manager: ManagerClient, name: str, value: str):
    """Change a live-updatable option through the system.config virtual table.

    This is the path an operator uses, and the one that matters: it goes through
    set_value_on_all_shards, which refuses the write outright unless the option
    was declared liveness::LiveUpdate.
    """
    await manager.get_cql().run_async(
        "UPDATE system.config SET value=%s WHERE name=%s", (value, name))


async def test_ttl_scheduling_group_shares_can_be_changed_at_runtime(manager: ManagerClient):
    """Changing alternator_ttl_scheduling_group_shares must move the group's
       shares without a restart.

       A single shard keeps the metric exact - it is published per shard, so on
       a bigger machine the reading would be a sum over shards.
    """
    server = await manager.server_add(config=stats_ttl_config, cmdline=['--smp=1'])

    before = await read_group_shares(manager, server.ip_addr, 'alternator_ttl')
    assert before is not None, (
        "there is no alternator_ttl scheduling group - main.cc only creates one "
        "when an Alternator port is configured")
    assert before == DEFAULT_SHARES, (
        f"the group started with {before} shares, expected {DEFAULT_SHARES}")

    await set_config(manager, 'alternator_ttl_scheduling_group_shares', '37')

    # The update fans out with smp::invoke_on_all and each shard's observer runs
    # synchronously, so this is quick - but the CQL write returning does not
    # guarantee the metric has been re-scraped yet.
    deadline = time.time() + 30
    while time.time() < deadline:
        shares = await read_group_shares(manager, server.ip_addr, 'alternator_ttl')
        if shares == 37:
            break
        await asyncio.sleep(0.2)
    assert shares == 37, (
        f"alternator_ttl shares are still {shares} after setting the config to 37. "
        "Either the option is not declared liveness::LiveUpdate - in which case "
        "set_value_on_all_shards rejects the write silently - or nothing observes it")

    # And back, so a stuck value cannot masquerade as a working one.
    await set_config(manager, 'alternator_ttl_scheduling_group_shares', str(DEFAULT_SHARES))
    deadline = time.time() + 30
    while time.time() < deadline:
        shares = await read_group_shares(manager, server.ip_addr, 'alternator_ttl')
        if shares == DEFAULT_SHARES:
            break
        await asyncio.sleep(0.2)
    assert shares == DEFAULT_SHARES, (
        f"shares stayed at {shares} when set back to {DEFAULT_SHARES} - the "
        "observer fired once but does not keep following the config")


async def test_ttl_shares_do_not_touch_streaming_when_alternator_is_off(manager: ManagerClient):
    """Without an Alternator port, main.cc points the TTL scheduling group at the
       streaming group instead of creating its own. The expiration service must
       then leave the shares alone - otherwise this option would silently retune
       streaming, repair and bootstrap on every non-Alternator cluster.
    """
    config = {k: v for k, v in stats_ttl_config.items()
              if k not in ('alternator_port', 'alternator_https_port')}
    server = await manager.server_add(config=config, cmdline=['--smp=1'])

    assert await read_group_shares(manager, server.ip_addr, 'alternator_ttl') is None, (
        "an alternator_ttl scheduling group exists even though Alternator is off")
    before = await read_group_shares(manager, server.ip_addr, 'streaming')
    assert before is not None, "no streaming scheduling group to check against"

    await set_config(manager, 'alternator_ttl_scheduling_group_shares', '37')
    await asyncio.sleep(3)   # long enough for any observer to have fired

    after = await read_group_shares(manager, server.ip_addr, 'streaming')
    assert after == before, (
        f"streaming shares moved from {before} to {after} when the Alternator TTL "
        "share count was changed. The expiration service is setting shares on a "
        "group it does not own")
