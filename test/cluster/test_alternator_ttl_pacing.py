#
# Copyright (C) 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
#

"""Tests for the pacing of Alternator's TTL expiration scan.

The expiration scanner used to run each pass at full speed and then idle until
the next period began. It now spreads the work evenly instead: before a pass it
counts the tablets this shard owns across all TTL-enabled tables, and gives each
tablet a slot in the period, sleeping between tablets to stay on schedule.

These tests check the two properties that follow from that:
  - the deletions really are spread across the period, not done in one burst,
  - the pass still finishes inside the period.

Pacing only applies to tables using tablets. The scanner counts tablets, and
vnode tables deliberately contribute nothing to the count and never tick the
pacer, so they are scanned unpaced and are not covered here.
"""

import asyncio
import logging
import time

from test.cluster.test_alternator import alternator_config, get_alternator, unique_table_name
from test.pylib.manager_client import ManagerClient

logger = logging.getLogger(__name__)

# How long a full expiration scan should take. Everything below is measured
# relative to this. It needs to be long enough that a paced scan is obviously
# different from an unpaced one - unpaced, this much data is deleted in well
# under a second - but short enough to keep the test bearable. The test waits
# up to one period for a scan to begin and then about 0.9 of a period for it to
# finish, so budget roughly two periods of wall time.
TTL_PERIOD = 60

# The scanner paces one tablet at a time, so the tablet count is the number of
# steps the deletions are spread over. 32 tablets across 90% of a 60s period is
# one step every ~1.7 seconds, which is easily visible at our sampling rate.
TABLETS = 32

# Enough items that every tablet holds a good handful, so the deletions form a
# visible ramp rather than a few isolated jumps.
ITEMS = 400

# How often we sample the deletion counter.
SAMPLE_INTERVAL = 0.5

paced_ttl_config = alternator_config | {
    'alternator_ttl_period_in_seconds': str(TTL_PERIOD),
}


async def get_expiration_metric(manager: ManagerClient, ip: str, name: str) -> float:
    """Read one of the expiration service's metrics.

    These metrics are registered with skip_when_empty, so they are simply
    absent from /metrics until they become non-zero, and get() returns None.
    """
    metrics = await manager.metrics.query(ip)
    return metrics.get(name) or 0


def create_tablet_table(alternator, tablets: int):
    """Create an Alternator table backed by a known number of tablets.

    TTL is deliberately *not* enabled here - see the comment in the test.
    """
    return alternator.create_table(TableName=unique_table_name(),
        BillingMode='PAY_PER_REQUEST',
        Tags=[{'Key': 'system:initial_tablets', 'Value': str(tablets)}],
        KeySchema=[{'AttributeName': 'p', 'KeyType': 'HASH'}],
        AttributeDefinitions=[{'AttributeName': 'p', 'AttributeType': 'N'}])


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
    table = create_tablet_table(alternator, TABLETS)
    try:
        # Write every item already expired, then enable TTL *afterwards*.
        # Order matters: the scanner ignores tables without the TTL tag, so
        # until we enable it no pass can look at this table. That keeps the
        # (potentially slow) write out of the window we are measuring - the
        # first pass to see the table sees all ITEMS items at once.
        expiration = int(time.time()) - 3600
        with table.batch_writer() as batch:
            for p in range(ITEMS):
                batch.put_item(Item={'p': p, 'expiration': expiration})
        assert table.scan(ConsistentRead=True, Select='COUNT')['Count'] == ITEMS

        baseline = await get_expiration_metric(manager, server.ip_addr,
                                               'scylla_expiration_items_deleted')
        table.meta.client.update_time_to_live(TableName=table.name,
            TimeToLiveSpecification={'AttributeName': 'expiration', 'Enabled': True})

        # Follow the deletions. Each sample is (seconds since we started
        # watching, items deleted since the baseline). We may wait up to one
        # full period before the next pass even begins, then about 0.9 of a
        # period for it to work through the tablets.
        samples: list[tuple[float, float]] = []
        started = time.time()
        deadline = started + 3 * TTL_PERIOD
        while time.time() < deadline:
            deleted = await get_expiration_metric(manager, server.ip_addr,
                                                  'scylla_expiration_items_deleted') - baseline
            samples.append((time.time() - started, deleted))
            if deleted >= ITEMS:
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


async def test_ttl_scanner_keeps_running_with_no_work(manager: ManagerClient):
    """With no TTL-enabled tables the pacer is handed zero work units.

       It has to switch itself off in that case rather than divide by zero or
       sleep forever, and the scanner has to keep completing passes. This is
       the common case on real clusters - most of them have no TTL tables at
       all - so it is worth its own test.

       This one uses the default half-second period from alternator_config, so
       it finishes in a few seconds.
    """
    server = await manager.server_add(config=alternator_config, cmdline=['--smp=1'])

    before = await get_expiration_metric(manager, server.ip_addr,
                                         'scylla_expiration_scan_passes')
    # Many passes fit in this window at a 0.5s period.
    await asyncio.sleep(5)
    after = await get_expiration_metric(manager, server.ip_addr,
                                        'scylla_expiration_scan_passes')

    assert after > before, (
        f"the expiration scanner stopped making passes ({before} -> {after}) "
        "when there were no TTL-enabled tables to scan")
