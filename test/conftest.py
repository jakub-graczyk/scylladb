#
# Copyright (C) 2024-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
#

import os

import pytest

from test import TEST_RUNNER
from test.pylib.report_plugin import ReportPlugin


pytest_plugins = []


if TEST_RUNNER == "runpy":
    @pytest.fixture(scope="session")
    def testpy_test() -> None:
        return None
else:
    pytest_plugins.append("test.pylib.runner")


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Top-level driver selection toggle.

    This option is intentionally defined at the top conftest so it's available
    regardless of which test subtree is executed. cqlpy-specific conftest files
    can then read it and decide which CQL session implementation to use.

    You can also set SCYLLA_PYTEST_USE_NEW_DRIVER=1 to enable by default.
    """
    default = os.environ.get("SCYLLA_PYTEST_USE_NEW_DRIVER", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    parser.addoption(
        "--use-old-driver",
        action="store_true",
        default=default,
        help="Use the new python-rs-driver (via compatibility layer) instead of cassandra-driver",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(ReportPlugin())
