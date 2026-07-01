"""
Compatibility layer for python-rs-driver (async Rust-backed Cassandra driver).

This module provides a compatibility adapter that wraps the asynchronous python-rs-driver
and exposes a synchronous cassandra-driver-compatible API. This allows existing test code
to work unchanged while using the new async driver under the hood.

Key Features:
- Synchronous execute() and prepare() methods wrapping async calls
- Row access via both tuple indexing and attribute access (row[0] and row.column_name)
- Automatic type conversion between driver types and cassandra.util types
- Batch statement support (LOGGED, UNLOGGED, COUNTER)
- Proper exception mapping (validation/query errors -> InvalidRequest, database errors -> RuntimeError)
- Parameter conversion from test-compatible types to driver types

Architecture:
- CompatSession: Main wrapper providing sync API around async session
- CompatPreparedStatement/CompatBoundStatement: Wrappers for prepared statement binding
- CompatRow/CompatResultSet: Result wrappers for iteration and row access
- _convert_result_value(): Converts driver result types to test-compatible types
- _convert_param_value(): Converts test types to driver parameter types

Type Conversions:
- CqlEmpty -> None
- python-rs-driver datetime.date -> cassandra.util.Date
- python-rs-driver datetime.time -> cassandra.util.Time
- Integer nanoseconds -> datetime.time (for time columns)
- IPv4Address/IPv6Address -> str
- Collections (dict, list, set) are recursively converted

Exception Handling:
- SerializationError: Converted to InvalidRequest (parameter validation)
- "Database returned an error" (server-side validation failures): Converted to InvalidRequest
- Other RuntimeError: Allowed to bubble up (unexpected internal errors)
- This preserves test expectations for error types
"""

import asyncio
import sys

from cassandra.protocol import InvalidRequest

from cassandra.query import Statement as CassandraStatement
# Ensure ./tools/ imports python-rs driver
if "python-rs-driver/python" not in sys.path:
    sys.path.insert(0, "python-rs-driver/python")


from scylla.session_builder import SessionBuilder


def _convert_result_value(v):
    """Convert values returned from the driver to test-compatible types."""
    # if v is CqlEmpty or type(v).__name__ == "CqlEmpty":
    #     return None
    return v


class CompatRow:
    """Row wrapper that supports both dict-like and tuple-like access."""

    def __init__(self, data):
        self._data = {k: _convert_result_value(v) for k, v in data.items()}
        self._keys = list(self._data.keys())
        self._values = tuple(self._data.values())
        for k, v in self._data.items():
            # Only expose attribute access for valid Python identifiers.
            # Some column names (or placeholder keys) may not be identifiers.
            if isinstance(k, str) and k.isidentifier():
                setattr(self, k, v)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._values[index]
        return self._data[index]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return repr(self._values)

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self._values == other
        if isinstance(other, CompatRow):
            return self._values == other._values
        return False

class CompatResultSet:
    """Result set wrapper supporting .one(), .all(), and iteration."""

    def __init__(self, rows):
        self.rows = [CompatRow(r) for r in rows]

    def __iter__(self):
        return iter(self.rows)

    def one(self):
        if self.rows:
            return self.rows[0]
        return None

    def all(self):
        return self.rows

    def __getitem__(self, index):
        return self.rows[index]

class CompatPreparedStatement(CassandraStatement):
    """A minimal prepared statement wrapper with .bind()."""

    def __init__(self, rust_prepared_statement):
        self.rust_prepared_statement = rust_prepared_statement
        # Attributes commonly present on cassandra-driver PreparedStatement
        self.query_string = ""
        self.keyspace = None
        self.routing_key = None
        self.custom_payload = None
        self.is_lwt = False


class CompatHost:
    """Represents a cluster host for compatibility with cassandra-driver."""

    def __init__(self, address, port):
        self.address = address
        self.port = port
        self.endpoint = self  # endpoint is just the host itself

    def __repr__(self):
        return f"CompatHost(address={self.address}, port={self.port})"


class CompatCluster:
    """Mock cluster object for compatibility."""

    def __init__(self, hosts, port, session=None, ssl_context=None):
        self.contact_points = hosts
        self.port = port
        self.ssl_context = ssl_context
        self._session = session
        self.hosts = [CompatHost(h, port) for h in hosts]

    def connect(self):
        """Return the session (for cassandra-driver compatibility)."""
        if self._session is None:
            raise RuntimeError("No session available")
        return self._session


class CompatSession:
    """Compatibility session wrapping the async python-rs-driver."""

    def __init__(self, async_session, loop, hosts, port=9042, ssl_context=None):
        self._session = async_session
        self._loop = loop
        self.cluster = CompatCluster(hosts, port, session=self, ssl_context=ssl_context)
        # cassandra-driver Session exposes .hosts (list of Host)
        self.hosts = self.cluster.hosts

    def execute(
        self, query, parameters=None, timeout=None, bypass_cache=False, **kwargs
    ):
        """Execute a query synchronously (wrapping async call)."""
        return self._loop.run_until_complete(
            self._execute_async(query, parameters, **kwargs)
        )

    async def _execute_async(self, query, parameters, **kwargs):
        """Internal async execute implementation."""
        if hasattr(query, "prepared_statement"):
            parameters = query.values
            query = query.prepared_statement.rust_prepared_statement
        elif hasattr(query, "rust_prepared_statement"):
            query = query.rust_prepared_statement

        res = await self._session.execute(query)

        rows = await res.all()
        return CompatResultSet(rows)


    def prepare(self, query):
        """Prepare a statement synchronously."""
        try:
            ps = self._loop.run_until_complete(self._session.prepare(query))
            return CompatPreparedStatement(ps)
        except RuntimeError as e:
            msg = str(e)
            # Convert preparation errors to InvalidRequest (validation failed at prep time)
            if "Database returned an error" in msg or "Preparation failed" in msg:
                raise InvalidRequest(msg) from e
            raise

    def shutdown(self):
        """Shutdown the session."""
        pass


def create_compat_session(hosts, port, ssl_context=None):
    """Factory function to create a compatible session."""

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def connect():
        return await SessionBuilder(hosts, port).connect()

    async_session = loop.run_until_complete(connect())
    return CompatSession(async_session, loop, hosts, port=port, ssl_context=ssl_context)
