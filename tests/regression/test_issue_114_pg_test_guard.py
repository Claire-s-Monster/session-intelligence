"""
Regression tests for issue #114: the PostgreSQL suites skipped silently.

Covers the pure guard logic in tests/_pg_guard.py: which database a DSN
resolves to, refusal of the production database, strict mode, and the skip
counter that drives the end-of-run warning.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._pg_guard import (
    DEFAULT_TEST_DSN,
    PG_SKIP_REASON,
    PRODUCTION_DATABASE,
    count_pg_skips,
    database_name,
    production_dsn_reason,
    require_pg,
)

pytestmark = pytest.mark.regression


class TestDatabaseName:
    def test_reads_url_path(self):
        assert database_name("postgresql://localhost/foo", {}) == "foo"

    def test_ignores_trailing_slash(self):
        assert database_name("postgresql://localhost:5432/foo/", {}) == "foo"

    def test_ignores_query_string_when_path_present(self):
        assert database_name("postgresql://localhost/foo?sslmode=disable", {}) == "foo"

    def test_falls_back_to_dbname_query_parameter(self):
        assert database_name("postgresql://localhost/?dbname=bar", {}) == "bar"

    def test_falls_back_to_pgdatabase(self):
        assert database_name("postgresql://localhost", {"PGDATABASE": "baz"}) == "baz"

    def test_falls_back_to_user_name(self):
        assert database_name("postgresql://alice@localhost", {}) == "alice"

    def test_unresolvable_returns_none(self):
        assert database_name("postgresql://localhost", {}) is None


class TestProductionDsnReason:
    @pytest.mark.parametrize(
        "dsn",
        [
            "postgresql://localhost/session_intelligence",
            "postgresql://localhost:5432/session_intelligence/",
            "postgresql://localhost/session_intelligence?sslmode=disable",
            "postgresql://localhost/?dbname=session_intelligence",
        ],
    )
    def test_refuses_production_database(self, dsn):
        reason = production_dsn_reason(dsn, {})
        assert reason is not None
        assert PRODUCTION_DATABASE in reason

    def test_refuses_production_via_pgdatabase(self):
        env = {"PGDATABASE": PRODUCTION_DATABASE}
        assert production_dsn_reason("postgresql://localhost", env) is not None

    def test_refuses_the_servers_own_dsn(self):
        env = {"SESSION_DB_DSN": "postgresql://db.internal/prod_copy"}
        assert production_dsn_reason("postgresql://db.internal/prod_copy/", env) is not None

    def test_allows_default_test_database(self):
        assert production_dsn_reason(DEFAULT_TEST_DSN, {}) is None

    def test_does_not_match_on_prefix(self):
        dsn = "postgresql://localhost/session_intelligence_test"
        assert production_dsn_reason(dsn, {}) is None


class TestRequirePg:
    @pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
    def test_truthy(self, value):
        assert require_pg({"SESSION_TEST_REQUIRE_PG": value})

    @pytest.mark.parametrize("value", ["", "0", "false", "no"])
    def test_falsy(self, value):
        assert not require_pg({"SESSION_TEST_REQUIRE_PG": value})

    def test_unset(self):
        assert not require_pg({})


class TestCountPgSkips:
    @staticmethod
    def _skip(reason):
        return SimpleNamespace(longrepr=("tests/x.py", 1, f"Skipped: {reason}"))

    def test_counts_only_postgresql_skips(self):
        reports = [
            self._skip(PG_SKIP_REASON),
            self._skip("POSTGRES_DSN not set; PostgreSQL not available"),
            self._skip("requires network"),
        ]
        assert count_pg_skips(reports) == 2

    def test_tolerates_non_tuple_longrepr(self):
        reports = [
            SimpleNamespace(longrepr=None),
            SimpleNamespace(longrepr="Skipped: PostgreSQL not available"),
        ]
        assert count_pg_skips(reports) == 1

    def test_every_postgresql_skipif_uses_the_shared_reason(self):
        """A drifted PG skip reason would make the end-of-run warning undercount."""
        this_file = Path(__file__).resolve()
        tests_root = this_file.parents[1]
        offenders = []
        for path in tests_root.rglob("*.py"):
            if path.resolve() == this_file:
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "skipif(" in line and "POSTGRES" in line and PG_SKIP_REASON not in line:
                    offenders.append(f"{path.relative_to(tests_root)}:{lineno}")
        assert not offenders, f"PostgreSQL skip reasons missing {PG_SKIP_REASON!r}: {offenders}"
