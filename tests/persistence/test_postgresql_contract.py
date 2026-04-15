"""Run persistence contract tests against PostgreSQL backend."""
import os
import pytest
from tests.persistence.conftest import POSTGRES_AVAILABLE
from tests.persistence.contract_tests import PersistenceContractTests


@pytest.mark.postgresql
@pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not available")
class TestPostgreSQLContract(PersistenceContractTests):
    @pytest.fixture
    async def backend(self):
        from persistence.postgresql import PostgreSQLBackend

        dsn = os.environ["POSTGRES_DSN"]
        db = PostgreSQLBackend(dsn=dsn)
        await db.initialize()
        yield db
        await db.close()

    # ------------------------------------------------------------------
    # PostgreSQL-only tests
    # ------------------------------------------------------------------

    async def test_recall_project(self, backend):
        from tests.persistence.builders import make_decision_data, make_session_data
        from tests.persistence.contract_tests import _session, _decision

        s = _session(project_name="recall-test")
        await backend.save_session(s)
        d = _decision(session_id=s["id"], description="unique recall keyword")
        await backend.save_decision(d)
        results = await backend.recall_project("recall-test", query="unique recall keyword")
        assert results is not None

    async def test_search_sessions_with_search_type(self, backend):
        from tests.persistence.contract_tests import _session

        s = _session(project_name="searchable-project")
        await backend.save_session(s)
        results = await backend.search_sessions(
            "searchable", search_type="project", limit=10
        )
        assert isinstance(results, list)
