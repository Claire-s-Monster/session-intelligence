"""Run persistence contract tests against SQLite backend."""
import pytest
from tests.persistence.contract_tests import PersistenceContractTests


class TestSQLiteContract(PersistenceContractTests):
    @pytest.fixture
    async def backend(self, tmp_path):
        from persistence.sqlite import SQLiteBackend

        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db
        await db.close()
