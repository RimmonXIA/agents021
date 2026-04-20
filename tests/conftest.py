
from typing import Any

import pytest

from core.config import settings


@pytest.fixture
def temp_db(tmp_path: Any) -> str:
    """Provides a temporary SQLite database path for tests."""
    db_path = tmp_path / "test_trinity.db"
    # Override settings for tests
    settings.sqlite_db_path = str(db_path)
    return str(db_path)

@pytest.fixture
def temp_lancedb(tmp_path: Any) -> str:
    """Provides a temporary LanceDB directory for tests."""
    db_dir = tmp_path / "test_lancedb"
    settings.lancedb_dir = str(db_dir)
    return str(db_dir)
