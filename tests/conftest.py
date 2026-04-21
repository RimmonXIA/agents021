from typing import Any

import pytest

from core.config import settings


@pytest.fixture
def temp_db(tmp_path: Any) -> str:
    """Temporary data dir; returns resolved SQLite path for assertions."""
    settings.data_dir = str(tmp_path)
    return settings.sqlite_db_path


@pytest.fixture
def temp_lancedb(tmp_path: Any) -> str:
    """Same tmp root as temp_db when both are used; returns LanceDB directory path."""
    settings.data_dir = str(tmp_path)
    return settings.lancedb_dir
