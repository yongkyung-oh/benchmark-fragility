import json
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture
def data_dir():
    return DATA


@pytest.fixture
def expected():
    return json.loads((DATA / "expected.json").read_text())
