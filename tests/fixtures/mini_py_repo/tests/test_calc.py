import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calc import add, divide


def test_add_positive():
    assert add(2, 3) == 5


def test_add_negative():
    assert add(-4, -6) == -10


def test_divide_broken():
    if os.environ.get("FAILCAST_FIXTURE_FAIL") == "1":
        assert divide(10, 2) == 999
    assert divide(10, 2) == 5.0


@pytest.mark.skip(reason="v2")
def test_skipped_feature():
    assert add(1, 1) == 2
