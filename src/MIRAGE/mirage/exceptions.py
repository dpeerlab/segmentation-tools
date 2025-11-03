"""
Assertions and exceptions for Mirage.
"""

import pytest

def assert_metric(target, value, label, *args, **kwargs):
    assert target == pytest.approx(value, *args, **kwargs), \
        f"{label} should be {round(target, 5)}, but got {round(value, 5)}."
