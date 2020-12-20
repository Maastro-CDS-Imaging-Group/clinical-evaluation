# -*- coding: utf-8 -*-

import pytest
from clinical_evaluation.skeleton import fib

__author__ = "Suraj Pai"
__copyright__ = "Suraj Pai"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
