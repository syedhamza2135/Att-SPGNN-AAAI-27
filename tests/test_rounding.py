"""Tests for largest_remainder_rounding in data/transform_hhcomp_crosstable.py.

The defining property of the largest-remainder method is that the rounded
integer vector sums exactly to the requested target while staying non-negative.
"""
import numpy as np
import pytest

# This module imports torch at the top; skip cleanly if torch is unavailable.
transform = pytest.importorskip(
    "transform_hhcomp_crosstable",
    reason="requires torch (and numpy/pandas)",
)


def test_sum_is_preserved_exactly():
    out = transform.largest_remainder_rounding(np.array([1.4, 1.4, 1.2]), 4)
    assert out.sum() == 4
    assert (out >= 0).all()


def test_zero_input_returns_zeros():
    out = transform.largest_remainder_rounding(np.zeros(5), 10)
    assert out.tolist() == [0, 0, 0, 0, 0]


def test_zero_target_returns_zeros():
    out = transform.largest_remainder_rounding(np.array([1.0, 2.0, 3.0]), 0)
    assert out.tolist() == [0, 0, 0]


def test_exact_integers_are_unchanged():
    out = transform.largest_remainder_rounding(np.array([2.0, 3.0, 5.0]), 10)
    assert out.tolist() == [2, 3, 5]


def test_sum_property_over_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(1, 12))
        values = rng.random(n) * 10.0
        target = int(rng.integers(1, 200))
        out = transform.largest_remainder_rounding(values, target)
        assert out.sum() == target
        assert (out >= 0).all()
