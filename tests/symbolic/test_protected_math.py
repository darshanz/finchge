import math

import numpy as np
import pytest

from finchge.symbolic.protected_math import ProtectedMath


def test_div_by_zero_returns_one():
    result = float(ProtectedMath.div(5.0, 0.0))
    assert result == pytest.approx(1.0)


def test_div_normal():
    result = float(ProtectedMath.div(6.0, 3.0))
    assert result == pytest.approx(2.0)


def test_div_near_zero_denominator_treated_as_zero():
    result = float(ProtectedMath.div(1.0, 1e-15))
    assert result == pytest.approx(1.0)


def test_log_negative_returns_finite():
    result = float(ProtectedMath.log(-5.0))
    assert math.isfinite(result)


def test_log_zero_returns_finite():
    result = float(ProtectedMath.log(0.0))
    assert math.isfinite(result)


def test_log_positive_matches_math_log():
    result = float(ProtectedMath.log(1.0))
    # log(1 + LOG_EPS) ≈ 0
    assert abs(result) < 1e-6


def test_sqrt_negative_returns_finite():
    result = float(ProtectedMath.sqrt(-4.0))
    assert result == pytest.approx(2.0)


def test_sqrt_zero():
    result = float(ProtectedMath.sqrt(0.0))
    assert result == pytest.approx(0.0, abs=1e-9)


def test_sqrt_positive():
    result = float(ProtectedMath.sqrt(9.0))
    assert result == pytest.approx(3.0)


def test_aq_formula():
    # aq(a, b) = a / sqrt(1 + b^2)
    a, b = 3.0, 4.0
    expected = a / math.sqrt(1 + b * b)
    result = float(ProtectedMath.aq(a, b))
    assert result == pytest.approx(expected)


def test_aq_b_zero():
    result = float(ProtectedMath.aq(5.0, 0.0))
    assert result == pytest.approx(5.0)


def test_nan_input_handled_not_propagated():
    result = ProtectedMath.div(float("nan"), 2.0)
    # _finite replaces nan with 0.0
    assert math.isfinite(float(result))


def test_array_input_returns_array():
    x = np.array([1.0, -1.0, 0.0])
    result = ProtectedMath.sqrt(x)
    assert result.shape == (3,)
    assert all(math.isfinite(v) for v in result)


def test_add_clips_overflow():
    big = 1e200
    result = float(ProtectedMath.add(big, big))
    assert math.isfinite(result)
    assert result == ProtectedMath.OUTPUT_MAX


def test_pow_zero_exponent():
    result = float(ProtectedMath.pow(5.0, 0.0))
    assert result == pytest.approx(1.0)
