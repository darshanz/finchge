from __future__ import annotations

from typing import Final, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ScalarOrArray = Union[float, ArrayLike]


class ProtectedMath:
    """
    Real-valued numerical helpers for symbolic regression.
    """

    # Small thresholds

    # treat denominators below this as zero
    DIV_EPS: Final[float] = 1e-12
    # avoids log(0)
    LOG_EPS: Final[float] = 1e-12
    # checking exponent for integer
    INTEGER_TOL: Final[float] = 1e-10
    # nudge away from tan poles
    TAN_POLE_EPS: Final[float] = 1e-8

    # Chosen to stay comfortably inside float64 limits.
    EXP_MIN: Final[float] = -700.0  # prevents underflow to 0 in exp
    EXP_MAX: Final[float] = 700.0  # prevents overflow in exp

    # avoid extreme shrink in power
    POW_EXP_MIN: Final[float] = -50.0
    # avoid extreme growth in power
    POW_EXP_MAX: Final[float] = 50.0

    # keeps sinh/cosh in range
    HYPERBOLIC_ARG_LIMIT: Final[float] = 50.0

    # hard floor for outputs
    OUTPUT_MIN: Final[float] = -1e100
    # hard ceiling for outputs
    OUTPUT_MAX: Final[float] = 1e100
    # reduce large trig arguments
    TRIG_REDUCTION_LIMIT: Final[float] = 1e6

    @classmethod
    def _as_float_array(cls, x: ScalarOrArray) -> FloatArray:
        return np.asarray(x, dtype=np.float64)

    @classmethod
    def _finite(
        cls,
        x: ScalarOrArray,
        *,
        fill_pos: float | None = None,
        fill_neg: float | None = None,
    ) -> FloatArray:
        arr = cls._as_float_array(x)
        if fill_pos is None:
            fill_pos = cls.OUTPUT_MAX
        if fill_neg is None:
            fill_neg = cls.OUTPUT_MIN
        arr = np.nan_to_num(arr, nan=0.0, posinf=fill_pos, neginf=fill_neg)
        return np.clip(arr, cls.OUTPUT_MIN, cls.OUTPUT_MAX)

    @classmethod
    def _broadcast_pair(
        cls, a: ScalarOrArray, b: ScalarOrArray
    ) -> tuple[FloatArray, FloatArray]:
        a_arr, b_arr = np.broadcast_arrays(
            cls._as_float_array(a),
            cls._as_float_array(b),
        )
        return a_arr, b_arr

    @classmethod
    def _reduce_trig_argument(cls, x: ScalarOrArray) -> FloatArray:
        x_arr = cls._as_float_array(x)
        large = np.abs(x_arr) > cls.TRIG_REDUCTION_LIMIT
        if np.any(large):
            period = 2.0 * np.pi
            x_arr = np.where(large, np.remainder(x_arr, period), x_arr)
        return x_arr

    @classmethod
    def add(cls, a: ScalarOrArray, b: ScalarOrArray) -> FloatArray:
        return cls._finite(cls._as_float_array(a) + cls._as_float_array(b))

    @classmethod
    def sub(cls, a: ScalarOrArray, b: ScalarOrArray) -> FloatArray:
        return cls._finite(cls._as_float_array(a) - cls._as_float_array(b))

    @classmethod
    def mul(cls, a: ScalarOrArray, b: ScalarOrArray) -> FloatArray:
        return cls._finite(cls._as_float_array(a) * cls._as_float_array(b))

    @classmethod
    def div(cls, a: ScalarOrArray, b: ScalarOrArray) -> FloatArray:
        """
        Protected division.
        This is a common GP convention and keeps the operator bounded.
        """
        num, den = cls._broadcast_pair(a, b)
        out = np.ones_like(num, dtype=np.float64)
        safe = np.abs(den) >= cls.DIV_EPS
        np.divide(num, den, out=out, where=safe)
        return cls._finite(out)

    @classmethod
    def inv(cls, x: ScalarOrArray) -> FloatArray:
        return cls.div(1.0, x)

    @classmethod
    def exp(cls, x: ScalarOrArray) -> FloatArray:
        x_arr = np.clip(cls._as_float_array(x), cls.EXP_MIN, cls.EXP_MAX)
        return cls._finite(np.exp(x_arr))

    @classmethod
    def log(cls, x: ScalarOrArray) -> FloatArray:
        """
        Protected natural log.
        the operator stays real-valued everywhere.
        """
        x_arr = np.abs(cls._as_float_array(x)) + cls.LOG_EPS
        return cls._finite(np.log(x_arr))

    @classmethod
    def log1p_abs(cls, x: ScalarOrArray) -> FloatArray:
        # Often behaves a bit better near zero than plain log.
        x_arr = np.abs(cls._as_float_array(x))
        return cls._finite(np.log1p(x_arr))

    @classmethod
    def sqrt(cls, x: ScalarOrArray) -> FloatArray:
        return cls._finite(np.sqrt(np.abs(cls._as_float_array(x))))

    @classmethod
    def psqrt(cls, x: ScalarOrArray) -> FloatArray:
        return cls.sqrt(x)

    @classmethod
    def square(cls, x: ScalarOrArray) -> FloatArray:
        x_arr = cls._as_float_array(x)
        return cls._finite(x_arr * x_arr)

    @classmethod
    def cube(cls, x: ScalarOrArray) -> FloatArray:
        x_arr = cls._as_float_array(x)
        return cls._finite(x_arr * x_arr * x_arr)

    @classmethod
    def sin(cls, x: ScalarOrArray) -> FloatArray:
        return cls._finite(np.sin(cls._reduce_trig_argument(x)))

    @classmethod
    def cos(cls, x: ScalarOrArray) -> FloatArray:
        return cls._finite(np.cos(cls._reduce_trig_argument(x)))

    @classmethod
    def tan(cls, x: ScalarOrArray) -> FloatArray:
        """
        Protected tangent. Nudge the argument slightly instead of letting it blow up.
        """
        x_arr = cls._reduce_trig_argument(x)
        shifted = np.remainder(x_arr - 0.5 * np.pi, np.pi)
        dist_to_pole = np.minimum(np.abs(shifted), np.pi - np.abs(shifted))
        near_pole = dist_to_pole < cls.TAN_POLE_EPS
        x_safe = np.where(near_pole, x_arr + cls.TAN_POLE_EPS, x_arr)
        return cls._finite(np.tan(x_safe))

    @classmethod
    def asin(cls, x: ScalarOrArray) -> FloatArray:
        return cls._finite(np.arcsin(np.clip(cls._as_float_array(x), -1.0, 1.0)))

    @classmethod
    def acos(cls, x: ScalarOrArray) -> FloatArray:
        return cls._finite(np.arccos(np.clip(cls._as_float_array(x), -1.0, 1.0)))

    @classmethod
    def atan(cls, x: ScalarOrArray) -> FloatArray:
        return cls._finite(np.arctan(cls._as_float_array(x)))

    @classmethod
    def sinh(cls, x: ScalarOrArray) -> FloatArray:
        x_arr = np.clip(
            cls._as_float_array(x),
            -cls.HYPERBOLIC_ARG_LIMIT,
            cls.HYPERBOLIC_ARG_LIMIT,
        )
        return cls._finite(np.sinh(x_arr))

    @classmethod
    def cosh(cls, x: ScalarOrArray) -> FloatArray:
        x_arr = np.clip(
            cls._as_float_array(x),
            -cls.HYPERBOLIC_ARG_LIMIT,
            cls.HYPERBOLIC_ARG_LIMIT,
        )
        return cls._finite(np.cosh(x_arr))

    @classmethod
    def tanh(cls, x: ScalarOrArray) -> FloatArray:
        x_arr = np.clip(
            cls._as_float_array(x),
            -cls.HYPERBOLIC_ARG_LIMIT,
            cls.HYPERBOLIC_ARG_LIMIT,
        )
        return cls._finite(np.tanh(x_arr))

    @classmethod
    def asinh(cls, x: ScalarOrArray) -> FloatArray:
        return cls._finite(np.arcsinh(cls._as_float_array(x)))

    @classmethod
    def acosh(cls, x: ScalarOrArray) -> FloatArray:
        x_arr = np.maximum(cls._as_float_array(x), 1.0)
        return cls._finite(np.arccosh(x_arr))

    @classmethod
    def atanh(cls, x: ScalarOrArray) -> FloatArray:
        x_arr = np.clip(cls._as_float_array(x), -1.0 + cls.LOG_EPS, 1.0 - cls.LOG_EPS)
        return cls._finite(np.arctanh(x_arr))

    @classmethod
    def abs(cls, x: ScalarOrArray) -> FloatArray:
        return cls._finite(np.abs(cls._as_float_array(x)))

    @classmethod
    def sign(cls, x: ScalarOrArray) -> FloatArray:
        return cls._finite(np.sign(cls._as_float_array(x)))

    @classmethod
    def aq(cls, a: ScalarOrArray, b: ScalarOrArray) -> FloatArray:
        """
        Analytic quotient: a / sqrt(1 + b^2)
        """
        num, den = cls._broadcast_pair(a, b)
        return cls._finite(num / np.sqrt(1.0 + den * den))

    @classmethod
    def pow(cls, a: ScalarOrArray, b: ScalarOrArray) -> FloatArray:
        """
        Protected real-valued power.
        In symbolic regression that is usually a better tradeoff than allowing
        invalid values or silently drifting into complex arithmetic.
        """
        base, exp = cls._broadcast_pair(a, b)
        exp = np.clip(exp, cls.POW_EXP_MIN, cls.POW_EXP_MAX)

        exp_round = np.round(exp)
        is_integer = np.abs(exp - exp_round) <= cls.INTEGER_TOL

        out = np.empty_like(base, dtype=np.float64)

        if np.any(is_integer):
            out[is_integer] = np.power(
                base[is_integer], exp_round[is_integer].astype(np.int64)
            )

        if np.any(~is_integer):
            frac = ~is_integer
            out[frac] = np.power(np.abs(base[frac]) + cls.LOG_EPS, exp[frac])

        return cls._finite(out)


pmath = ProtectedMath()

__all__ = ["ProtectedMath", "pmath"]
