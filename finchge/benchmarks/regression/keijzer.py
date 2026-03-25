import warnings
from enum import Enum
from typing import Callable, Dict, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from finchge.benchmarks.base import Benchmark, BenchmarkMetadata
from finchge.benchmarks.registry import register
from finchge.benchmarks.typing import BenchmarkFunctionInfoBase, Range
from finchge.grammar import Grammar


class KeijzerFunction(Enum):
    K1 = 1
    K2 = 2
    K3 = 3
    K4 = 4
    K5 = 5
    K6 = 6
    K7 = 7
    K8 = 8
    K9 = 9
    K10 = 10
    K11 = 11
    K12 = 12
    K13 = 13
    K14 = 14
    K15 = 15


# Type alias for the function type
KeijzerFunctionType = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _keijzer_k1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 1:
        raise ValueError(f"Expected input with at least 1 column, got {x.shape[1]}")
    return np.asarray(0.3 * x[:, 0] * np.sin(2 * np.pi * x[:, 0]), dtype=np.float64)


def _keijzer_k2(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 1:
        raise ValueError(f"Expected input with at least 1 column, got {x.shape[1]}")
    return np.asarray(0.3 * x[:, 0] * np.sin(2 * np.pi * x[:, 0]), dtype=np.float64)


def _keijzer_k3(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 1:
        raise ValueError(f"Expected input with at least 1 column, got {x.shape[1]}")
    return np.asarray(0.3 * x[:, 0] * np.sin(2 * np.pi * x[:, 0]), dtype=np.float64)


def _keijzer_k4(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 1:
        raise ValueError(f"Expected input with at least 1 column, got {x.shape[1]}")

    x_val = x[:, 0]

    # Calculate each term for clarity and numerical stability
    x3 = x_val**3
    exp_neg_x = np.exp(-x_val)
    cos_x = np.cos(x_val)
    sin_x = np.sin(x_val)
    sin2_x = sin_x**2
    sin2_cos = sin2_x * cos_x
    last_term = sin2_cos - 1

    # Product of all terms
    result = x3 * exp_neg_x * cos_x * sin_x * last_term

    return np.asarray(result, dtype=np.float64)


def _keijzer_k5(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 1:
        raise ValueError(f"Expected input with at least 1 column, got {x.shape[1]}")

    x_val = x[:, 0]
    # Protected division for safety
    denominator = (x_val - 2) ** 2
    result = np.zeros_like(x_val, dtype=np.float64)
    mask = np.abs(denominator) > 1e-10
    result[mask] = 30 * (x_val[mask] - 1) * (x_val[mask] - 3) / denominator[mask]
    # At x=2, function tends to infinity
    result[~mask] = np.inf
    return np.asarray(result, dtype=np.float64)


def _keijzer_k6(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 1:
        raise ValueError(f"Expected input with at least 1 column, got {x.shape[1]}")
    return np.asarray(x[:, 0] + np.sin(x[:, 0]), dtype=np.float64)


def _keijzer_k7(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 1:
        raise ValueError(f"Expected input with at least 1 column, got {x.shape[1]}")
    return np.asarray(np.log(x[:, 0]), dtype=np.float64)


def _keijzer_k8(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 1:
        raise ValueError(f"Expected input with at least 1 column, got {x.shape[1]}")
    return np.asarray(np.sqrt(x[:, 0]), dtype=np.float64)


def _keijzer_k9(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 1:
        raise ValueError(f"Expected input with at least 1 column, got {x.shape[1]}")
    return np.asarray(np.arcsinh(x[:, 0]), dtype=np.float64)


def _keijzer_k10(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 2:
        raise ValueError(f"Expected input with at least 2 columns, got {x.shape[1]}")
    return np.asarray(x[:, 0] ** x[:, 1], dtype=np.float64)


def _keijzer_k11(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 2:
        raise ValueError(f"Expected input with at least 2 columns, got {x.shape[1]}")
    return np.asarray(
        x[:, 0] * x[:, 1] + np.sin((x[:, 0] - 1) * (x[:, 1] - 1)), dtype=np.float64
    )


def _keijzer_k12(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 2:
        raise ValueError(f"Expected input with at least 2 columns, got {x.shape[1]}")
    return np.asarray(
        x[:, 0] ** 4 - x[:, 0] ** 3 + 0.5 * x[:, 1] ** 2 - x[:, 1], dtype=np.float64
    )


def _keijzer_k13(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 2:
        raise ValueError(f"Expected input with at least 2 columns, got {x.shape[1]}")
    return np.asarray(6 * np.sin(x[:, 0]) * np.cos(x[:, 1]), dtype=np.float64)


def _keijzer_k14(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 2:
        raise ValueError(f"Expected input with at least 2 columns, got {x.shape[1]}")
    denominator = 2 + x[:, 0] ** 2 + x[:, 1] ** 2
    return np.asarray(8 / denominator, dtype=np.float64)


def _keijzer_k15(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 2:
        raise ValueError(f"Expected input with at least 2 columns, got {x.shape[1]}")
    return np.asarray(
        x[:, 0] ** 3 / 5 + x[:, 1] ** 3 / 2 - x[:, 1] - x[:, 0], dtype=np.float64
    )


class KeijzerBenchmark(Benchmark):
    """
    Keijzer benchmark suite for symbolic regression.

    Keijzer, M., 2003, April. Improving symbolic regression with interval arithmetic and linear scaling.
    In European Conference on Genetic Programming (pp. 70-82). Berlin, Heidelberg: Springer Berlin Heidelberg.

    This implements all 15 functions from Keijzer (2003) with the exact
    training and testing configurations specified in the original paper.
    """

    # Function definitions with their equations, domains, and sampling parameters
    _FUNCTIONS: Dict[KeijzerFunction, BenchmarkFunctionInfoBase] = {
        KeijzerFunction.K1: {
            "name": "Keijzer-1",
            "equation": "0.3x sin(2πx)",
            "description": "Oscillatory function",
            "function": _keijzer_k1,
            "input_dim": 1,
            "output_dim": 1,
            "train_range": (-1, 1),
            "train_step": 0.1,
            "test_range": (-1, 1),
            "test_step": 0.01,
            "complexity": "Simple",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K2: {
            "name": "Keijzer-2",
            "equation": "0.3x sin(2πx)",
            "description": "Oscillatory function (wider range)",
            "function": _keijzer_k2,
            "input_dim": 1,
            "output_dim": 1,
            "train_range": (-2, 2),
            "train_step": 0.1,
            "test_range": (-2, 2),
            "test_step": 0.01,
            "complexity": "Simple",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K3: {
            "name": "Keijzer-3",
            "equation": "0.3x sin(2πx)",
            "description": "Oscillatory function (widest range)",
            "function": _keijzer_k3,
            "input_dim": 1,
            "output_dim": 1,
            "train_range": (-3, 3),
            "train_step": 0.1,
            "test_range": (-3, 3),
            "test_step": 0.01,
            "complexity": "Simple",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K4: {
            "name": "Keijzer-4",
            "equation": "x^3 e^{-x} cos(x) sin(x) (sin^2(x) cos(x) - 1)",
            "description": "Complex product function",
            "function": _keijzer_k4,
            "input_dim": 1,
            "output_dim": 1,
            "train_range": (0, 10),
            "train_step": 0.05,
            "test_range": (0.05, 10.05),
            "test_step": 0.05,
            "complexity": "Hard",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K5: {
            "name": "Keijzer-5",
            "equation": "30 (x-1)(x-3) / (x-2)^2",
            "description": "Rational function with asymptote at x=2",
            "function": _keijzer_k5,
            "input_dim": 1,
            "output_dim": 1,
            "train_range": (0.05, 2),
            "train_step": 0.05,
            "test_range": (0.05, 2),
            "test_step": 0.05,
            "complexity": "Hard",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K6: {
            "name": "Keijzer-6",
            "equation": "x + sin(x)",
            "description": "Linear plus trigonometric",
            "function": _keijzer_k6,
            "input_dim": 1,
            "output_dim": 1,
            "train_range": (-1, 1),
            "train_step": 0.1,
            "test_range": (-1, 1),
            "test_step": 0.01,
            "complexity": "Simple",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K7: {
            "name": "Keijzer-7",
            "equation": "ln(x)",
            "description": "Natural logarithm",
            "function": _keijzer_k7,
            "input_dim": 1,
            "output_dim": 1,
            "train_range": (1, 100),
            "train_step": 1,
            "test_range": (1, 100),
            "test_step": 0.1,
            "complexity": "Simple",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K8: {
            "name": "Keijzer-8",
            "equation": "sqrt(x)",
            "description": "Square root",
            "function": _keijzer_k8,
            "input_dim": 1,
            "output_dim": 1,
            "train_range": (0, 100),
            "train_step": 1,
            "test_range": (0, 100),
            "test_step": 0.1,
            "complexity": "Simple",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K9: {
            "name": "Keijzer-9",
            "equation": "asinh(x)",
            "description": "Inverse hyperbolic sine",
            "function": _keijzer_k9,
            "input_dim": 1,
            "output_dim": 1,
            "train_range": (0, 100),
            "train_step": 1,
            "test_range": (0, 100),
            "test_step": 0.1,
            "complexity": "Medium",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K10: {
            "name": "Keijzer-10",
            "equation": "x^y",
            "description": "Exponential function",
            "function": _keijzer_k10,
            "input_dim": 2,
            "output_dim": 1,
            "train_range": (0, 1),
            "train_step": None,
            "test_range": (0, 1),
            "test_n_points": 1000,
            "complexity": "Medium",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K11: {
            "name": "Keijzer-11",
            "equation": "xy + sin((x-1)(y-1))",
            "description": "Mixed polynomial-trigonometric",
            "function": _keijzer_k11,
            "input_dim": 2,
            "output_dim": 1,
            "train_range": (-3, 3),
            "train_step": None,
            "test_range": (-3, 3),
            "test_n_points": 1000,
            "complexity": "Medium",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K12: {
            "name": "Keijzer-12",
            "equation": "x^4 - x^3 + y^2/2 - y",
            "description": "4th degree polynomial",
            "function": _keijzer_k12,
            "input_dim": 2,
            "output_dim": 1,
            "train_range": (-3, 3),
            "train_step": None,
            "test_range": (-3, 3),
            "test_n_points": 1000,
            "complexity": "Medium",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K13: {
            "name": "Keijzer-13",
            "equation": "6 sin(x) cos(y)",
            "description": "Trigonometric product",
            "function": _keijzer_k13,
            "input_dim": 2,
            "output_dim": 1,
            "train_range": (-3, 3),
            "train_step": None,
            "test_range": (-3, 3),
            "test_n_points": 1000,
            "complexity": "Simple",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K14: {
            "name": "Keijzer-14",
            "equation": "8 / (2 + x^2 + y^2)",
            "description": "Rational function (peak)",
            "function": _keijzer_k14,
            "input_dim": 2,
            "output_dim": 1,
            "train_range": (-3, 3),
            "train_step": None,
            "test_range": (-3, 3),
            "test_n_points": 1000,
            "complexity": "Medium",
            "reference": "Keijzer (2003)",
        },
        KeijzerFunction.K15: {
            "name": "Keijzer-15",
            "equation": "x^3/5 + y^3/2 - y - x",
            "description": "Cubic polynomial",
            "function": _keijzer_k15,
            "input_dim": 2,
            "output_dim": 1,
            "train_range": (-3, 3),
            "train_step": None,
            "test_range": (-3, 3),
            "test_n_points": 1000,
            "complexity": "Medium",
            "reference": "Keijzer (2003)",
        },
    }

    def __init__(
        self,
        function: Union[KeijzerFunction, int, str],
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)

        # Parse function identifier
        self.function = self._parse_function(function)
        self.func_info = self._FUNCTIONS[self.function]

        # Set parameters from function info
        self.input_dim: int = self.func_info["input_dim"]

        # Initialize with default values
        self.train_range: Range
        self.test_range: Range
        self.train_step: Optional[float] = None
        self.test_step: Optional[float] = None
        self.test_n_points: int = 1000

        if self.input_dim == 1:
            self.train_range = cast(Range, self.func_info.get("train_range", (0, 1)))
            self.test_range = cast(Range, self.func_info.get("test_range", (0, 1)))
            self.train_step = self.func_info.get("train_step")
            self.test_step = self.func_info.get("test_step")
        else:
            # 2D functions - same range for both dimensions
            range_val = cast(
                Tuple[float, float], self.func_info.get("train_range", (0, 1))
            )
            # This creates a tuple of tuples
            self.train_range = (range_val, range_val)

            # For test_range, use the test_range from func_info if available
            if "test_range" in self.func_info:
                test_range_val = self.func_info["test_range"]
                self.test_range = (test_range_val, test_range_val)
            else:
                self.test_range = (range_val, range_val)

            self.test_n_points = self.func_info.get("test_n_points", 1000)

        # Create metadata
        self._metadata = BenchmarkMetadata(
            name=self.func_info["name"],
            category="regression",
            description=f"{self.func_info['description']}: {self.func_info['equation']}",
            reference=self.func_info.get("reference", "Keijzer (2003)"),
            input_dim=self.input_dim,
            output_dim=self.func_info["output_dim"],
            train_size=self._get_train_size(),
            test_size=self._get_test_size(),
        )

    def _parse_function(
        self,
        function: Union[KeijzerFunction, int, str],
    ) -> KeijzerFunction:
        """
        Parse function identifier to KeijzerFunction enum.
        """
        if isinstance(function, KeijzerFunction):
            return function
        elif isinstance(function, int):
            if 1 <= function <= 15:
                return KeijzerFunction(function)
            raise ValueError(f"Keijzer function must be between 1-15, got {function}")
        elif isinstance(function, str):
            # Handle strings like "K1", "Keijzer-1", "keijzer1", "1"
            clean = (
                function.lower()
                .replace("keijzer", "")
                .replace("k", "")
                .replace("-", "")
                .strip()
            )
            try:
                num = int(clean)
                if 1 <= num <= 15:
                    return KeijzerFunction(num)
            except ValueError:
                pass

        raise ValueError(f"Invalid Keijzer function identifier: {function}")

    def _get_train_size(self) -> int:
        if self.input_dim == 1:
            if self.train_step is not None:
                low, high = cast(Tuple[float, float], self.train_range)
                return int(round((high - low) / self.train_step)) + 1
            else:
                # Default for 2D functions
                return 100
        else:
            # 2D functions - default to 100 training points
            return 100

    def _get_test_size(self) -> int:
        """Calculate test set size."""
        if self.input_dim == 1:
            if self.test_step is not None:
                low, high = cast(Tuple[float, float], self.test_range)
                return int(round((high - low) / self.test_step)) + 1
            else:
                return 1000
        else:
            return self.test_n_points

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata

    def grammar_str(self) -> str:
        """Return appropriate grammar based on complexity."""
        if self.input_dim == 1:
            return self._get_1d_grammar()
        else:
            return self._get_2d_grammar()

    def grammar(self) -> Grammar:
        """
        Return the Grammar object.
        """
        return Grammar(grammar_str=self.grammar_str())

    def _get_1d_grammar(self) -> str:
        """
        Grammar for 1D functions.
        """
        base_grammar = """
        <expr> ::= <expr> <op> <expr>
                | <func> ( <expr> )
                | <var>
                | <const>

        <op> ::= + | - | * | /

        <func> ::= sin | cos | exp | log | sqrt | abs | sinh | cosh | asinh

        <var> ::= x0

        <const> ::= 0.1 | 0.3 | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 8.0 | 10.0 | 30.0
        """

        return base_grammar

    def _get_2d_grammar(self) -> str:
        """
        Grammar for 2D functions.
        """
        return """
        <expr> ::= <expr> <op> <expr>
                | <func> ( <expr> )
                | <var>
                | <const>

        <op> ::= + | - | * | /

        <func> ::= sin | cos | exp | log | sqrt | abs | sinh | cosh | asinh | pow

        <var> ::= x0 | x1

        <const> ::= 0.1 | 0.3 | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 8.0 | 10.0 | 30.0
        """

    def _generate_train_points(self) -> NDArray[np.float64]:
        """
        Generate training points
        """
        if self.input_dim == 1:
            if self.train_step is not None:
                low, high = cast(Tuple[float, float], self.train_range)
                n_points = int(round((high - low) / self.train_step)) + 1
                points = np.linspace(low, high, n_points, dtype=np.float64).reshape(
                    -1, 1
                )
                return np.asarray(points, dtype=np.float64)
            else:
                low, high = cast(Tuple[float, float], self.train_range)
                train_size = self.metadata.train_size
                if train_size is None:
                    raise ValueError("train_size cannot be None")

                points = self.np_rng.uniform(low, high, (train_size, 1))
                return np.asarray(points, dtype=np.float64)
        else:
            # 2D functions
            range_tuple = cast(
                Tuple[Tuple[float, float], Tuple[float, float]], self.train_range
            )
            (low, high), _ = range_tuple

            # For 2D training, use random sampling
            x1 = self.np_rng.uniform(low, high, self.metadata.train_size)
            x2 = self.np_rng.uniform(low, high, self.metadata.train_size)
            points = np.column_stack([x1, x2])
            return np.asarray(points, dtype=np.float64)

    def _generate_test_points(self) -> NDArray[np.float64]:
        """
        Generate test points
        """
        if self.input_dim == 1:
            if self.test_step is not None:
                low, high = cast(Tuple[float, float], self.test_range)
                n_points = int(round((high - low) / self.test_step)) + 1
                points = np.linspace(low, high, n_points, dtype=np.float64).reshape(
                    -1, 1
                )
                return np.asarray(points, dtype=np.float64)
            else:
                low, high = cast(Tuple[float, float], self.test_range)
                test_size = self.metadata.test_size
                if test_size is None:
                    raise ValueError("test_size cannot be None")

                points = np.linspace(low, high, test_size, dtype=np.float64).reshape(
                    -1, 1
                )
                return np.asarray(points, dtype=np.float64)
        else:
            # 2D functions
            range_tuple = cast(
                Tuple[Tuple[float, float], Tuple[float, float]], self.test_range
            )
            (low, high), _ = range_tuple
            test_size = self.metadata.test_size
            if test_size is None:
                raise ValueError("test_size cannot be None")

            n_per_dim = int(np.ceil(np.sqrt(test_size)))

            # Create grid
            x = np.linspace(low, high, n_per_dim, dtype=np.float64)
            X1, X2 = np.meshgrid(x, x)

            # Stack points
            points = np.column_stack([X1.ravel(), X2.ravel()])

            # If there are too many points, sample them
            if len(points) > test_size:
                idx = self.np_rng.choice(len(points), test_size, replace=False)
                points = points[idx]

            return np.asarray(points, dtype=np.float64)

    def _generate_data(
        self,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        # Generate points
        X_train = self._generate_train_points()
        X_test = self._generate_test_points()

        # Compute target values
        y_train = self._safe_evaluate(self.func_info["function"], X_train)
        y_test = self._safe_evaluate(self.func_info["function"], X_test)

        return X_train, y_train, X_test, y_test

    def _safe_evaluate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Safe evaluation"""
        try:
            y = func(X)

            # Handle infinite values for K5
            if self.function == KeijzerFunction.K5:
                # Replace inf with large finite values
                y = np.where(np.isinf(y), 1e10 * np.sign(y), y)

            return np.asarray(y.flatten(), dtype=np.float64)

        except Exception as e:
            warnings.warn(f"Function evaluation failed: {e}")
            return np.zeros(len(X), dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"{self.func_info['name']}(dim={self.input_dim}, "
            f"train={self.metadata.train_size})"
        )


# subclasses for each Keijzer function
@register("keijzer-1", "regression")
class Keijzer1Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K1, random_state)


@register("keijzer-2", "regression")
class Keijzer2Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K2, random_state)


@register("keijzer-3", "regression")
class Keijzer3Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K3, random_state)


@register("keijzer-4", "regression")
class Keijzer4Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K4, random_state)


@register("keijzer-5", "regression")
class Keijzer5Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K5, random_state)


@register("keijzer-6", "regression")
class Keijzer6Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K6, random_state)


@register("keijzer-7", "regression")
class Keijzer7Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K7, random_state)


@register("keijzer-8", "regression")
class Keijzer8Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K8, random_state)


@register("keijzer-9", "regression")
class Keijzer9Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K9, random_state)


@register("keijzer-10", "regression")
class Keijzer10Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K10, random_state)


@register("keijzer-11", "regression")
class Keijzer11Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K11, random_state)


@register("keijzer-12", "regression")
class Keijzer12Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K12, random_state)


@register("keijzer-13", "regression")
class Keijzer13Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K13, random_state)


@register("keijzer-14", "regression")
class Keijzer14Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K14, random_state)


@register("keijzer-15", "regression")
class Keijzer15Benchmark(KeijzerBenchmark):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(KeijzerFunction.K15, random_state)
