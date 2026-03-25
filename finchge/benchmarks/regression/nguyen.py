from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from finchge.benchmarks.base import Benchmark, BenchmarkMetadata
from finchge.benchmarks.registry import register
from finchge.benchmarks.typing import BenchmarkFunctionInfoBase, Range
from finchge.grammar import Grammar

# Type alias for the function type
NguyenFunctionType = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _nguyen1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(x**3 + x**2 + x, dtype=np.float64)


def _nguyen2(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(x**4 + x**3 + x**2 + x, dtype=np.float64)


def _nguyen3(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(x**5 + x**4 + x**3 + x**2 + x, dtype=np.float64)


def _nguyen4(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(x**6 + x**5 + x**4 + x**3 + x**2 + x, dtype=np.float64)


def _nguyen5(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.sin(x**2) * np.cos(x) - 1.0, dtype=np.float64)


def _nguyen6(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.sin(x) + np.sin(x + x**2), dtype=np.float64)


def _nguyen7(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.log(x + 1.0) + np.log(x**2 + 1.0), dtype=np.float64)


def _nguyen8(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.sqrt(x), dtype=np.float64)


def _nguyen9(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 2:
        raise ValueError(f"Expected input with at least 2 columns, got {x.shape[1]}")
    return np.asarray(np.sin(x[:, 0]) + np.sin(x[:, 1] ** 2), dtype=np.float64)


def _nguyen10(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 2:
        raise ValueError(f"Expected input with at least 2 columns, got {x.shape[1]}")
    return np.asarray(2.0 * np.sin(x[:, 0]) * np.cos(x[:, 1]), dtype=np.float64)


def _nguyen11(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 2:
        raise ValueError(f"Expected input with at least 2 columns, got {x.shape[1]}")
    return np.asarray(x[:, 0] ** x[:, 1], dtype=np.float64)


def _nguyen12(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[1] < 2:
        raise ValueError(f"Expected input with at least 2 columns, got {x.shape[1]}")
    return np.asarray(
        x[:, 0] ** 4 - x[:, 0] ** 3 + 0.5 * x[:, 1] ** 2 - x[:, 1], dtype=np.float64
    )


class NguyenFunction(Enum):
    N1 = 1
    N2 = 2
    N3 = 3
    N4 = 4
    N5 = 5
    N6 = 6
    N7 = 7
    N8 = 8
    N9 = 9
    N10 = 10
    N11 = 11
    N12 = 12


class NguyenBenchmark(Benchmark):
    _FUNCTIONS: Dict[NguyenFunction, BenchmarkFunctionInfoBase] = {
        NguyenFunction.N1: {
            "name": "Nguyen-1",
            "equation": "x^3 + x^2 + x",
            "description": "Cubic polynomial",
            "function": _nguyen1,
            "input_dim": 1,
            "output_dim": 1,
            "default_range": (-1, 1),
            "complexity": "Simple",
            "reference": "Nguyen et al. (2009)",
        },
        NguyenFunction.N2: {
            "name": "Nguyen-2",
            "equation": "x^4 + x^3 + x^2 + x",
            "description": "Quartic polynomial",
            "function": _nguyen2,
            "input_dim": 1,
            "output_dim": 1,
            "default_range": (-1, 1),
            "complexity": "Simple",
            "reference": "Nguyen et al. (2009)",
        },
        NguyenFunction.N3: {
            "name": "Nguyen-3",
            "equation": "x^5 + x^4 + x^3 + x^2 + x",
            "description": "Quintic polynomial",
            "function": _nguyen3,
            "input_dim": 1,
            "output_dim": 1,
            "default_range": (-1, 1),
            "complexity": "Simple",
            "reference": "Nguyen et al. (2009)",
        },
        NguyenFunction.N4: {
            "name": "Nguyen-4",
            "equation": "x^6 + x^5 + x^4 + x^3 + x^2 + x",
            "description": "Sextic polynomial",
            "function": _nguyen4,
            "input_dim": 1,
            "output_dim": 1,
            "default_range": (-1, 1),
            "complexity": "Simple",
            "reference": "Nguyen et al. (2009)",
        },
        NguyenFunction.N5: {
            "name": "Nguyen-5",
            "equation": "sin(x^2) * cos(x) - 1",
            "description": "Trigonometric with polynomial",
            "function": _nguyen5,
            "input_dim": 1,
            "output_dim": 1,
            "default_range": (-1, 1),
            "complexity": "Medium",
            "reference": "Nguyen et al. (2009)",
        },
        NguyenFunction.N6: {
            "name": "Nguyen-6",
            "equation": "sin(x) + sin(x + x^2)",
            "description": "Composite trigonometric",
            "function": _nguyen6,
            "input_dim": 1,
            "output_dim": 1,
            "default_range": (-1, 1),
            "complexity": "Medium",
            "reference": "Nguyen et al. (2009)",
        },
        NguyenFunction.N7: {
            "name": "Nguyen-7",
            "equation": "log(x + 1) + log(x^2 + 1)",
            "description": "Logarithmic function",
            "function": _nguyen7,
            "input_dim": 1,
            "output_dim": 1,
            "default_range": (0, 2),  # Note: domain positive for log
            "complexity": "Medium",
            "reference": "Nguyen et al. (2009)",
        },
        NguyenFunction.N8: {
            "name": "Nguyen-8",
            "equation": "sqrt(x)",
            "description": "Square root",
            "function": _nguyen8,
            "input_dim": 1,
            "output_dim": 1,
            "default_range": (0, 4),  # Note: domain non-negative
            "complexity": "Simple",
            "reference": "Nguyen et al. (2009)",
        },
        NguyenFunction.N9: {
            "name": "Nguyen-9",
            "equation": "sin(x) + sin(y^2)",
            "description": "2D trigonometric",
            "function": _nguyen9,
            "input_dim": 2,
            "output_dim": 1,
            "default_range": (0, 1),
            "complexity": "Medium",
            "reference": "Nguyen et al. (2009)",
        },
        NguyenFunction.N10: {
            "name": "Nguyen-10",
            "equation": "2*sin(x)*cos(y)",
            "description": "2D product of sines/cosines",
            "function": _nguyen10,
            "input_dim": 2,
            "output_dim": 1,
            "default_range": (0, 1),
            "complexity": "Medium",
            "reference": "Nguyen et al. (2009)",
        },
        NguyenFunction.N11: {
            "name": "Nguyen-11",
            "equation": "x^y",
            "description": "Exponential function",
            "function": _nguyen11,
            "input_dim": 2,
            "output_dim": 1,
            "default_range": (0, 1),
            "complexity": "Hard",
            "reference": "Nguyen et al. (2009)",
        },
        NguyenFunction.N12: {
            "name": "Nguyen-12",
            "equation": "x^4 - x^3 + 0.5*y^2 - y",
            "description": "2D polynomial",
            "function": _nguyen12,
            "input_dim": 2,
            "output_dim": 1,
            "default_range": (-1, 1),
            "complexity": "Hard",
            "reference": "Nguyen et al. (2009)",
        },
    }

    def __init__(
        self,
        function: Union[NguyenFunction, int, str],
        random_state: Optional[int] = None,
        train_samples: int = 20,
        test_samples: int = 1000,
        x_range: Optional[
            Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]
        ] = None,
        noise_std: Optional[float] = None,
        train_type: str = "uniform",
        test_type: str = "grid",
        **kwargs: Any,
    ) -> None:
        super().__init__(random_state=random_state)

        self.noise_std = noise_std
        self.train_type = train_type
        self.test_type = test_type
        self.train_samples = train_samples
        self.test_samples = test_samples

        # Parse function FIRST
        self.function = self._parse_function(function)
        self.func_info = self._FUNCTIONS[self.function]

        # Set input range (before validation)
        self.x_range: Range
        if x_range is None:
            self.x_range = self.func_info["default_range"]
        else:
            self.x_range = x_range

        # For 2D functions, range is applied to both dimensions
        # must happen before validation
        if self.func_info["input_dim"] == 2:
            # Check if there is a single range or tuple of ranges
            if isinstance(self.x_range[0], (tuple, list)):
                # Already properly formatted as ((low1, high1), (low2, high2))
                pass
            else:
                # Single range for both dimensions
                low, high = self.x_range
                self.x_range = ((low, high), (low, high))

        # validate parameters
        if train_samples <= 0:
            raise ValueError(f"train_samples must be positive, got {train_samples}")
        if test_samples <= 0:
            raise ValueError(f"test_samples must be positive, got {test_samples}")

        valid_types = ["uniform", "grid", "random"]
        if train_type not in valid_types:
            raise ValueError(
                f"train_type must be one of {valid_types}, got {train_type}"
            )
        if test_type not in valid_types:
            raise ValueError(f"test_type must be one of {valid_types}, got {test_type}")

        # Validate range based on dimension
        if self.func_info["input_dim"] == 1:
            range_1d = cast(Tuple[float, float], self.x_range)
            if range_1d[0] >= range_1d[1]:
                raise ValueError(
                    f"Invalid range: low={range_1d[0]} >= high={range_1d[1]}"
                )
        else:  # input dim == 2
            range_2d = cast(
                Tuple[Tuple[float, float], Tuple[float, float]], self.x_range
            )
            (low1, high1), (low2, high2) = range_2d
            if low1 >= high1:
                raise ValueError(
                    f"Invalid range for dimension 1: low={low1} >= high={high1}"
                )
            if low2 >= high2:
                raise ValueError(
                    f"Invalid range for dimension 2: low={low2} >= high={high2}"
                )

        # Metadata
        self._metadata = BenchmarkMetadata(
            name=self.func_info["name"],
            category="regression",
            description=f"{self.func_info['description']}: {self.func_info['equation']}",
            reference=self.func_info["reference"],
            input_dim=self.func_info["input_dim"],
            output_dim=self.func_info["output_dim"],
            train_size=train_samples,
            test_size=test_samples,
        )

    def _parse_function(
        self, function: Union[NguyenFunction, int, str]
    ) -> NguyenFunction:
        # already enum -- return directly
        if isinstance(function, NguyenFunction):
            return function

        # Handle integers
        if isinstance(function, int):
            if 1 <= function <= 12:
                return NguyenFunction(function)
            raise ValueError(
                f"Nguyen function number must be between 1-12, got {function}"
            )

        # Handle strings
        if isinstance(function, str):
            # Remove common prefixes and clean up
            clean = function.lower().strip()
            clean = clean.replace("nguyen-", "").replace("nguyen", "")
            clean = clean.replace("n-", "").replace("n", "")
            clean = clean.replace("-", "").strip()

            # Try to parse as number
            try:
                num = int(clean)
                if 1 <= num <= 12:
                    return NguyenFunction(num)
            except ValueError:
                pass

            # Try exact match with enum names
            for member in NguyenFunction:
                if member.name.lower() == function.lower():
                    return member
            for member in NguyenFunction:
                if member.name.lower().replace("n", "") == clean:
                    return member

        raise ValueError(
            f"Invalid Nguyen function identifier: {function}. "
            f"Expected: int 1-12, or string like 'N1', 'Nguyen-1', 'nguyen1', '1'"
        )

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata

    def grammar_str(self) -> str:
        """
        Return grammar string based on function complexity.
        """
        if self.func_info["input_dim"] == 1:
            return self._get_1d_grammar()
        else:
            return self._get_2d_grammar()

    def grammar(self) -> Grammar:
        """
        Return the Grammar object on function complexity.
        """
        return Grammar(grammar_str=self.grammar_str())

    def _get_1d_grammar(self) -> str:
        """Grammar for 1D functions."""
        grammar = """
        <expr> ::= <expr> <op> <expr>
                | <func> ( <expr> )
                | <var>
                | <const>

        <op> ::= + | - | * | /
        """

        # Add protected operators for functions with domain restrictions
        if self.function in [NguyenFunction.N7, NguyenFunction.N8]:
            grammar += """
        <func> ::= sin | cos | exp | plog | psqrt
        <const> ::= 0.1 | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 0.01 | 0.001
            """
        else:
            grammar += """
        <func> ::= sin | cos | exp | log | sqrt
        <const> ::= 0.1 | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0
            """

        grammar += """
        <var> ::= x0
        """

        return grammar

    def _get_2d_grammar(self) -> str:
        return """
        <expr> ::= <expr> <op> <expr>
                | <func> ( <expr> )
                | <var>
                | <const>

        <op> ::= + | - | * | /

        <func> ::= sin | cos | exp | log | sqrt | pow

        <var> ::= x0 | x1

        <const> ::= 0.1 | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0
        """

    def _generate_points(self, n_samples: int, sample_type: str) -> NDArray[np.float64]:
        input_dim = self.func_info["input_dim"]

        if input_dim == 1:
            # 1D case
            range_1d = cast(Tuple[float, float], self.x_range)
            low, high = range_1d

            if sample_type in ["uniform", "random"]:
                points = self.np_rng.uniform(low, high, (n_samples, 1))
                return np.asarray(points, dtype=np.float64)
            elif sample_type == "grid":
                points = np.linspace(low, high, n_samples).reshape(-1, 1)
                return np.asarray(points, dtype=np.float64)
            else:
                raise ValueError(f"Unknown sample type: {sample_type}")

        else:
            # 2D case
            range_2d = cast(
                Tuple[Tuple[float, float], Tuple[float, float]], self.x_range
            )
            (low1, high1), (low2, high2) = range_2d

            if sample_type in ["uniform", "random"]:
                x1 = self.np_rng.uniform(low1, high1, n_samples)
                x2 = self.np_rng.uniform(low2, high2, n_samples)
                points = np.column_stack([x1, x2])
                return np.asarray(points, dtype=np.float64)

            elif sample_type == "grid":
                # Calculate grid size that gives at least n_samples
                n_per_dim = int(np.ceil(np.sqrt(n_samples)))
                x1 = np.linspace(low1, high1, n_per_dim)
                x2 = np.linspace(low2, high2, n_per_dim)
                X1, X2 = np.meshgrid(x1, x2)
                points = np.column_stack([X1.ravel(), X2.ravel()])
                # If more points than needed, randomly sample them
                if len(points) > n_samples:
                    idx = self.np_rng.choice(len(points), n_samples, replace=False)
                    points = points[idx]
                return np.asarray(points, dtype=np.float64)

            else:
                raise ValueError(f"Unknown sample type: {sample_type}")

    def _generate_data(
        self,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        # training points
        X_train = self._generate_points(self.train_samples, self.train_type)
        # test points
        X_test = self._generate_points(self.test_samples, self.test_type)
        # target values
        y_train = self.func_info["function"](X_train)
        y_test = self.func_info["function"](X_test)

        # add noise if specified
        if self.noise_std is not None:
            noise_train = self.np_rng.normal(0, self.noise_std, size=y_train.shape)
            noise_test = self.np_rng.normal(0, self.noise_std, size=y_test.shape)
            y_train = y_train + noise_train
            y_test = y_test + noise_test

        # Ensure 1D arrays for targets
        y_train_flat = y_train.flatten()
        y_test_flat = y_test.flatten()

        return X_train, y_train_flat, X_test, y_test_flat

    def __repr__(self) -> str:
        return f"{self.func_info['name']}(range={self.x_range}, train={self.train_samples})"


@register("Nguyen-1", "regression")
class Nguyen1Benchmark(NguyenBenchmark):
    def __init__(self, random_state: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(NguyenFunction.N1, random_state, **kwargs)


@register("Nguyen-2", "regression")
class Nguyen2Benchmark(NguyenBenchmark):
    def __init__(self, random_state: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(NguyenFunction.N2, random_state, **kwargs)


@register("Nguyen-3", "regression")
class Nguyen3Benchmark(NguyenBenchmark):
    def __init__(self, random_state: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(NguyenFunction.N3, random_state, **kwargs)


@register("Nguyen-4", "regression")
class Nguyen4Benchmark(NguyenBenchmark):
    def __init__(self, random_state: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(NguyenFunction.N4, random_state, **kwargs)


@register("Nguyen-5", "regression")
class Nguyen5Benchmark(NguyenBenchmark):
    def __init__(self, random_state: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(NguyenFunction.N5, random_state, **kwargs)


@register("Nguyen-6", "regression")
class Nguyen6Benchmark(NguyenBenchmark):
    def __init__(self, random_state: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(NguyenFunction.N6, random_state, **kwargs)


@register("Nguyen-7", "regression")
class Nguyen7Benchmark(NguyenBenchmark):
    def __init__(self, random_state: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(NguyenFunction.N7, random_state, **kwargs)


@register("Nguyen-8", "regression")
class Nguyen8Benchmark(NguyenBenchmark):
    def __init__(self, random_state: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(NguyenFunction.N8, random_state, **kwargs)


@register("Nguyen-9", "regression")
class Nguyen9Benchmark(NguyenBenchmark):
    def __init__(self, random_state: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(NguyenFunction.N9, random_state, **kwargs)


@register("Nguyen-10", "regression")
class Nguyen10Benchmark(NguyenBenchmark):
    def __init__(self, random_state: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(NguyenFunction.N10, random_state, **kwargs)


@register("Nguyen-11", "regression")
class Nguyen11Benchmark(NguyenBenchmark):
    def __init__(self, random_state: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(NguyenFunction.N11, random_state, **kwargs)


@register("Nguyen-12", "regression")
class Nguyen12Benchmark(NguyenBenchmark):
    def __init__(self, random_state: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(NguyenFunction.N12, random_state, **kwargs)
