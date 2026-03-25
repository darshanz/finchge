from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from finchge.benchmarks.base import Benchmark, BenchmarkMetadata
from finchge.benchmarks.registry import register
from finchge.grammar import Grammar


@register("koza_quartic", "regression")
class KozaQuarticBenchmark(Benchmark):
    """
    Koza's quartic polynomial benchmark.

    Target function:
        f(x) = x^4 + x^3 + x^2 + x

    Default configuration:
        - Training data: 100 random points in [-1, 1]
        - Test data: 1000 evenly spaced points in [-1, 1]

    Reference:
        Koza, J.R. (1992). *Genetic Programming*.
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        train_samples: int = 100,
        test_samples: int = 1000,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        train_type: Literal["uniform", "grid", "random"] = "uniform",
        test_type: Literal["uniform", "grid", "random"] = "grid",
    ) -> None:
        super().__init__(random_state=random_state)

        if train_samples <= 0:
            raise ValueError(f"train_samples must be positive, got {train_samples}")
        if test_samples <= 0:
            raise ValueError(f"test_samples must be positive, got {test_samples}")
        if x_range[0] >= x_range[1]:
            raise ValueError(f"Invalid range: low={x_range[0]} >= high={x_range[1]}")

        valid_types = {"uniform", "grid", "random"}
        if train_type not in valid_types:
            raise ValueError(
                f"train_type must be one of {valid_types}, got {train_type}"
            )
        if test_type not in valid_types:
            raise ValueError(f"test_type must be one of {valid_types}, got {test_type}")

        self.train_samples = train_samples
        self.test_samples = test_samples
        self.x_range = x_range
        self.train_type = train_type
        self.test_type = test_type

        self._metadata = BenchmarkMetadata(
            name="Koza Quartic",
            category="regression",
            description="Quartic polynomial: x^4 + x^3 + x^2 + x",
            reference="Koza, J.R. (1992). Genetic Programming.",
            input_dim=1,
            output_dim=1,
            train_size=train_samples,
            test_size=test_samples,
        )

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata

    def grammar_str(self) -> str:
        return """
        <expr> ::= <expr> <op> <expr>
                | <func> ( <expr> )
                | <var>
                | <const>

        <op> ::= + | - | * | /

        <func> ::= sin | cos | exp | log | sqrt

        <var> ::= x0

        <const> ::= 0.1 | 0.5 | 1..5 | 10
        """

    def grammar(self) -> Grammar:
        """
        Return the Grammar object.
        """
        return Grammar(grammar_str=self.grammar_str())

    def _generate_points(
        self,
        n_samples: int,
        sample_type: Literal["uniform", "grid", "random"],
    ) -> NDArray[np.float64]:
        low, high = self.x_range

        if sample_type in ("uniform", "random"):
            return self.np_rng.uniform(low, high, (n_samples, 1))
        if sample_type == "grid":
            return np.linspace(low, high, n_samples, dtype=np.float64).reshape(-1, 1)

        raise ValueError(f"Unknown sample type: {sample_type}")

    def _quartic_function(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray((X**4 + X**3 + X**2 + X).reshape(-1), dtype=np.float64)

    def _generate_data(
        self,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        X_train = self._generate_points(self.train_samples, self.train_type)
        y_train = self._quartic_function(X_train)
        X_test = self._generate_points(self.test_samples, self.test_type)
        y_test = self._quartic_function(X_test)
        return X_train, y_train, X_test, y_test

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.metadata.name,
            "function": "x^4 + x^3 + x^2 + x",
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "x_range": self.x_range,
            "train_type": self.train_type,
            "test_type": self.test_type,
            "random_state": self.random_state,
        }

    def __repr__(self) -> str:
        return f"KozaQuartic(range={self.x_range}, train={self.train_samples})"
