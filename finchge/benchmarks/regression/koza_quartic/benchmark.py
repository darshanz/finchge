import json
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

from finchge.benchmarks import Benchmark, BenchmarkMetadata
from finchge.benchmarks.regression.koza_quartic.data import generate_koza_points
from finchge.benchmarks.regression.koza_quartic.grammar import get_koza_grammar
from finchge.benchmarks.typing import FloatArray, Range
from finchge.grammar import Grammar
from finchge.symbolic import SymbolicExpression


class KozaQuarticBenchmark(Benchmark):
    def __init__(
        self,
        random_state: Optional[int] = None,
        train_samples: Optional[int] = None,
        test_samples: Optional[int] = None,
        x_range: Optional[
            Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]
        ] = None,
        train_type: Literal["uniform", "grid", "random"] = "uniform",
        test_type: Literal["uniform", "grid", "random"] = "grid",
    ):
        super().__init__(random_state=random_state)

        # Load internal spec
        with open(Path(__file__).parent / "functions.json") as f:
            spec = json.load(f)["quartic"]

        self.expression_str = spec["expr"]
        self.dim = spec["dim"]

        self.train_samples = (
            train_samples if train_samples is not None else spec["train_samples"]
        )
        self.test_samples = (
            test_samples if test_samples is not None else spec["test_samples"]
        )
        self.train_type = train_type
        self.test_type = test_type

        self.x_range: Range = x_range if x_range is not None else spec["range"]
        if self.train_samples <= 0:
            raise ValueError(
                f"train_samples must be positive, got {self.train_samples}"
            )
        if self.test_samples <= 0:
            raise ValueError(f"test_samples must be positive, got {self.test_samples}")
        if self.x_range[0] >= self.x_range[1]:  # type: ignore
            raise ValueError(
                f"Invalid range: low={self.x_range[0]} >= high={self.x_range[1]}"
            )

        valid_types = {"uniform", "grid", "random"}
        if self.train_type not in valid_types:
            raise ValueError(
                f"train_type must be one of {valid_types}, got {self.train_type}"
            )
        if self.test_type not in valid_types:
            raise ValueError(
                f"test_type must be one of {valid_types}, got {self.test_type}"
            )

        self._metadata = BenchmarkMetadata(
            name=spec["name"],
            category="regression",
            input_dim=self.dim,
            output_dim=1,
            train_size=self.train_samples,
            test_size=self.test_samples,
        )

    @property
    def func(self) -> Callable[[FloatArray], FloatArray]:
        return SymbolicExpression(self.expression_str).eval

    def grammar(self) -> Grammar:
        return Grammar(get_koza_grammar())

    def grammar_str(self) -> str:
        return get_koza_grammar()

    def _generate_data(self) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        if not isinstance(self.x_range[0], (float, int)):
            raise TypeError(
                f"Koza expects a 1D range (tuple[float, float]), got {type(self.x_range[0])}"
            )

        X_train = generate_koza_points(
            self.train_samples, self.x_range, self.train_type, self.np_rng
        )
        X_test = generate_koza_points(
            self.test_samples, self.x_range, self.test_type, self.np_rng
        )

        y_train = self.func(X_train).flatten()
        y_test = self.func(X_test).flatten()

        return X_train, y_train, X_test, y_test

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata
