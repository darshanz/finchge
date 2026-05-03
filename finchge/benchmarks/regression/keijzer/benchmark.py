import json
from pathlib import Path
from typing import Callable, Optional, Tuple

from finchge.benchmarks import Benchmark, BenchmarkMetadata
from finchge.benchmarks.regression.keijzer.data import (
    calc_keijzer_size,
    generate_keijzer_points,
)
from finchge.benchmarks.regression.keijzer.grammar import get_keijzer_grammar
from finchge.benchmarks.typing import FloatArray
from finchge.grammar import Grammar
from finchge.symbolic import SymbolicExpression


class KeijzerBenchmark(Benchmark):
    def __init__(
        self,
        version: int,
        random_state: Optional[int] = None,
        train_samples: Optional[int] = None,
        test_samples: Optional[int] = None,
    ):
        super().__init__(random_state=random_state)

        # Load the spec
        json_path = Path(__file__).parent / "functions.json"
        with open(json_path) as f:
            specs = json.load(f)
        spec = specs[str(version)]

        self.version = version
        self.dim = spec["dim"]
        self.expression_str = spec["expr"]
        self.train_range = spec["train_range"]
        self.test_range = spec["test_range"]
        self.train_step = spec["train_step"]
        self.test_step = spec["test_step"]

        # Calculate sizes: User Override > Step Calculation > Defaults
        self.train_size = train_samples or calc_keijzer_size(
            self.train_range, self.train_step, 100
        )
        self.test_size = test_samples or calc_keijzer_size(
            self.test_range, self.test_step, 1000
        )

        self._metadata = BenchmarkMetadata(
            name=spec["name"],
            category="regression",
            input_dim=self.dim,
            output_dim=1,
            train_size=self.train_size,
            test_size=self.test_size,
        )

    @property
    def func(self) -> Callable[[FloatArray], FloatArray]:
        return SymbolicExpression(self.expression_str).eval

    def grammar(self) -> Grammar:
        return Grammar(get_keijzer_grammar(self.dim))

    def grammar_str(self) -> str:
        return get_keijzer_grammar(self.dim)

    def _generate_data(self) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        # X generation
        X_train = generate_keijzer_points(
            self.dim,
            self.train_size,
            self.train_range,
            self.train_step,
            False,
            self.np_rng,
        )
        X_test = generate_keijzer_points(
            self.dim, self.test_size, self.test_range, self.test_step, True, self.np_rng
        )

        # Target evaluation
        y_train = self.func(X_train).flatten()
        y_test = self.func(X_test).flatten()

        return X_train, y_train, X_test, y_test

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata
