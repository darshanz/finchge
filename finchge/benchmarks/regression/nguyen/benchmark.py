import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

from finchge.benchmarks import Benchmark, BenchmarkMetadata
from finchge.benchmarks.regression.nguyen.data import generate_nguyen_points
from finchge.benchmarks.regression.nguyen.grammar import get_nguyen_grammar
from finchge.benchmarks.typing import FloatArray, Range
from finchge.grammar import Grammar
from finchge.symbolic import SymbolicExpression


class NguyenBenchmark(Benchmark):
    def __init__(
        self,
        version: int,
        random_state: Optional[int] = None,
        train_samples: Optional[int] = None,  # Optional default taken from JSON file
        test_samples: Optional[int] = None,  # same
        x_range: Optional[
            Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]
        ] = None,
        noise_std: Optional[float] = None,
        train_type: str = "grid",
        test_type: str = "uniform",
    ):
        super().__init__(random_state=random_state)

        # load our function definitions
        json_path = Path(__file__).parent / "functions.json"
        with open(json_path) as f:
            specs = json.load(f)

        if str(version) not in specs:
            raise ValueError(f"Nguyen version {version} not found in functions.json")

        spec = specs[str(version)]

        self.version = version
        self.dim = spec["dim"]

        self.train_samples = (
            train_samples if train_samples is not None else spec["train_samples"]
        )
        self.test_samples = test_samples if test_samples is not None else 1000
        # validation
        if self.train_samples <= 0:
            raise ValueError(
                f"train_samples must be positive, got {self.train_samples}"
            )
        if self.test_samples <= 0:
            raise ValueError(f"test_samples must be positive, got {self.test_samples}")

        self.train_type = train_type
        self.test_type = test_type

        # validation
        valid_types = ["uniform", "grid", "random"]
        if train_type not in valid_types:
            raise ValueError(
                f"train_type must be one of {valid_types}, got {train_type}"
            )
        if test_type not in valid_types:
            raise ValueError(f"test_type must be one of {valid_types}, got {test_type}")

        self.noise_std = noise_std

        self.name = spec["name"]
        self.noise_std = noise_std
        self.x_range: Range
        raw_range = x_range if x_range is not None else spec["range"]
        self.x_range = self._verify_range(raw_range)
        self.expression_str = spec["expr"]

        print(self.dim)

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

    def _verify_range(self, r: Range) -> Any:
        # if we only got one range, duplicate it
        if self.dim == 2 and not isinstance(r[0], (list, tuple)):
            r = (r, r)

        # quick sanity check for reversed ranges
        to_check = r if self.dim == 1 else r[0]
        if to_check[0] >= to_check[1]:  # type: ignore
            raise ValueError(f"Invalid range: {r}. Low must be less than high.")
        return r

    def grammar(self) -> Grammar:
        return Grammar(get_nguyen_grammar(self.version, self.dim))

    def grammar_str(self) -> str:
        return get_nguyen_grammar(self.version, self.dim)

    def _generate_data(self) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        X_train = generate_nguyen_points(
            self.dim, self.train_samples, self.x_range, self.train_type, self.np_rng
        )
        X_test = generate_nguyen_points(
            self.dim, self.test_samples, self.x_range, self.test_type, self.np_rng
        )

        # evaluate target values
        y_train = self.func(X_train)
        y_test = self.func(X_test)

        if self.noise_std:
            y_train += self.np_rng.normal(0, self.noise_std, y_train.shape)

        return X_train, y_train.flatten(), X_test, y_test.flatten()

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata
