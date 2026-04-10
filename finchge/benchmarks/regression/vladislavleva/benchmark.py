import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np

from finchge.grammar import Grammar
from finchge.symbolic import SymbolicExpression

from ... import Benchmark, BenchmarkMetadata
from ...typing import FloatArray
from .data import generate_vlad_points
from .grammar import get_vlad_grammar


class VladislavlevaBenchmark(Benchmark):
    def __init__(
        self,
        version: int,
        random_state: Optional[int] = None,
        train_samples: int = 1024,
        test_samples: int = 5000,
        x_range: Optional[Any] = None,
        noise_std: Optional[float] = None,
    ):
        super().__init__(random_state=random_state)

        # Load Spec
        with open(Path(__file__).parent / "functions.json") as f:
            spec = json.load(f)[str(version)]

        self.version = version
        self.dim = spec["dim"]
        self.expression_str = spec["expr"]
        self.noise_std = noise_std
        self.train_samples = train_samples
        self.test_samples = test_samples

        # Range Formatting: ensures we have a (low, high) for every dimension
        raw_range = x_range if x_range is not None else spec["range"]
        self.x_range = self._verify_range(raw_range)

        self._metadata = BenchmarkMetadata(
            name=f"Vladislavleva-{version}",
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
        return Grammar(get_vlad_grammar(self.dim))

    def grammar_str(self) -> str:
        return get_vlad_grammar(self.dim)

    def _verify_range(self, r: Any) -> Tuple[Tuple[float, float], ...]:
        """Convert [low, high] into ((low, high), (low, high)...) based on dim."""
        if isinstance(r[0], (float, int)):
            return tuple((float(r[0]), float(r[1])) for _ in range(self.dim))
        # Otherwise assume it's already a list of lists
        return tuple((float(sub[0]), float(sub[1])) for sub in r)

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train = generate_vlad_points(
            self.dim, self.train_samples, self.x_range, self.np_rng
        )
        X_test = generate_vlad_points(
            self.dim, self.test_samples, self.x_range, self.np_rng
        )

        y_train = self.func(X_train).flatten()
        y_test = self.func(X_test).flatten()

        if self.noise_std:
            y_train += self.np_rng.normal(0, self.noise_std, y_train.shape)

        return X_train, y_train, X_test, y_test

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata
