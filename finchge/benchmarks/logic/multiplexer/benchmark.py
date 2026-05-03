import json
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from finchge.benchmarks import Benchmark, BenchmarkMetadata
from finchge.benchmarks.logic.multiplexer.data import generate_multiplexer_table
from finchge.benchmarks.logic.multiplexer.grammar import get_logic_grammar
from finchge.grammar import Grammar


class MultiplexerBenchmark(Benchmark):
    def __init__(self, version: int, random_state: Optional[int] = None):
        super().__init__(random_state=random_state)

        with open(Path(__file__).parent / "specs.json") as f:
            spec = json.load(f)[str(version)]

        self.n_bits = spec["bits"]
        self.n_address = spec["address_bits"]

        self._metadata = BenchmarkMetadata(
            name=spec["name"],
            category="logic",
            input_dim=self.n_bits,
            output_dim=1,
            train_size=2**self.n_bits,
            test_size=2**self.n_bits,
        )

    def grammar(self) -> Grammar:
        return Grammar(get_logic_grammar(self.n_bits))

    def grammar_str(self) -> str:
        return get_logic_grammar(self.n_bits)

    def _generate_data(
        self,
    ) -> Tuple[NDArray[np.int8], NDArray[np.int8], NDArray[np.int8], NDArray[np.int8]]:
        X, y = generate_multiplexer_table(self.n_bits, self.n_address)
        return X, y, X.copy(), y.copy()

    def create_runner(self, data_type: str = "train") -> Any:
        # Avoid circular imports
        from finchge.runners.logic import LogicRunner

        X, y, _, _ = self._generate_data()
        return LogicRunner(X, y)

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata
