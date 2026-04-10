import functools
import json
from pathlib import Path
from typing import Optional

import numpy as np

from finchge.benchmarks.base import Benchmark, BenchmarkMetadata
from finchge.grammar import Grammar

from .environment import MazeEnvironment
from .runner import MazeRunner


class MazeBenchmark(Benchmark):
    def __init__(self, version: str = "medium", max_steps: Optional[int] = None):
        super().__init__()

        with open(Path(__file__).parent / "specs.json") as f:
            specs = json.load(f)

        if version not in specs:
            raise ValueError(f"Unknown maze version: {version}")

        spec = specs[version]
        self.version = version
        self.grid = np.array(spec["grid"], dtype=np.int8)
        self.max_steps = max_steps or spec["max_steps"]

        self._metadata = BenchmarkMetadata(
            name=spec["name"],
            category="control",
            input_dim=0,
            output_dim=1,
            train_size=1,
            test_size=1,
        )

    def grammar_str(self) -> str:
        return """
        <code> ::= <line> | <code> <line>
        <line> ::= <condition> | <op>
        <condition> ::= if-wall-ahead ( <line> ) ( <line> )
                      | if-wall-left ( <line> ) ( <line> )
                      | if-wall-right ( <line> ) ( <line> )
        <op> ::= up | down | left | right
        """

    def grammar(self) -> Grammar:
        """
        Return the Grammar object.
        """
        return Grammar(grammar_str=self.grammar_str())

    def create_runner(self, data_type: str = "train") -> MazeRunner:
        env_factory = functools.partial(
            MazeEnvironment, grid=self.grid, max_steps=self.max_steps
        )
        return MazeRunner(env_factory)

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata
