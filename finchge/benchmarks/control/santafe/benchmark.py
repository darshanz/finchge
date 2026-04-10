import functools
import json
from pathlib import Path
from typing import Optional

import numpy as np

from finchge.benchmarks.base import Benchmark, BenchmarkMetadata
from finchge.grammar import Grammar

from .environment import create_santafe_env
from .grammar import get_santafe_grammar
from .runner import SantaFeRunner


class SantaFeTrailBenchmark(Benchmark):
    """
    Santa Fe Trail (Artificial Ant) benchmark.

    The ant must navigate a 32x32 grid with 89 food pellets along a winding trail.
    Fitness is the number of food pellets eaten within the time limit.
    """

    def __init__(
        self, random_state: Optional[int] = None, max_steps: Optional[int] = None
    ):
        super().__init__(random_state=random_state)

        base_path = Path(__file__).parent
        with open(base_path / "specs.json") as f:
            specs = json.load(f)

        with open(base_path / "trail.txt") as f:
            grid_data = [[int(c) for c in line.strip()] for line in f if line.strip()]
            self.grid = np.array(grid_data, dtype=np.int8)

        self.max_steps = max_steps if max_steps is not None else specs["max_steps"]
        self.total_food = specs["total_food"]
        self.start_pos = tuple(specs["start_pos"])
        self.start_dir = specs["start_dir"]

        self._metadata = BenchmarkMetadata(
            name=specs["name"],
            category="control",
            input_dim=0,
            output_dim=1,
            train_size=1,
            test_size=1,
        )

    def grammar(self) -> Grammar:
        return Grammar(get_santafe_grammar())

    def grammar_str(self) -> str:
        """
        Return grammar for Santa Fe Trail problem.
        """
        return get_santafe_grammar()

    def create_runner(self, data_type: str = "train") -> SantaFeRunner:
        env_factory = functools.partial(
            create_santafe_env,
            grid=self.grid,
            max_steps=self.max_steps,
            total_food=self.total_food,
            start_pos=self.start_pos,
            start_dir=self.start_dir,
        )
        return SantaFeRunner(env_factory, random_state=self.random_state)

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata
