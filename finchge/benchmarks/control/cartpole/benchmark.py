import functools
import json
from pathlib import Path
from typing import Any, Optional

from finchge.benchmarks.base import Benchmark, BenchmarkMetadata
from finchge.grammar import Grammar

from .environment import CartPoleEnvironment
from .runner import CartPoleRunner


class CartPoleBenchmark(Benchmark):
    def __init__(
        self,
        random_state: Optional[Any] = None,
        max_steps: Optional[int] = None,
        n_episodes: int = 1,
    ):
        super().__init__(random_state=random_state)

        # Load defaults
        base_path = Path(__file__).parent
        with open(base_path / "specs.json") as f:
            self.specs = json.load(f)

        # Override if necessary
        self.max_steps = max_steps or self.specs["max_steps"]
        self.n_episodes = n_episodes
        self._metadata = BenchmarkMetadata(
            name=self.specs["name"],
            category="control",
            input_dim=0,
            output_dim=1,
            train_size=self.n_episodes,
            test_size=self.n_episodes,
        )

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata

    def grammar_str(self) -> str:
        return """
        <code> ::= <if> | <action>
        <if> ::= if ( <condition> ) ( <code> ) else ( <code> )
        <condition> ::= <state> <op> <number>
        <state> ::= cart_pos | cart_vel | pole_angle | pole_ang_vel
        <op> ::= < | > | <= | >=
        <number> ::= -1.0 | -0.5 | 0.0 | 0.5 | 1.0
        <action> ::= left | right
        """

    def grammar(self) -> Grammar:
        """
        Return the Grammar object.
        """
        return Grammar(grammar_str=self.grammar_str())

    def create_runner(self, data_type: str = "train"):
        # Inject the specs into the factory
        # We make a copy of specs so we don't modify the original dict
        env_params = self.specs.copy()
        env_params.pop("name", None)
        env_params["max_steps"] = self.max_steps

        env_factory = functools.partial(CartPoleEnvironment, **env_params)
        return CartPoleRunner(
            env_factory, n_episodes=self.n_episodes, random_state=self.random_state
        )
