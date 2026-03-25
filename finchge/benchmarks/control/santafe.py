import functools
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from finchge.benchmarks.base import Benchmark, BenchmarkMetadata
from finchge.benchmarks.control.base import ControlEnvironment
from finchge.benchmarks.registry import register
from finchge.grammar import Grammar
from finchge.runners.control import ControlRunner

SFE_TRAIL_STRING = """
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
00000000000000000000000000000000
11111111111111111111111111111110
00000000000000000000000000000010
00000000000000000000000000000010
00000000000000000000000000000010
00000000000000000000000000000010
00000000000000000000000000000010
00000000000000000000000000000010
00000000000000000000000000000010
00000000000000000000000000000010
00000000000000000000000000000010
00000000000000000000000000000010
00000000000000000000000000000011
"""


def parse_santafe_trail(string_repr: str) -> NDArray[np.int8]:
    """Convert string representation to numpy array."""
    lines = string_repr.strip().split("\n")
    # Filter out empty lines
    lines = [line for line in lines if line.strip()]
    return np.array([[int(c) for c in line] for line in lines], dtype=np.int8)


SANTA_FE_TRAIL: NDArray[np.int8] = parse_santafe_trail(SFE_TRAIL_STRING)

START_POS: Tuple[int, int] = (20, 0)  # Row 20, Column 0
START_DIR: int = 1  # 0=up, 1=right, 2=down, 3=left
MAX_STEPS: int = 600
TOTAL_FOOD: int = 89


def create_santafe_env(max_steps: int = MAX_STEPS) -> "SantaFeEnvironment":
    """Factory function for creating Santa Fe environments."""
    return SantaFeEnvironment(max_steps=max_steps)


# Environment Class
class SantaFeEnvironment(ControlEnvironment):
    """
    Santa Fe Trail environment for the artificial ant problem.

    The ant navigates a toroidal 32x32 grid with 89 food pellets.
    Reward is +1 for each food pellet eaten.
    """

    def __init__(self, max_steps: int = MAX_STEPS) -> None:
        """
        Initialize Santa Fe environment.

        Args:
            max_steps: Maximum number of steps per episode
        """
        super().__init__("SantaFe", max_steps)
        self.original_grid: NDArray[np.int8] = SANTA_FE_TRAIL.copy()
        self.total_food: int = TOTAL_FOOD
        self.grid: Optional[NDArray[np.int8]] = None
        self.pos: Optional[Tuple[int, int]] = None
        self.dir: Optional[int] = None
        self.food_eaten: int = 0
        self.n_rows: int = 0
        self.n_cols: int = 0

        # Direction vectors: up, right, down, left
        self.dirs: List[Tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        self.reset()

    def reset(self) -> bool:
        """Reset environment to initial state."""
        self.grid = self.original_grid.copy()
        self.pos = START_POS
        self.dir = START_DIR
        self.steps_taken = 0
        self.episode_reward = 0.0
        self.food_eaten = 0
        self.done = False

        # Check if starting position has food
        if self.grid is not None and self.pos is not None:
            if self.grid[self.pos] == 1:
                self.food_eaten += 1
                self.episode_reward += 1
                self.grid[self.pos] = 0

        self.n_rows, self.n_cols = self.grid.shape if self.grid is not None else (0, 0)

        return self.get_observation()

    def step(self, action: str) -> Tuple[bool, float, bool, Dict[str, Any]]:
        """
        Execute action.

        Args:
            action: 'move', 'turn-left', or 'turn-right'

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            return self.get_observation(), 0.0, True, {"error": "Episode already done"}

        reward = 0.0
        info: Dict[str, Any] = {}

        # Ensure grid and pos are initialized
        assert self.grid is not None, "Grid not initialized"
        assert self.pos is not None, "Position not initialized"
        assert self.dir is not None, "Direction not initialized"

        # Update direction for turns
        if action == "turn-left":
            self.dir = (self.dir - 1) % 4
        elif action == "turn-right":
            self.dir = (self.dir + 1) % 4
        elif action == "move":
            # Move forward
            dr, dc = self.dirs[self.dir]
            new_row = (self.pos[0] + dr) % self.n_rows
            new_col = (self.pos[1] + dc) % self.n_cols
            new_pos = (int(new_row), int(new_col))

            # Check for food
            if self.grid[new_pos] == 1:
                reward = 1.0
                self.food_eaten += 1
                self.episode_reward += reward
                self.grid[new_pos] = 0

            self.pos = new_pos
        else:
            raise ValueError(f"Invalid action: {action}")

        self.steps_taken += 1

        # Check termination conditions
        if self.steps_taken >= self.max_steps:
            self.done = True
            info["termination"] = "max_steps"
        elif self.food_eaten >= self.total_food:
            self.done = True
            info["termination"] = "all_food_eaten"

        info["food_eaten"] = self.food_eaten
        info["steps"] = self.steps_taken
        info["position"] = self.pos
        info["direction"] = self.dir

        return self.get_observation(), reward, self.done, info

    def get_observation(self) -> bool:
        """
        Get current observation (food ahead sensor).

        Returns:
            True if food is in the cell directly ahead, False otherwise
        """
        if self.grid is None or self.pos is None or self.dir is None:
            return False

        dr, dc = self.dirs[self.dir]
        ahead_row = (self.pos[0] + dr) % 32
        ahead_col = (self.pos[1] + dc) % 32
        return bool(self.grid[ahead_row, ahead_col] == 1)

    def get_available_actions(self) -> List[str]:
        """Return available actions (all actions always available)."""
        return ["move", "turn-left", "turn-right"]

    def get_action_space(self) -> List[str]:
        """Return complete action space."""
        return ["move", "turn-left", "turn-right"]

    def render(self) -> Optional[NDArray[np.float64]]:
        """Simple text rendering."""
        if self.grid is None or self.pos is None:
            return None

        grid_viz = self.grid.astype(np.float64)
        grid_viz[self.pos] = 0.5  # Mark ant position
        return grid_viz


# Program Interpreter
class SantaFeInterpreter:
    """
    Interpreter for Santa Fe ant programs using cleaner grammar.

    Grammar:
        <code> ::= <line> | <code> <line>
        <line> ::= <condition> | <op>
        <condition> ::= ifelse food-ahead (<line>) (<line>)
        <op> ::= turn-left | turn-right | move
    """

    def __init__(self) -> None:
        self.actions: List[str] = ["move", "turn-left", "turn-right"]
        self.keywords: List[str] = self.actions + ["ifelse", "food-ahead"]

    def tokenize(self, program: str) -> List[str]:
        """Convert program string to list of tokens."""
        tokens: List[str] = []
        i = 0
        while i < len(program):
            # Skip whitespace
            if program[i].isspace():
                i += 1
                continue

            # Parentheses as separate tokens
            if program[i] == "(":
                tokens.append("(")
                i += 1
                continue
            if program[i] == ")":
                tokens.append(")")
                i += 1
                continue

            # Check for keywords
            matched = False
            for kw in self.keywords:
                if program.startswith(kw, i):
                    tokens.append(kw)
                    i += len(kw)
                    matched = True
                    break

            if not matched:
                # Skip unknown characters
                i += 1

        return tokens

    def evaluate(
        self, tokens: List[str], observation: bool, pc: int = 0
    ) -> Tuple[Optional[str], int]:
        """
        Evaluate tokenized program.

        Args:
            tokens: List of tokens from tokenize()
            observation: Current food-ahead sensor value
            pc: Program counter (index into tokens)

        Returns:
            Tuple of (action, new_pc)
        """
        if pc >= len(tokens):
            return None, pc

        token = tokens[pc]

        # Actions
        if token in self.actions:
            return token, pc + 1

        # ifelse statement
        if token == "ifelse":
            # Expect 'food-ahead'
            if pc + 1 >= len(tokens) or tokens[pc + 1] != "food-ahead":
                return None, pc + 2

            # Expect '('
            if pc + 2 >= len(tokens) or tokens[pc + 2] != "(":
                return None, pc + 3

            # Parse true branch
            true_action, true_pc = self.evaluate(tokens, observation, pc + 3)

            # Expect ')'
            if true_pc >= len(tokens) or tokens[true_pc] != ")":
                return None, true_pc

            # Expect '(' for false branch
            if true_pc + 1 >= len(tokens) or tokens[true_pc + 1] != "(":
                return None, true_pc + 2

            # Parse false branch
            false_action, false_pc = self.evaluate(tokens, observation, true_pc + 2)

            # Expect ')'
            if false_pc >= len(tokens) or tokens[false_pc] != ")":
                return None, false_pc

            # Return appropriate action based on observation
            if observation:
                return true_action, false_pc + 1
            else:
                return false_action, false_pc + 1

        return self.evaluate(tokens, observation, pc + 1)

    def program_to_string(self, tokens: List[str]) -> str:
        """
        Convert tokens back to readable program string.
        Tokens separated by space
        """
        return " ".join(tokens)


# Executor Class
class SantaFeRunner(ControlRunner[SantaFeEnvironment]):
    """
    Runner for Santa Fe Trail problem.
    """

    def __init__(
        self,
        env_factory: Callable[[], SantaFeEnvironment],
        n_episodes: int = 1,
        interpreter: Optional[SantaFeInterpreter] = None,
        random_state: Optional[Any] = None,
    ) -> None:
        """
        Initialize Santa Fe runner.

        Args:
            env_factory: Function that creates new SantaFeEnvironment instances
            n_episodes: Number of episodes to run per evaluation
            interpreter: Program interpreter (creates new one if None)
        """
        super().__init__(
            random_state=random_state,
            env_factory=env_factory,
            n_episodes=n_episodes,
            optimal_fitness=89.0,
        )
        self.interpreter = interpreter or SantaFeInterpreter()

    def _run_episode(self, phenotype: str, env: SantaFeEnvironment) -> float:
        """Run one episode in Santa Fe environment."""
        # Tokenize program
        tokens = self.interpreter.tokenize(phenotype)

        env.reset()
        program_counter = 0
        total_reward = 0.0

        while not env.is_done():
            food_ahead = env.get_observation()

            action, program_counter = self.interpreter.evaluate(
                tokens, food_ahead, program_counter
            )

            if action is None:
                # Program terminated, take random action
                actions = env.get_available_actions()
                action = self.rng.choice(actions) if actions else None
                program_counter = 0

            if action:
                _, reward, done, _ = env.step(action)
                total_reward += reward

            # Wrap program counter if at end
            if program_counter >= len(tokens):
                program_counter = 0

        return total_reward

    def __getstate__(self) -> Dict[str, Any]:
        """Called when pickling."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Called when unpickling."""
        self.__dict__.update(state)


@register("santafe_trail", "control")
class SantaFeTrailBenchmark(Benchmark):
    """
    Santa Fe Trail (Artificial Ant) benchmark.

    The ant must navigate a 32x32 grid with 89 food pellets along a winding trail.
    Fitness is the number of food pellets eaten within the time limit.
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        max_steps: int = MAX_STEPS,
        n_episodes: int = 1,
    ) -> None:
        """
        Initialize Santa Fe Trail benchmark.

        Args:
            random_state: Seed for reproducibility (environment is deterministic)
            max_steps: Maximum number of steps per episode
            n_episodes: Number of episodes per evaluation
        """
        super().__init__(random_state=random_state)

        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.total_food = TOTAL_FOOD
        self.random_state = random_state

        self._metadata = BenchmarkMetadata(
            name="Santa Fe Trail",
            category="control",
            description=f"Artificial ant on 32x32 grid with {TOTAL_FOOD} food pellets",
            reference="Koza, J.R. (1992). Genetic Programming",
            input_dim=0,  # Control problems do not have fixed input dim
            output_dim=1,  # Single fitness value
            train_size=n_episodes,
            test_size=n_episodes,
        )

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata

    def grammar_str(self) -> str:
        """
        Return grammar for Santa Fe Trail problem.
        """
        return """
        <code> ::= <line> | <code> <line>
        <line> ::= <condition> | <op>
        <condition> ::= ifelse food-ahead (<line>) (<line>)
        <op> ::= turn-left | turn-right | move
        """

    def grammar(self) -> Grammar:
        """
        Return the Grammar object.
        """
        return Grammar(grammar_str=self.grammar_str())

    def create_runner(self, data_type: str = "train") -> SantaFeRunner:
        """
        Create runner for this benchmark.

        Args:
            data_type: 'train' or 'test' (same environment for both)

        Returns:
            SantaFeExecutor instance
        """
        # Create a factory function with correct max_steps
        env_factory: Callable[[], SantaFeEnvironment] = functools.partial(
            create_santafe_env, max_steps=self.max_steps
        )
        return SantaFeRunner(
            env_factory, n_episodes=self.n_episodes, random_state=self.random_state
        )
