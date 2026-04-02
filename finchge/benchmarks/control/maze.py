from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from finchge.benchmarks.base import Benchmark, BenchmarkMetadata
from finchge.benchmarks.control.base import ControlEnvironment
from finchge.benchmarks.registry import register
from finchge.grammar import Grammar
from finchge.runners.control import ControlRunner

# Easy 5x5
MAZE_SIMPLE: NDArray[np.int8] = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 2, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 3, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ],
    dtype=np.int8,
)  # 2=start, 3=goal

# Medium 8x8 (Koza)
MAZE_MEDIUM: NDArray[np.int8] = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 3, 1],
        [1, 0, 1, 0, 1, 0, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.int8,
)

# Hard 11x11
MAZE_HARD: NDArray[np.int8] = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 3, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.int8,
)

MAZES: Dict[str, NDArray[np.int8]] = {
    "simple": MAZE_SIMPLE,
    "medium": MAZE_MEDIUM,
    "hard": MAZE_HARD,
}


# environment
@dataclass
class MazeState:
    position: Tuple[int, int]
    steps_taken: int
    reached_goal: bool
    path: List[Tuple[int, int]] = field(default_factory=list)


class MazeEnvironment(ControlEnvironment):
    def __init__(
        self,
        maze: NDArray[np.int8],
        max_steps: int = 200,
        step_penalty: float = -1.0,
        goal_reward: float = 100.0,
    ) -> None:
        super().__init__("Maze", max_steps)

        self.original_maze: NDArray[np.int8] = maze.copy()
        self.step_penalty: float = step_penalty
        self.goal_reward: float = goal_reward
        self.maze: NDArray[np.int8]
        self.pos: Tuple[int, int]
        self.steps_taken: int
        self.episode_reward: float
        self.reached_goal: bool
        self.done: bool
        self.path: List[Tuple[int, int]]

        # start and goal positions
        self.start_pos: Optional[Tuple[int, int]] = None
        self.goal_pos: Optional[Tuple[int, int]] = None

        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if maze[i, j] == 2:
                    self.start_pos = (i, j)
                elif maze[i, j] == 3:
                    self.goal_pos = (i, j)

        if self.start_pos is None:
            raise ValueError("Maze must contain a start position (value 2)")
        if self.goal_pos is None:
            raise ValueError("Maze must contain a goal position (value 3)")

        # direction vectors and names
        self.directions: List[Tuple[str, Tuple[int, int]]] = [
            ("up", (-1, 0)),
            ("down", (1, 0)),
            ("left", (0, -1)),
            ("right", (0, 1)),
        ]

        self.reset()

    def reset(self) -> Dict[str, bool]:
        self.maze = self.original_maze.copy()
        self.pos = cast(Tuple[int, int], self.start_pos)
        self.steps_taken = 0
        self.episode_reward = 0.0
        self.reached_goal = False
        self.done = False
        self.path = [self.pos]

        return self.get_observation()

    def step(self, action: str) -> Tuple[Dict[str, bool], float, bool, Dict[str, Any]]:
        if self.done:
            return self.get_observation(), 0.0, True, {"error": "Episode already done"}

        reward = self.step_penalty
        info: Dict[str, Any] = {}

        # direction vector for action
        dir_vector: Optional[Tuple[int, int]] = None
        for name, vec in self.directions:
            if name == action:
                dir_vector = vec
                break

        if dir_vector is None:
            raise ValueError(f"Invalid action: {action}")

        # Calculate new position
        dr, dc = dir_vector
        new_row = self.pos[0] + dr
        new_col = self.pos[1] + dc

        # check if move is valid, should be within bounds... not a wall
        if (
            0 <= new_row < self.maze.shape[0]
            and 0 <= new_col < self.maze.shape[1]
            and self.maze[new_row, new_col] != 1
        ):
            self.pos = (new_row, new_col)
            self.path.append(self.pos)

            # if reached goal
            if self.goal_pos is not None and self.pos == self.goal_pos:
                reward = self.goal_reward
                self.reached_goal = True
                self.done = True
                info["termination"] = "goal_reached"

        self.steps_taken += 1
        self.episode_reward += reward

        # Check max steps
        if self.steps_taken >= self.max_steps:
            self.done = True
            info["termination"] = "max_steps"

        info["position"] = self.pos
        info["steps"] = self.steps_taken
        info["reached_goal"] = self.reached_goal
        info["path_length"] = len(self.path)

        return self.get_observation(), reward, self.done, info

    def get_observation(self) -> Dict[str, bool]:
        observation: Dict[str, bool] = {}
        for direction, (dr, dc) in self.directions:
            new_row = self.pos[0] + dr
            new_col = self.pos[1] + dc

            if 0 <= new_row < self.maze.shape[0] and 0 <= new_col < self.maze.shape[1]:
                # Check if wall
                observation[direction] = self.maze[new_row, new_col] == 1
            else:
                # Out of bounds treated as wall
                observation[direction] = True

        return observation

    def get_available_actions(self) -> List[str]:
        actions: List[str] = []

        for direction, (dr, dc) in self.directions:
            new_row = self.pos[0] + dr
            new_col = self.pos[1] + dc

            # Check if move is valid
            if (
                0 <= new_row < self.maze.shape[0]
                and 0 <= new_col < self.maze.shape[1]
                and self.maze[new_row, new_col] != 1
            ):
                actions.append(direction)

        return actions

    def get_action_space(self) -> List[str]:
        return ["up", "down", "left", "right"]

    def render(self) -> Optional[NDArray[np.float64]]:
        """Return visual representation of current state."""
        viz = self.maze.astype(np.float64)
        viz[self.pos] = 4.0  # Mark agent position
        return viz


# Maze Interpreter
class MazeInterpreter:
    """
    Interpreter for maze navigation programs.

    Grammar:
        <code> ::= <line> | <code> <line>
        <line> ::= <condition> | <op>
        <condition> ::= if-wall-ahead ( <line> ) ( <line> )
                      | if-wall-left ( <line> ) ( <line> )
                      | if-wall-right ( <line> ) ( <line> )
        <op> ::= up | down | left | right
    """

    def __init__(self) -> None:
        self.actions: List[str] = ["up", "down", "left", "right"]
        self.conditions: List[str] = ["if-wall-ahead", "if-wall-left", "if-wall-right"]
        self.keywords: List[str] = self.actions + self.conditions

    def tokenize(self, program: str) -> List[str]:
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

            # Check for conditions (longest first)
            matched = False
            for cond in sorted(self.conditions, key=len, reverse=True):
                if program.startswith(cond, i):
                    tokens.append(cond)
                    i += len(cond)
                    matched = True
                    break

            if matched:
                continue

            # Check for actions
            for action in self.actions:
                if program.startswith(action, i):
                    tokens.append(action)
                    i += len(action)
                    matched = True
                    break

            if not matched:
                # Unknown token, skip one character
                i += 1

        return tokens

    def evaluate(
        self, tokens: List[str], observation: Dict[str, bool], pc: int = 0
    ) -> Tuple[Optional[str], int]:
        if pc >= len(tokens):
            return None, pc

        token = tokens[pc]

        # Actions
        if token in self.actions:
            return token, pc + 1

        # Wall conditionals
        if token in self.conditions:
            # which direction to check
            wall_present: bool = False
            if token == "if-wall-ahead":
                wall_present = observation.get("up", False)
            elif token == "if-wall-left":
                wall_present = observation.get("left", False)
            elif token == "if-wall-right":
                wall_present = observation.get("right", False)
            if pc + 1 >= len(tokens) or tokens[pc + 1] != "(":
                return None, pc + 2

            true_action, true_pc = self.evaluate(tokens, observation, pc + 2)
            if true_pc >= len(tokens) or tokens[true_pc] != ")":
                return None, true_pc
            if true_pc + 1 >= len(tokens) or tokens[true_pc + 1] != "(":
                return None, true_pc + 2
            false_action, false_pc = self.evaluate(tokens, observation, true_pc + 2)
            if false_pc >= len(tokens) or tokens[false_pc] != ")":
                return None, false_pc
            if wall_present:
                return true_action, false_pc + 1
            else:
                return false_action, false_pc + 1
        return self.evaluate(tokens, observation, pc + 1)


# Maze Executor
class MazeRunner(ControlRunner[MazeEnvironment]):
    def __init__(
        self,
        env_factory: Callable[[], MazeEnvironment],
        n_episodes: int = 1,
        interpreter: Optional[MazeInterpreter] = None,
        optimal_fitness: Optional[float] = None,
        random_state: Optional[Any] = None,
    ) -> None:
        super().__init__(
            random_state=random_state,
            env_factory=env_factory,
            n_episodes=n_episodes,
            optimal_fitness=optimal_fitness,
        )
        self.interpreter = interpreter or MazeInterpreter()
        self._last_rewards: List[float] = []

    def _run_episode(self, phenotype: str, env: MazeEnvironment) -> float:
        tokens = self.interpreter.tokenize(phenotype)
        env.reset()
        program_counter = 0
        total_reward = 0.0

        while not env.is_done():
            observation = env.get_observation()
            action, program_counter = self.interpreter.evaluate(
                tokens, observation, program_counter
            )

            if action is None:
                actions = env.get_available_actions()
                action = self.rng.choice(actions) if actions else None
                program_counter = 0

            if action is not None:
                _, reward, _, _ = env.step(action)
                total_reward += reward
            if program_counter >= len(tokens):
                program_counter = 0

        return total_reward

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        if state is not None:
            state.update(
                {
                    "interpreter": self.interpreter,
                    "_last_rewards": self._last_rewards,
                }
            )
        return state or {}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


# Maze Benchmark
def create_maze_environment(
    maze_name: str = "medium", max_steps: int = 200
) -> MazeEnvironment:
    if maze_name not in MAZES:
        raise ValueError(f"Unknown maze: {maze_name}. Choose from {list(MAZES.keys())}")

    return MazeEnvironment(maze=MAZES[maze_name], max_steps=max_steps)


@register("maze_simple", "control")
class MazeSimpleBenchmark(Benchmark):
    def __init__(
        self,
        random_state: Optional[int] = None,
        max_steps: int = 100,
        n_episodes: int = 1,
    ) -> None:
        super().__init__(random_state=random_state)
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.name = "Maze (Simple)"
        self.total_food: Optional[int] = None
        self.random_state = random_state

        self._metadata = BenchmarkMetadata(
            name="Maze (Simple)",
            category="control",
            description="Simple 5x5 maze navigation task",
            reference="Koza, J.R. (1992). Genetic Programming",
            input_dim=0,  # Control problems do not have fixed input dim
            output_dim=1,  # Single fitness value
            train_size=n_episodes,
            test_size=n_episodes,
        )

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata

    def _generate_data(
        self,
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        raise NotImplementedError(
            "Control Problems do not implement _generate_data function."
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
        def env_factory() -> MazeEnvironment:
            return create_maze_environment("simple", self.max_steps)

        return MazeRunner(
            env_factory, n_episodes=self.n_episodes, random_state=self.random_state
        )


@register("maze_medium", "control")
class MazeMediumBenchmark(Benchmark):
    def __init__(
        self,
        random_state: Optional[int] = None,
        max_steps: int = 200,
        n_episodes: int = 1,
    ) -> None:
        super().__init__(random_state=random_state)
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.name = "Maze (Medium)"
        self.total_food: Optional[int] = None
        self.random_state = random_state

        self._metadata = BenchmarkMetadata(
            name="Maze (Medium)",
            category="control",
            description="Medium 8x8 maze navigation task from Koza",
            reference="Koza, J.R. (1992). Genetic Programming",
            input_dim=0,
            output_dim=1,
            train_size=n_episodes,
            test_size=n_episodes,
        )

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata

    def _generate_data(
        self,
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        raise NotImplementedError(
            "Control Problems do not implement _generate_data function."
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
        def env_factory() -> MazeEnvironment:
            return create_maze_environment("medium", self.max_steps)

        return MazeRunner(
            env_factory, n_episodes=self.n_episodes, random_state=self.random_state
        )


@register("maze_hard", "control")
class MazeHardBenchmark(Benchmark):
    def __init__(
        self,
        random_state: Optional[int] = None,
        max_steps: int = 500,
        n_episodes: int = 1,
    ) -> None:
        super().__init__(random_state=random_state)
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.name = "Maze (Hard)"
        self.total_food: Optional[int] = None
        self.random_state = random_state

        self._metadata = BenchmarkMetadata(
            name="Maze (Hard)",
            category="control",
            description="Hard 11x11 maze navigation task",
            reference="Koza, J.R. (1992). Genetic Programming",
            input_dim=0,
            output_dim=1,
            train_size=n_episodes,
            test_size=n_episodes,
        )

    @property
    def metadata(self) -> BenchmarkMetadata:
        return self._metadata

    def _generate_data(
        self,
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        raise NotImplementedError(
            "Control Problems do not implement _generate_data function."
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
        def env_factory() -> MazeEnvironment:
            return create_maze_environment("hard", self.max_steps)

        return MazeRunner(
            env_factory, n_episodes=self.n_episodes, random_state=self.random_state
        )
