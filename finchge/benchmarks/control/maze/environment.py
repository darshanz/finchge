import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from finchge.benchmarks.control.base import ControlEnvironment


class MazeEnvironment(ControlEnvironment):
    def __init__(
        self,
        grid: np.ndarray,
        max_steps: int = 200,
        step_penalty: float = -1.0,
        goal_reward: float = 100.0,
    ) -> None:
        super().__init__("Maze", max_steps)

        self.original_maze = grid.copy()
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.n_rows, self.n_cols = self.original_maze.shape

        start_indices = np.argwhere(self.original_maze == 2)
        goal_indices = np.argwhere(self.original_maze == 3)

        if len(start_indices) == 0 or len(goal_indices) == 0:
            raise ValueError("Maze must contain a start (2) and goal (3)")

        self.start_pos: Tuple[int, int] = tuple(start_indices[0])  # type: ignore
        self.goal_pos: Tuple[int, int] = tuple(goal_indices[0])  # type: ignore

        self.directions = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        self.maze: np.ndarray = None  # type: ignore
        self.pos: Tuple[int, int] = (0, 0)
        self.steps_taken: int = 0
        self.episode_reward: float = 0.0
        self.reached_goal: bool = False
        self.done: bool = False
        self.path: List[Tuple[int, int]] = []

        self.reset()

    def reset(self) -> Dict[str, bool]:
        self.maze = self.original_maze.copy()
        self.pos = self.start_pos
        self.steps_taken = 0
        self.episode_reward = 0.0
        self.reached_goal = False
        self.done = False
        self.path = [self.pos]

        return self.get_observation()

    def step(self, action: str) -> Tuple[Dict[str, bool], float, bool, Dict[str, Any]]:
        if self.done:
            return self.get_observation(), 0.0, True, {"info": "already done"}

        dr, dc = self.directions[action]
        new_row, new_col = self.pos[0] + dr, self.pos[1] + dc

        # bounds and walls
        if (
            0 <= new_row < self.n_rows
            and 0 <= new_col < self.n_cols
            and self.maze[new_row, new_col] != 1
        ):
            self.pos = (new_row, new_col)
            self.path.append(self.pos)

        # Calculate rewards
        if self.pos == self.goal_pos:
            reward = self.goal_reward
            self.reached_goal = True
            self.done = True
        else:
            reward = self.step_penalty

        self.steps_taken += 1
        self.episode_reward += reward

        if self.steps_taken >= self.max_steps:
            self.done = True

        return (
            self.get_observation(),
            reward,
            self.done,
            {
                "position": self.pos,
                "steps": self.steps_taken,
                "path_length": len(self.path),
            },
        )

    def get_observation(self) -> Dict[str, bool]:
        obs = {}
        for name, (dr, dc) in self.directions.items():
            r, c = self.pos[0] + dr, self.pos[1] + dc
            in_bounds = (
                0 <= r < self.original_maze.shape[0]
                and 0 <= c < self.original_maze.shape[1]
            )
            obs[name] = not in_bounds or self.original_maze[r, c] == 1
        return obs

    def get_available_actions(self) -> List[str]:
        """Returns only the actions that won't result in hitting a wall or going OOB."""
        actions = []
        for name, (dr, dc) in self.directions.items():
            r, c = self.pos[0] + dr, self.pos[1] + dc
            # Check if inside grid and not a wall (1)
            if (
                0 <= r < self.original_maze.shape[0]
                and 0 <= c < self.original_maze.shape[1]
                and self.original_maze[r, c] != 1
            ):
                actions.append(name)
        return actions

    def get_action_space(self) -> List[str]:
        return list(self.directions.keys())

    @classmethod
    def from_version(cls, version: str = "medium") -> "MazeEnvironment":
        """
        Creating env for given version
        """
        base_path = Path(__file__).parent
        with open(base_path / "specs.json") as f:
            specs = json.load(f)

        if version not in specs:
            raise ValueError(
                f"Unknown maze version: {version}. Available: {list(specs.keys())}"
            )

        spec = specs[version]
        grid = np.array(spec["grid"], dtype=np.int8)

        return cls(grid=grid, max_steps=spec["max_steps"])
