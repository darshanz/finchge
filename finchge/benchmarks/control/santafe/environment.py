import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from finchge.benchmarks.control.base import ControlEnvironment


class SantaFeEnvironment(ControlEnvironment):
    def __init__(
        self,
        grid: NDArray[np.int8],
        max_steps: int,
        total_food: int,
        start_pos: Tuple[int, int],
        start_dir: int,
    ) -> None:
        super().__init__("SantaFe", max_steps)
        self.original_grid = grid.copy()
        self.total_food = total_food
        self.start_pos = start_pos
        self.start_dir = start_dir

        self.dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        self.n_rows, self.n_cols = self.original_grid.shape
        self.reset()

    def reset(self) -> bool:
        self.grid = self.original_grid.copy()
        self.pos = self.start_pos
        self.dir = self.start_dir
        self.steps_taken = 0
        self.episode_reward = 0.0
        self.food_eaten = 0
        self.done = False

        # Check if starting position has food
        if self.grid[self.pos] == 1:
            self.food_eaten += 1
            self.grid[self.pos] = 0

        return self.get_observation()

    def step(self, action: str) -> Tuple[bool, float, bool, Dict[str, Any]]:
        if self.done:
            return self.get_observation(), 0.0, True, {}

        reward = 0.0
        if action == "turn-left":
            self.dir = (self.dir - 1) % 4
        elif action == "turn-right":
            self.dir = (self.dir + 1) % 4
        elif action == "move":
            dr, dc = self.dirs[self.dir]
            self.pos = (
                (self.pos[0] + dr) % self.n_rows,
                (self.pos[1] + dc) % self.n_cols,
            )
            if self.grid[self.pos] == 1:
                reward = 1.0
                self.food_eaten += 1
                self.grid[self.pos] = 0

        self.steps_taken += 1
        self.episode_reward += reward
        self.done = (
            self.steps_taken >= self.max_steps or self.food_eaten >= self.total_food
        )

        return self.get_observation(), reward, self.done, {"food": self.food_eaten}

    def get_observation(self) -> bool:
        dr, dc = self.dirs[self.dir]
        ahead = ((self.pos[0] + dr) % self.n_rows, (self.pos[1] + dc) % self.n_cols)
        return bool(self.grid[ahead] == 1)

    def get_available_actions(self) -> List[str]:
        """Return available actions (all actions always available)."""
        return ["move", "turn-left", "turn-right"]

    def get_action_space(self) -> List[str]:
        """Return complete action space."""
        return ["move", "turn-left", "turn-right"]

    def render(self) -> Any:
        if self.grid is None or self.pos is None:
            return None

        grid_viz = self.grid.astype(np.float64)
        grid_viz[self.pos] = 0.5  # Mark ant position
        return grid_viz

    @classmethod
    def from_default(cls) -> "SantaFeEnvironment":
        """
        Static factory to create environment
        """
        base_path = Path(__file__).parent
        with open(base_path / "specs.json") as f:
            specs = json.load(f)

        with open(base_path / "trail.txt") as f:
            grid_data = [[int(c) for c in line.strip()] for line in f if line.strip()]
            grid = np.array(grid_data, dtype=np.int8)
        return cls(
            grid=grid,
            max_steps=specs["max_steps"],
            total_food=specs["total_food"],
            start_pos=tuple(specs["start_pos"]),
            start_dir=specs["start_dir"],
        )


def create_santafe_env(**kwargs: Any) -> SantaFeEnvironment:
    return SantaFeEnvironment(**kwargs)
