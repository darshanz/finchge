from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Optional

import numpy as np

from finchge.runners.base import PhenotypeRunner, T


class ControlRunner(PhenotypeRunner, ABC, Generic[T]):
    """
    Base class for control problem runners.

    Type parameter T represents the environment type.
    """

    def __init__(
        self,
        env_factory: Callable[[], T],
        n_episodes: int = 1,
        optimal_fitness: Optional[float] = None,
        random_state: Optional[Any] = None,
    ) -> None:
        """
        Initialize control runner.

        Args:
            env_factory: Function that creates new environment instances
            n_episodes: Number of episodes to run per evaluation
            optimal_fitness: Known optimal fitness value
        """
        super().__init__(random_state=random_state)
        self.env_factory = env_factory
        self.n_episodes = n_episodes
        self.optimal_fitness = optimal_fitness
        self._last_rewards: list[float] = []

    @abstractmethod
    def _run_episode(self, phenotype: str, env: T) -> float:
        """
        Run one episode in the environment.

        Returns:
            Total reward for the episode
        """
        pass

    def run(
        self, phenotype: str, context_hints: Optional[set[str]] = None
    ) -> dict[str, Any]:
        """
        Run multiple episodes and return rewards.
        """
        episode_rewards = []

        for _ in range(self.n_episodes):
            env = self.env_factory()
            reward = self._run_episode(phenotype, env)
            episode_rewards.append(reward)

        self._last_rewards = episode_rewards

        # Predictions are the rewards
        y_pred = np.array(episode_rewards, dtype=np.float64)

        # Targets are optimal fitness (or zeros if unknown)
        if self.optimal_fitness is not None:
            y_true = np.full_like(y_pred, self.optimal_fitness, dtype=np.float64)
        else:
            y_true = np.zeros_like(y_pred, dtype=np.float64)

        context = {
            "y_pred": np.array(episode_rewards),
            "y_true": y_true,
            "phenotype": phenotype,
        }
        return context

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata about the control problem."""
        return {
            "n_episodes": self.n_episodes,
            "optimal_fitness": self.optimal_fitness,
            "last_rewards": self._last_rewards,
        }

    def __getstate__(self) -> dict[str, Any]:
        # Let RandomStateMixin handle RNGs first
        state = super().__getstate__()

        # Add our own attributes
        state.update(
            {
                "env_factory": self.env_factory,
                "n_episodes": self.n_episodes,
                "optimal_fitness": self.optimal_fitness,
                "_last_rewards": self._last_rewards,
            }
        )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Let RandomStateMixin restore RNGs first
        super().__setstate__(state)
        # All attributes are already restored via __dict__.update
        # But we can add post-processing if needed
        if not hasattr(self, "_last_rewards"):
            self._last_rewards = []

    @property
    def provided_context_keys(self) -> set[str]:
        return super().provided_context_keys
