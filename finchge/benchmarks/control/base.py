from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from finchge.utils.random_mixin import RandomStateMixin


class ControlEnvironment(RandomStateMixin, ABC):
    """
    Base class for all control problem environments.

    Each environment maintains its own state during an episode.
    A new instance should be created for each episode/evaluation.
    """

    def __init__(self, name: str, max_steps: int, random_state: Optional[Any] = None):
        super().__init__(random_state=random_state)
        self.name = name
        self.max_steps = max_steps
        self.steps_taken = 0
        self.episode_reward = 0.0
        self.done = False

    @abstractmethod
    def reset(self) -> Any:
        """
        Reset environment to initial state.

        Returns:
            Initial observation
        """
        pass

    @abstractmethod
    def step(self, action: str) -> Tuple[Any, float, bool, dict[str, Any]]:
        """
        Execute action in environment.

        Args:
            action: Action string from grammar

        Returns:
            Tuple of (next_observation, reward, done, info)
        """
        pass

    @abstractmethod
    def get_observation(self) -> Any:
        """Get current observation."""
        pass

    @abstractmethod
    def get_available_actions(self) -> list[str]:
        """Return available actions in current state."""
        pass

    @abstractmethod
    def get_action_space(self) -> list[str]:
        """Return all possible actions."""
        pass

    def is_done(self) -> bool:
        """Check if episode is finished."""
        return self.done

    def render(self) -> Optional[NDArray[np.float64]]:
        """Render current state (optional)."""
        return None
