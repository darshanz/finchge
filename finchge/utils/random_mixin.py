import logging
import random
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState


class RandomStateMixin:
    """
    Mixin to add random state control
    All classes using randomness either using python random or numpy random should extend from RandomMixin.
    And, should provide random_state parmeter in the constructor, Otherwise, the determinism is not guaranteed across runs.
    """

    def __init__(
        self,
        *args: tuple[Any, ...],
        random_state: Optional[Any] = None,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(*args, **kwargs)

        self._original_random_state = random_state

        self._init_rngs()

    def _init_rngs(self) -> None:
        if self._original_random_state is None:
            self._rng = random.Random()
            self._np_rng = np.random.RandomState()
            self._effective_seed: int | str | None = None
        elif isinstance(self._original_random_state, int):
            self._rng = random.Random(self._original_random_state)
            self._np_rng = np.random.RandomState(self._original_random_state)
            self._effective_seed = self._original_random_state
        elif isinstance(self._original_random_state, random.Random):
            self._rng = self._original_random_state
            try:
                state = self._original_random_state.getstate()
                self._effective_seed = (
                    state[1][0]
                    if len(state) > 1 and len(state[1]) > 0
                    else "RNG_instance"
                )
            except (AttributeError, TypeError, IndexError) as e:
                logging.debug(f"Could not extract seed from RNG: {e}")
                self._effective_seed = "RNG_instance"
            self._np_rng = np.random.RandomState()
        else:
            self._rng = random.Random()
            self._np_rng = np.random.RandomState()
            self._effective_seed = None

    @property
    def rng(self) -> random.Random:
        return self._rng

    @property
    def np_rng(self) -> RandomState:
        return self._np_rng

    def get_seed_info(self) -> dict[str, Any | None]:
        """
        Get information about the seed for logging
        """
        return {
            "original_random_state": self._original_random_state,
            "effective_seed": self._effective_seed,
            "seed_type": (
                type(self._original_random_state).__name__
                if self._original_random_state
                else None
            ),
        }

    def __getstate__(self) -> dict[str, Any]:
        """
        Called when pickling. Removes unpicklable RNGs and stores only the seed.
        """
        state = self.__dict__.copy()

        # Remove unpicklable RNG objects
        state.pop("_rng", None)
        state.pop("_np_rng", None)

        # _original_random_state (the seed) is already picklable and stays
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Called when unpickling. Restores object state and reinitializes RNGs.
        """
        # Restore all attributes
        self.__dict__.update(state)

        # Reinitialize RNGs from the stored seed
        self._init_rngs()

    def inject_rng(self, rng: Any, np_rng: RandomState | None = None) -> None:
        self._rng = rng
        if np_rng is not None and hasattr(self, "_np_rng"):
            self._np_rng = np_rng
