from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from finchge.core.individual import Individual
from finchge.utils.random_mixin import RandomStateMixin
from finchge.grammar.tree_generator import TreeGenerator


class InitializerType(str, Enum):
    RANDOM_GENOME = "random_genome"
    FULL = "full"
    GROW = "grow"
    PI_GROW = "pi_grow"
    RHH = "rhh"
    PTC1 = "ptc1"
    PTC2 = "ptc2"
    PTC2D = "ptc2d"


class GEInitialiser(RandomStateMixin, ABC):
    """
    Base class for population initialisation strategies.
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(random_state=random_state)

    @abstractmethod
    def initialise(self) -> Individual:
        """
        Create and return a new Individual.
        Grammar is only required by tree based inititalizers.
        For RandomGenomeInitialiser, there is no need to pass grammar.

        Returns:
            Individual: _description_
        """
        ...

    @classmethod
    @abstractmethod
    def from_config(
        cls, cfg: dict[str, Any], random_state: int | None = None
    ) -> "GEInitialiser":
        """
        Construct an initialiser from configuration.

        Args:
            random_state:
            cfg:

        Returns:
        """
        raise NotImplementedError


class GETreeInitialiser(GEInitialiser):
    """
    Base class for tree-based population initialisation strategies.

    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(random_state=random_state)

    def set_tree_generator(self, tree_generator: "TreeGenerator") -> None: ...
