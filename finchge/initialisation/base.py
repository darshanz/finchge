from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from finchge.config import FinchConfig
from finchge.core.individual import Individual
from finchge.grammar.tree_generator import TreeGenerator
from finchge.utils.random_mixin import RandomStateMixin


# Initialiser types, just a helper class with string constants to be passed to factory method make_initializer for clarity
class InitialiserType(str, Enum):
    RANDOM_GENOME = "random_genome"
    FULL = "full"
    GROW = "grow"
    PI_GROW = "pi_grow"
    RHH = "rhh"
    PTC1 = "ptc1"
    PTC2 = "ptc2"
    RAMPED_PTC2 = "ramped_ptc2"


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
    def from_config(cls, cfg: FinchConfig) -> "GEInitialiser":
        """
        Construct an initialiser from configuration.

        Args:
            cfg: FinchConfig

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
