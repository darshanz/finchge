from typing import TYPE_CHECKING, Optional

from finchge.core.individual import Individual

if TYPE_CHECKING:
    from finchge.initialisation import GEInitialiser


class Population:
    """
    Container for Individuals in an evolutionary run.

    Population is representation-agnostic and does not perform
    genotype, tree, or phenotype initialisation logic.
    """

    def __init__(
        self,
        initialiser: Optional["GEInitialiser"],
        population_size: Optional[int],
    ) -> None:
        """
        Create a Population.

        Exactly one of `individuals` or `initialiser` must be provided.

        Args:
            individuals:
                Pre-constructed individuals.
            initialiser:
                Initializer used to generate individuals.
            population_size:
                Number of individuals to generate (required if initialiser is used).
        """
        self.initialiser = initialiser
        if self.initialiser is None or population_size is None:
            raise ValueError(
                "`initialiser` and `population_size` must be provided to initialize a population."
            )

        self.population_size = population_size
        self.individuals: list[Individual] = []
        assert self.initialiser is not None
        assert population_size is not None

        for _ in range(population_size):
            ind = self.initialiser.initialise()
            self.individuals.append(ind)

    def __len__(self) -> int:
        return len(self.individuals)

    @classmethod
    def from_individuals(
        cls,
        individuals: list[Individual],
        population_size: int,
    ) -> "Population":
        """
        Construct a Population directly from a list of Individuals.

        This is used when offspring are produced by crossover/mutation
        strategies that already return fully-formed Individuals.
        """

        if not individuals:
            raise ValueError("Cannot create Population from empty individuals list")

        pop = cls.__new__(cls)
        # Core fields
        pop.individuals = individuals
        pop.population_size = population_size
        return pop
