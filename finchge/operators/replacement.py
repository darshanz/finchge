from typing import Optional

from finchge.algorithm.utils import calculate_crowding_distance, fast_non_dominated_sort
from finchge.core.individual import Individual
from finchge.operators.base import GEReplacementStrategy


class GenerationalReplacement(GEReplacementStrategy):
    """
    Elitist generational replacement strategy for single-objective optimization.

    This replacement strategy implements classic generational replacement used
    in standard genetic algorithms. At each generation, the entire population
    is replaced by newly generated offspring, while preserving a fixed number
    of elite individuals from the previous population.

    Individuals are ranked using a scalar fitness value, and elites are selected
    based on either maximization or minimization of that fitness.


    Args:
        max_best (bool):
            If True, higher fitness values are considered better (maximization).
            If False, lower fitness values are considered better (minimization).

    Methods:
        replace:
            Replaces the old population with a new population while preserving
            elite individuals from the previous generation.

    Notes:
        - Assumes that each individual's ``fitness`` attribute is a scalar value.
        - Elites are selected exclusively from the old population.
        - Fitness values must already be evaluated prior to replacement.

    See Also:
        - SteadyStateReplacement
        - RandomElitistReplacement
        - NSGA2ElitistReplacement

    """

    def __init__(self, max_best: bool, random_state: Optional[int] = None) -> None:
        super().__init__(random_state=random_state)
        self.max_best = max_best

    def replace(
        self,
        new_population: list[Individual],
        old_population: list[Individual],
        elite_size: int,
        population_size: int,
    ) -> list[Individual]:
        """
        Replaces old population with new population, preserving elite individuals.

        Args:
            new_population (list[Individual]): List of new individuals
            old_population (list[Individual]): List of current individuals
            elite_size (int): Number of elite individuals to preserve
            population_size (int): Target size of the population

        Returns:
            list: New population with elites
        """
        valid_old = [ind for ind in old_population if ind.has_usable_fitness()]
        valid_old.sort(key=lambda ind: ind.get_scalar_fitness(), reverse=self.max_best)

        elites = valid_old[:elite_size]

        combined_population = new_population + elites

        combined_population.sort(
            key=lambda ind: ind.sort_key(self.max_best),
            reverse=self.max_best,
        )

        return combined_population[:population_size]


class SteadyStateReplacement(GEReplacementStrategy):
    """
    Steady-state replacement strategy for single-objective optimization.

    This replacement strategy implements a steady-state evolutionary model,
    where only a subset of individuals is replaced at each generation rather
    than replacing the entire population. A fixed number of elite individuals
    from the current population are preserved, while the remaining slots are
    filled with the best individuals from the newly generated population.

    Selection is based on a scalar fitness value, using either maximization or
    minimization as specified at construction time.

    Args:
        max_best (bool):
            If True, higher fitness values are considered better (maximization).
            If False, lower fitness values are considered better (minimization).

    Methods:
        replace:
            Preserves elite individuals from the old population and replaces
            the remaining individuals with the best-performing offspring.

    Notes:
        - Assumes that each individual's ``fitness`` attribute is a scalar value.
        - Fitness values must be evaluated prior to replacement.
        - Elites are selected exclusively from the old population.
        - The number of preserved individuals is controlled by ``elite_size``.

    See Also:
        - GenerationalReplacement
        - RandomElitistReplacement
        - NSGA2ElitistReplacement

    """

    def __init__(self, max_best: bool, random_state: Optional[int] = None) -> None:
        super().__init__(random_state=random_state)
        self.max_best = max_best

    def replace(
        self,
        new_population: list[Individual],
        old_population: list[Individual],
        elite_size: int,
        population_size: int,
    ) -> list[Individual]:
        """
        Replaces worst individuals in old population with new individuals.
        Preserves elite_size best individuals.
        """
        valid_old = [ind for ind in old_population if ind.has_usable_fitness()]
        valid_old.sort(key=lambda ind: ind.get_scalar_fitness(), reverse=self.max_best)

        preserved = valid_old[:elite_size]

        new_population.sort(
            key=lambda ind: ind.sort_key(self.max_best),
            reverse=self.max_best,
        )

        replacements = new_population[: population_size - elite_size]
        return preserved + replacements


class RandomElitistReplacement(GEReplacementStrategy):
    """
    Random replacement strategy with elitism for single-objective optimization.

    This replacement strategy preserves a fixed number of elite individuals from
    the current population and fills the remaining population slots by randomly
    sampling from the non-elite individuals of the old population and the newly
    generated individuals.

    Elites are selected based on a scalar fitness value, using either maximization
    or minimization as specified at construction time. Diversity is introduced
    through random sampling rather than explicit distance or dominance measures.

    Args:
        max_best (bool):
            If True, higher fitness values are considered better (maximization).
            If False, lower fitness values are considered better (minimization).

    Methods:
        replace:
            Preserves elite individuals and randomly selects remaining individuals
            to form the next generation.

    Raises:
        ValueError:
            If ``elite_size`` is invalid.
        ValueError:
            If any individual does not have a single-objective fitness
            representation (fitness list length != 1).
        ValueError:
            If there are not enough eligible individuals to fill the population.

    Notes:
        - Assumes that fitness values are evaluated prior to replacement.
        - Does not use Pareto dominance, crowding distance, or reference-point
          niching.
        - Random sampling is performed without replacement.

    See Also:
        - GenerationalReplacement
        - SteadyStateReplacement
        - NSGA2ElitistReplacement
    """

    def __init__(self, max_best: bool, random_state: Optional[int] = None) -> None:
        super().__init__(random_state=random_state)
        self.max_best = max_best

    def replace(
        self,
        new_population: list[Individual],
        old_population: list[Individual],
        elite_size: int,
        population_size: int,
    ) -> list[Individual]:
        if elite_size < 0 or elite_size > population_size:
            raise ValueError("Invalid elite_size")

        # Validate single-objective fitness
        for ind in old_population:
            # Check for multi-objective only if there is fitness (skip unevaluated or invalid ones)
            if ind.fitness and len(ind.fitness) != 1:
                raise ValueError(
                    "RandomReplacement requires single-objective fitness "
                    "(fitness must be a list of length 1)"
                )

        valid_old = [ind for ind in old_population if ind.has_usable_fitness()]

        sorted_old = sorted(
            valid_old,
            key=lambda ind: ind.get_scalar_fitness(),
            reverse=self.max_best,
        )

        elites = sorted_old[:elite_size]

        eligible = sorted_old[elite_size:] + new_population

        needed = population_size - elite_size
        if needed > len(eligible):
            raise ValueError("Not enough individuals to fill population")

        random_selection = self.rng.sample(eligible, needed)

        return elites + random_selection


"""

REPLACEMENT STRATEGY FOR MULTI-OBJECTIVE OPTIMIZATION

"""


class NSGA2ElitistReplacement(GEReplacementStrategy):
    """NSGA-II elitist replacement strategy using Pareto rank and crowding distance.

    This replacement strategy implements the environmental selection step of
    NSGA-II. It preserves elite individuals from the previous population and
    fills the remaining population by selecting individuals from successive
    Pareto fronts, prioritizing diversity via crowding distance.

    The strategy assumes that:
    - Non-dominated sorting has already been performed.
    - Each individual has a valid Pareto rank stored in ``meta["rank"]``.
    - Crowding distance is computed per front using NSGA-II rules.

    This class is **not compatible with NSGA-III**, as it relies on crowding
    distance for diversity preservation. For NSGA-III, reference-point-based
    environmental selection must be used instead.

    Args:
        maximize_flags (list[bool]):
            List indicating whether each objective should be maximized (True)
            or minimized (False). This is used to perform non-dominated sorting
            when combining parent and offspring populations.

    Methods:
        replace:
            Performs elitist environmental selection according to NSGA-II.

    See Also:
        - fast_non_dominated_sort
        - calculate_crowding_distance
        - NSGA3Replacement
    """

    def __init__(
        self, maximize_flags: list[bool], random_state: Optional[int] = None
    ) -> None:
        super().__init__(random_state=random_state)
        self.maximize_flags = maximize_flags

    def replace(
        self,
        new_population: list[Individual],
        old_population: list[Individual],
        elite_size: int,
        population_size: int,
    ) -> list[Individual]:
        """
        Replacement strategy that preserves elite solutions and maintains diversity.
        """

        # Sort by rank, handle None values
        valid_old = [ind for ind in old_population if ind.has_meta("rank")]

        valid_old.sort(key=lambda x: x.get_meta("rank", int))
        elite_individuals = valid_old[:elite_size]
        elite_set = set(elite_individuals)

        combined = old_population + new_population
        fronts = fast_non_dominated_sort(
            individuals=combined, maximize_flags=self.maximize_flags
        )

        for front in fronts:
            if any(ind.get_meta("crowding_distance", float) is None for ind in front):
                calculate_crowding_distance(front)

        selected = []
        selected.extend(elite_individuals)

        # Then fill from fronts (excluding elites that were already added)
        for front in fronts:
            # Sort front by crowding distance
            front.sort(
                key=lambda x: x.get_meta("crowding_distance", float), reverse=True
            )

            for ind in front:
                if len(selected) >= population_size:
                    break

                # Add if not already in elites
                if ind not in elite_set:
                    selected.append(ind)

        # fill with remaining , (not sure if needed. Should not happen though... Handle later..)
        if len(selected) < population_size:
            remaining = [ind for ind in combined if ind not in selected]
            selected.extend(remaining[: population_size - len(selected)])

        return selected
