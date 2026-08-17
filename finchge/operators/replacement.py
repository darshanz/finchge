from typing import Optional

from finchge.algorithm.utils import calculate_crowding_distance, fast_non_dominated_sort
from finchge.core.individual import Individual
from finchge.operators.base import GEReplacementStrategy


class GenerationalReplacement(GEReplacementStrategy):
    """
    Generational replacement with protected elitism for single-objective optimization.

    At each generation the entire offspring population replaces the old one, but a
    fixed number of elite individuals from the previous generation are guaranteed
    a slot in the next — they are never competed against offspring. This matches
    the behaviour of PonyGE2's ``generational`` replacement (Whitley, 1989).

    Elites are taken from the best ``elite_size`` individuals in the old population
    and placed unconditionally into the result. The remaining ``population_size -
    elite_size`` slots are filled by the best-ranked offspring.

    Args:
        max_best (bool):
            If True, higher fitness is better (maximisation).
            If False, lower fitness is better (minimisation).

    Notes:
        - Elites are selected exclusively from the old population.
        - An elite is never displaced by a better offspring (protected elitism).
        - Fitness values must be evaluated before calling ``replace``.
        - Individuals with unusable fitness are excluded from the elite pool.

    See Also:
        - RandomElitistReplacement
        - NSGA2Replacement
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
        Applies protected elitism: reserves the best ``elite_size`` individuals
        from the old population unconditionally, then fills the remaining slots
        with the best-ranked offspring.

        Args:
            new_population: Offspring generated this generation.
            old_population: Previous generation population.
            elite_size: Number of individuals from old_population to preserve.
            population_size: Target size of the returned population.

        Returns:
            List of ``population_size`` individuals: elites first, then
            the best ``population_size - elite_size`` offspring.
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
        - NSGA2Replacement
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


class NSGA2Replacement(GEReplacementStrategy):
    """NSGA-II environmental selection using Pareto rank and crowding distance.

    Implements the environmental selection step of NSGA-II (Deb et al., 2002).
    The old and new populations are combined into one pool, non-dominated
    sorting assigns Pareto ranks, and the next generation is filled front by
    front; when a front only partially fits the remaining slots, crowding
    distance breaks ties in favour of individuals in less-crowded regions of
    objective space.

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
            Performs NSGA-II environmental selection on the combined population.

    See Also:
        - fast_non_dominated_sort
        - calculate_crowding_distance
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
        Combines old and new populations, runs non-dominated sorting and
        crowding-distance ranking, and selects ``population_size`` individuals
        from successive Pareto fronts.

        Args:
            new_population: Offspring generated this generation.
            old_population: Previous generation population.
            elite_size: Unused. Accepted only for interface consistency with
                other ``GEReplacementStrategy`` implementations -- NSGA-II's
                combined-population truncation is already elitist; see the
                class docstring.
            population_size: Target size of the returned population.

        Returns:
            List of ``population_size`` individuals ranked by Pareto front
            and crowding distance.
        """

        combined = old_population + new_population
        fronts = fast_non_dominated_sort(
            individuals=combined, maximize_flags=self.maximize_flags
        )

        for front in fronts:
            calculate_crowding_distance(front)

        selected = []

        for front in fronts:
            # Sort front by crowding distance
            front.sort(
                key=lambda x: x.get_meta("crowding_distance", float), reverse=True
            )

            for ind in front:
                if len(selected) >= population_size:
                    break
                selected.append(ind)

        # fill with remaining , (not sure if needed. Should not happen though... Handle later..)
        if len(selected) < population_size:
            remaining = [ind for ind in combined if ind not in selected]
            selected.extend(remaining[: population_size - len(selected)])

        return selected
