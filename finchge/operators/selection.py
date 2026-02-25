import warnings
from typing import Optional

from finchge.core.individual import Individual
from finchge.operators.base import GESelectionStrategy


class TournamentSelection(GESelectionStrategy):
    """
    Selection strategy that chooses individuals using tournament selection.

    In tournament selection, a fixed number of individuals (tournament_size) are randomly
    chosen from the population, and the best among them (based on fitness) is selected.
    This process repeats until the desired number of individuals is selected. Selection
    pressure increases with larger tournament sizes.

    Args:
        max_best (bool): Whether higher fitness values are better. True for maximization,
            False for minimization.
        tournament_size (int): The number of individuals competing in each tournament.
            Must be >= 2. Defaults to 3.
        random_state (int) : Random State

    """

    def __init__(
        self,
        max_best: bool,
        tournament_size: int = 3,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(max_best=max_best, random_state=random_state)
        self.tournament_size = tournament_size

    def select(
        self, population_size: int, individuals: list[Individual]
    ) -> list[Individual]:
        """
        Selects a subset of individuals using tournament selection.

        Repeatedly selects a group of individuals (a "tournament") at random from the
        population, and chooses the best among them based on fitness. This process
        continues until the desired number of individuals is selected. Whether the
        best individual is determined by maximizing or minimizing fitness depends on
        the `max_best` setting.

        Args:
            population_size: The number of individuals to select for the next generation.
            individuals: A list of individuals to select from. Each must have a `fitness`
                attribute, which is expected to be a list containing a single float value
                (e.g., [fitness]).

        Returns:
            A list of selected individuals based on tournament outcomes.

        """

        if len(individuals) < self.tournament_size:
            raise ValueError(
                f"Population size ({len(individuals)}) must be >= tournament size ({self.tournament_size})"
            )

        selected_pop: list[Individual] = []
        while len(selected_pop) < population_size:
            participants = self.rng.sample(individuals, self.tournament_size)

            winner = (
                max(participants, key=lambda ind: ind.fitness)
                if self.max_best
                else min(participants, key=lambda ind: ind.fitness)
            )

            selected_pop.append(winner)
        return selected_pop


class RouletteWheelSelection(GESelectionStrategy):
    """
    Fitness-proportionate selection (roulette wheel selection).

    Roulette wheel selection is one of the earliest and most widely used
    selection mechanisms in evolutionary computation. The method assigns
    selection probabilities to individuals proportional to their fitness,
    enabling stochastic sampling while maintaining selection pressure toward
    fitter individuals.

    Each individual occupies a proportion of a conceptual roulette wheel,
    where the area assigned to each individual is proportional to its
    fitness. Selection is performed by sampling from this distribution,
    ensuring that individuals with higher fitness are more likely—but not
    guaranteed—to be selected.

    Fitness-proportionate selection originates from early genetic algorithm
    research and is commonly attributed to the foundational work on genetic
    algorithms by John Holland.

    Args:
        max_best (bool): Whether higher fitness values are better. True for maximization,
            False for minimization.
        random_state (int): Random state
    """

    def __init__(self, max_best: bool, random_state: Optional[int] = None) -> None:
        super().__init__(max_best=max_best, random_state=random_state)

    def select(
        self, population_size: int, individuals: list[Individual]
    ) -> list[Individual]:
        """
        Selects a subset of individuals using roulette wheel selection.

        This method performs fitness-proportionate selection, where individuals are assigned
        selection probabilities based on their fitness values. The fitness values are shifted
        to ensure they are non-negative, and if necessary, inverted for minimization problems.
        If the total weight of the shifted fitness values is zero or negative, the method falls
        back to uniform random selection.

        Args:
            population_size: The number of individuals to select for the next generation.
            individuals: A list of individuals to select from. Each individual must have a
                `fitness` attribute, which is expected to be a list containing a single
                float value (e.g., [fitness]).

        Returns:
            A list of selected individuals based on roulette wheel selection probabilities.
        """

        # fitness is an array or can be jsut a single fitness value
        raw_fitness = [
            ind.fitness[0] if isinstance(ind.fitness, list) else ind.fitness
            for ind in individuals
        ]
        min_fitness = min(raw_fitness)
        if min_fitness < 0:
            shifted_fitness = [fit - min_fitness + 1e-10 for fit in raw_fitness]
        else:
            shifted_fitness = [fit + 1e-10 for fit in raw_fitness]
        if not self.max_best:
            max_fitness = max(shifted_fitness)
            shifted_fitness = [max_fitness - fit for fit in shifted_fitness]

        total_weight = sum(shifted_fitness)

        if total_weight <= 0:
            warnings.warn(
                "Roulette selection fell back to uniform due to zero total weights."
            )
            return self.rng.choices(individuals, k=population_size)

        return self.rng.choices(individuals, weights=shifted_fitness, k=population_size)


class RankSelection(GESelectionStrategy):
    """
    Selection strategy that selects individuals based on their rank in the population.

    In rank selection, individuals are first sorted by fitness, and then assigned
    selection probabilities based on their rank rather than their raw fitness value.
    The selection pressure can be adjusted using the `selection_pressure` parameter,
    which influences how strongly the best individuals are favored. A higher selection
    pressure makes the selection more biased toward higher-ranked individuals.

    Args:
        max_best (bool): Whether higher fitness values are better. True for maximization,
            False for minimization.
        selection_pressure (float): The selection pressure that controls the bias toward
            higher-ranked individuals. Must be between 1.0 and 2.0. Defaults to 1.5.
        random_state (int) : random state

    Returns:
        None

    Raises:
        ValueError: If selection_pressure is not between 1.0 and 2.0.
    """

    def __init__(
        self,
        max_best: bool,
        selection_pressure: float = 1.5,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(max_best=max_best, random_state=random_state)
        if not 1.0 <= selection_pressure <= 2.0:
            raise ValueError("Selection pressure must be between 1.0 and 2.0")
        self.selection_pressure = selection_pressure

    def select(
        self, population_size: int, individuals: list[Individual]
    ) -> list[Individual]:
        """
        Selects a subset of individuals using linear rank-based selection.

        This method applies rank selection by sorting individuals based on fitness and assigning
        selection probabilities based on their rank rather than their raw fitness values. The
        selection pressure parameter controls how much more likely the best-ranked individuals
        are to be selected compared to lower-ranked ones.

        Args:
            population_size: The number of individuals to select.
            individuals: A list of individuals to select from. Each individual must have a
                `fitness` attribute, which is a list containing a single float value
                representing its fitness.


        Returns:
            A list of selected individuals from the population based on rank-weighted probabilities.
        """
        sorted_pop = sorted(
            individuals, key=lambda ind: ind.fitness[0], reverse=self.max_best
        )
        n = len(individuals)
        weights = []
        for rank in range(n):
            weight = (
                2
                - self.selection_pressure
                + (2 * (self.selection_pressure - 1) * (n - 1 - rank)) / (n - 1)
            )
            weights.append(
                weight
            )  # not sure if weights need normalization. think about this later.

        return self.rng.choices(sorted_pop, weights=weights, k=population_size)


class TruncationSelection(GESelectionStrategy):
    """
    Truncation selection strategy.

    Truncation selection is a deterministic selection mechanism in which only
    the top-performing portion of a population is allowed to reproduce. The
    selected elite subset is determined by sorting individuals according to
    fitness and selecting a predefined fraction of the best individuals.

    The population is ranked by fitness and only the top k fraction of
    individuals is retained as eligible parents. Selection from this elite
    pool may then occur randomly or deterministically.

    Args:
        max_best (bool): Whether higher fitness values are better. True for maximization,
            False for minimization.
        truncation_threshold (float):  threshold value
        random_state (int) : random state
    """

    def __init__(
        self,
        max_best: bool,
        truncation_threshold: float = 0.5,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(max_best=max_best, random_state=random_state)
        if not 0.0 < truncation_threshold <= 1.0:
            raise ValueError("Truncation threshold must be between 0.0 and 1.0")
        self.truncation_threshold = truncation_threshold

    def select(
        self, population_size: int, individuals: list[Individual]
    ) -> list[Individual]:
        """
        Selects a subset of individuals using truncation selection.

        First, the population is sorted by fitness (descending for maximization,
        ascending for minimization). Then, the top fraction specified by
        `truncation_threshold` is selected as the elite pool. Finally, individuals
        are randomly chosen from this elite pool to reach the desired population size.

        Args:
            population_size: The number of individuals to select for the next generation.
            individuals: A list of individuals to select from. Each individual must have a
                `fitness` attribute that is comparable.

        Returns:
            A list of selected individuals randomly chosen from the elite portion of
            the population.
        """
        # Sort population
        sorted_pop = sorted(
            individuals, key=lambda ind: ind.fitness, reverse=self.max_best
        )

        # Select top individuals based on threshold
        cutoff = max(2, int(len(individuals) * self.truncation_threshold))
        eligible = sorted_pop[:cutoff]

        # Randomly select from truncated population

        return self.rng.choices(eligible, k=population_size)


"""

SELECTION FUNCTIONS FOR  MULTI OBJECTIVE OPTIMISATION

"""


class NSGA2TournamentSelection(GESelectionStrategy):
    """
    Tournament selection using NSGA-II crowded-comparison operator.

    This selection method implements the parent selection strategy introduced
    in the NSGA-II multi-objective evolutionary algorithm. Selection is based
    on Pareto rank and crowding distance to balance convergence toward the
    Pareto front with population diversity.

    Individuals are compared using a crowded comparison operator:

    - Prefer individuals with lower Pareto rank.
    - If ranks are equal, prefer individuals with higher crowding
       distance.

    Tournament selection is applied using this comparison operator.

    """

    def __init__(
        self,
        max_best: bool = False,
        tournament_size: int = 2,
        exploration_prob: float = 0.1,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state, max_best=max_best)
        self.tournament_size = tournament_size
        self.exploration_prob = exploration_prob

    def select(
        self, population_size: int, individuals: list[Individual]
    ) -> list[Individual]:
        """
        Selects individuals using NSGA-II tournament selection.

        For each tournament:
        1. With probability `exploration_prob`, a random individual is chosen
        2. Otherwise, the crowded comparison operator selects the best individual
           based on Pareto rank and crowding distance

        Args:
            population_size: Number of individuals to select.
            individuals: List of individuals to select from. Each individual must have
                'rank' and 'crowding_distance' metadata.

        Returns:
            List of selected individuals.
        """
        selected = []

        for _ in range(population_size):
            actual_tournament_size = min(self.tournament_size, len(individuals))
            tournament = self.rng.sample(individuals, actual_tournament_size)
            if self.rng.random() < self.exploration_prob:
                winner = self.rng.choice(tournament)
            else:
                winner = self.crowded_comparison_operator(tournament)
            selected.append(winner)
        return selected

    def crowded_comparison_operator(self, tournament: list[Individual]) -> Individual:
        """
        Selects the best individual using NSGA-II's crowded comparison operator.

        The operator selects individuals based on lower Pareto rank (better) If same rank, higher crowding distance (better)

        Args:
            tournament: The individuals competing in the tournament.

        Returns:
            The selected winner from the tournament.
        """
        if len(tournament) == 1:
            return tournament[0]

        winner = tournament[0]
        for candidate in tournament[1:]:
            if candidate.get_meta("rank", int) < winner.get_meta("rank", int) or (
                candidate.get_meta("rank", int) == winner.get_meta("rank", int)
                and candidate.get_meta("crowding_distance", float)
                > winner.get_meta("crowding_distance", float)
            ):
                winner = candidate

        return winner


class NSGA3TournamentSelection(GESelectionStrategy):
    """
    Tournament selection using NSGA-III rank-based selection.

    This selection strategy implements the rank-based selection component of
    NSGA-III, a many-objective evolutionary algorithm designed for problems
    involving four or more objective functions.

    Selection is performed using Pareto rank only. When multiple
    individuals share the same rank, selection among them is performed
    randomly to preserve diversity. NSGA-III relies on reference
    point-based niching rather than crowding distance.
    """

    def __init__(
        self,
        tournament_size: int = 2,
        exploration_prob: float = 0.1,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            random_state=random_state, max_best=False
        )  # max_best may not be needed

        self.tournament_size = tournament_size
        self.exploration_prob = exploration_prob

    def select(
        self, population_size: int, individuals: list[Individual]
    ) -> list[Individual]:
        """
        Selects individuals using NSGA-III tournament selection.

        For each tournament a random individual is chosen with probability `exploration_prob`,
         otherwise, the rank-only operator selects from the best-ranked individuals

        Args:
            population_size: Number of individuals to select.
            individuals: List of individuals to select from. Each individual must have
                'rank' metadata.

        Returns:
            List of selected individuals.
        """
        selected: list[Individual] = []

        for _ in range(population_size):
            k = min(self.tournament_size, len(individuals))
            tournament = self.rng.sample(individuals, k)
            if self.rng.random() < self.exploration_prob:
                winner = self.rng.choice(tournament)
            else:
                winner = self.rank_only_operator(tournament)
            selected.append(winner)
        return selected

    def rank_only_operator(self, tournament: list[Individual]) -> Individual:
        """
        Selects an individual based on Pareto rank only.
        First finds the individuals with the best (lowest) rank, then randomly
        selects one from among them to maintain diversity.

        Args:
            tournament: The individuals competing in the tournament.

        Returns:
            A randomly selected individual from those with the best rank.
        """
        best_rank = min(ind.get_meta("rank", int) for ind in tournament)
        best = [ind for ind in tournament if ind.get_meta("rank", int) == best_rank]
        return self.rng.choice(best)
