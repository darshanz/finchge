import itertools
import logging
from typing import Any, Generator, cast

import numpy as np
from numpy.typing import NDArray

from finchge.algorithm.base import BaseAlgorithm
from finchge.core.individual import Individual
from finchge.fitness.fitness_evaluator import FitnessEvaluator

ALGORITHMS = {
    "GeneticAlgorithm": {"min_objectives": 1, "max_objectives": 1},
    "NSGA2": {"min_objectives": 2},
    "NSGA3": {"min_objectives": 2},
}


def validate_algorithm_fitness_match(
    algorithm: BaseAlgorithm, fitness_evaluator: FitnessEvaluator
) -> None:
    num_objectives = len(fitness_evaluator.fitness_functions)
    algorithm_name = algorithm.__class__.__name__
    expectations = ALGORITHMS.get(algorithm_name)

    if expectations:
        if (
            "min_objectives" in expectations
            and num_objectives < expectations["min_objectives"]
        ):
            raise ValueError(
                f"{algorithm_name.upper()} requires at least {expectations['min_objectives']} objectives, got {num_objectives}."
            )
        if (
            "max_objectives" in expectations
            and num_objectives > expectations["max_objectives"]
        ):
            raise ValueError(
                f"{algorithm_name.upper()} supports at most {expectations['max_objectives']} objectives, got {num_objectives}. Check fitness evaluator."
            )
    else:
        logging.warning(
            f"Warning: Unknown algorithm '{algorithm}'. Skipping fitness-function validation. Ensure the algorithm handles {num_objectives} objectives correctly."
        )


def dominates(ind1: Individual, ind2: Individual, maximize_flags: list[bool]) -> bool:
    """
    Determines whether ind1 dominates ind2 in a multi-objective context.

    An individual dominates another if it is no worse in all objectives and
    strictly better in at least one, according to the specified optimization directions.

    Args:
        ind1: First individual with a 'fitness' attribute (scalar or list).
        ind2: Second individual with a 'fitness' attribute (scalar or list).
        maximize_flags (list of bool): List indicating whether each objective should
            be maximized (True) or minimized (False).

    Returns:
        bool: True if ind1 dominates ind2, False otherwise.
    """
    better_in_any = False

    # Get objectives from fitness values
    f1 = ind1.fitness if isinstance(ind1.fitness, list) else [ind1.fitness]
    f2 = ind2.fitness if isinstance(ind2.fitness, list) else [ind2.fitness]

    # Send invalids to the last pareto front instead of crashing the sort.
    ind1_usable = len(f1) == len(maximize_flags)
    ind2_usable = len(f2) == len(maximize_flags)
    if not ind1_usable:
        return False
    if not ind2_usable:
        return True

    for i, (val1, val2) in enumerate(zip(f1, f2)):
        # Compare based on optimization direction
        if maximize_flags[i]:
            if val1 < val2:  # ind1 is worse
                return False
            elif val1 > val2:  # ind1 is better
                better_in_any = True
        else:
            if val1 > val2:  # ind1 is worse
                return False
            elif val1 < val2:  # ind1 is better
                better_in_any = True

    return better_in_any


def fast_non_dominated_sort(
    individuals: list[Individual],
    maximize_flags: list[bool],
) -> list[list[Individual]]:
    if not individuals:
        return []

    n = len(individuals)

    fronts: list[list[Individual]] = [[]]
    dominated_sets: list[list[int]] = [[] for _ in range(n)]
    domination_counts: list[int] = [0] * n

    for i, p in enumerate(individuals):
        for j, q in enumerate(individuals):
            if i == j:
                continue
            if dominates(p, q, maximize_flags):
                dominated_sets[i].append(j)
            elif dominates(q, p, maximize_flags):
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            p.meta["rank"] = 0
            fronts[0].append(p)

    if not fronts[0]:
        raise ValueError("No individuals in first front")

    current = 0
    while current < len(fronts):
        next_front: list[Individual] = []
        for p in fronts[current]:
            p_idx = individuals.index(p)
            for q_idx in dominated_sets[p_idx]:
                domination_counts[q_idx] -= 1
                if domination_counts[q_idx] == 0:
                    q = individuals[q_idx]
                    q.meta["rank"] = current + 1
                    next_front.append(q)

        if next_front:
            fronts.append(next_front)
        current += 1

    return fronts


def calculate_crowding_distance(
    front: list[Individual],
) -> None:
    """Calculate crowding distance for individuals in a front"""

    if not front:
        return

    # individuals without usable fitness get zero distance and are excluded from geometry calculation
    usable = [ind for ind in front if ind.has_usable_fitness()]
    for ind in front:
        if ind not in usable:
            ind.meta["crowding_distance"] = 0.0

    if not usable:
        return

    if len(usable) <= 2:
        for ind in usable:
            ind.meta["crowding_distance"] = float("inf")
        return

    num_objectives = len(front[0].fitness)

    # Initialize distances
    for ind in front:
        ind.meta["crowding_distance"] = 0.0

    for m in range(num_objectives):
        # Sort by m-th objective
        front_sorted = sorted(front, key=lambda ind: ind.fitness[m])

        # Boundary points
        front_sorted[0].meta["crowding_distance"] = float("inf")
        front_sorted[-1].meta["crowding_distance"] = float("inf")

        f_min = front_sorted[0].fitness[m]
        f_max = front_sorted[-1].fitness[m]
        f_range = f_max - f_min

        if f_range == 0:
            continue

        for i in range(1, len(front_sorted) - 1):
            prev_f = front_sorted[i - 1].fitness[m]
            next_f = front_sorted[i + 1].fitness[m]
            distance = (next_f - prev_f) / f_range
            front_sorted[i].meta["crowding_distance"] += float(distance)


def environmental_selection_nsga3(
    individuals: list[Individual],
    population_size: int,
    reference_points: NDArray[np.float64],
    maximize_flags: list[bool],
    epsilon: float = 1e-12,  # TODO EPSLON setting in config
    rng: Any = None,
) -> list[Individual]:
    if not individuals:
        return []

    # Non-dominated sorting
    fronts: list[list[Individual]] = fast_non_dominated_sort(
        individuals, maximize_flags
    )

    # Fill next population front by front
    chosen: list[Individual] = []
    last_front: list[Individual] | None = None

    for front in fronts:
        if len(chosen) + len(front) <= population_size:
            chosen.extend(front)
        else:
            last_front = front
            break

    if len(chosen) == population_size:
        return chosen

    if last_front is None:
        return chosen[:population_size]

    # Normalize objectives (on ALL candidates involved in niching)
    # Using the combined set (or at least chosen + last_front) for normalization
    R = reference_points
    assert R is not None

    pool = chosen + last_front
    norm_objs = normalize_objectives_minimization(
        pool, maximize_flags
    )  # shape (len(pool), M)

    # Split back
    chosen_norm = norm_objs[: len(chosen)]
    last_norm = norm_objs[len(chosen) :]

    # Associate chosen + last_front to reference points
    chosen_assoc, _ = associate(chosen_norm, R)
    last_assoc, last_dist = associate(last_norm, R)

    # Niching: fill remaining slots from last_front
    remaining = population_size - len(chosen)
    picked_from_last = niching_select(
        last_front=last_front,
        last_assoc=last_assoc,
        last_dist=last_dist,
        ref_points=R,
        niche_count_init=chosen_assoc,
        k=remaining,
        rng=rng,
    )

    chosen.extend(picked_from_last)
    return chosen


def normalize_objectives_minimization(
    inds: list[Individual],
    maximize_flags: list[bool],
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """
    Returns normalized objectives for NSGA-III association.
    We convert maximization objectives to minimization by negating them.
    Then normalize each objective to [0, 1] using (x - min) / (max - min + eps).
    """

    if not inds:
        raise ValueError("No individuals provided")

    for ind in inds:
        if ind.fitness is None:
            raise ValueError("Individual missing fitness")

    objs = np.array(
        [ind.fitness for ind in inds],
        dtype=np.float64,
    )

    for m, is_max in enumerate(maximize_flags):
        if is_max:
            objs[:, m] = -objs[:, m]

    ideal = np.min(objs, axis=0)
    worst = np.max(objs, axis=0)

    denom = (worst - ideal) + epsilon
    norm = (objs - ideal) / denom

    return cast(NDArray[np.float64], norm)


def associate(
    normalized_objectives: NDArray[np.float64],  # (N, M)
    reference_points: NDArray[np.float64],  # (K, M)
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """
    For each solution, find the closest reference point by perpendicular distance.
    Returns:
        assoc_idx: (N,) index of reference point
        dist:      (N,) perpendicular distance
    """
    n = normalized_objectives.shape[0]
    k = reference_points.shape[0]

    assoc = np.empty(n, dtype=np.int_)
    dist = np.empty(n, dtype=np.float64)

    for i in range(n):
        obj = normalized_objectives[i]
        best_j = 0
        best_d = float("inf")

        for j in range(k):
            ref = reference_points[j]
            denom = float(np.dot(ref, ref))
            if denom <= 0.0:
                continue
            proj = float(np.dot(obj, ref)) / denom
            d = float(np.linalg.norm(obj - proj * ref))
            if d < best_d:
                best_d = d
                best_j = j

        assoc[i] = best_j
        dist[i] = best_d

    return assoc, dist


def niching_select(
    last_front: list[Individual],
    last_assoc: NDArray[np.int_],  # (L,)
    last_dist: NDArray[np.float64],  # (L,)
    ref_points: NDArray[np.float64],  # (K, M)
    niche_count_init: NDArray[np.int_],  # (|chosen|,) assoc indices for already chosen
    k: int,
    rng: Any = None,
) -> list[Individual]:
    """
    NSGA-III niching selection for the last front.
    - niche counts start from already chosen individuals
    - iteratively pick least-crowded reference point, then pick a candidate associated with it
    """
    K = ref_points.shape[0]

    # Initialize niche counts from already chosen
    niche_count = np.zeros(K, dtype=np.int_)
    for a in niche_count_init:
        niche_count[int(a)] += 1

    # Candidates grouped by reference point
    buckets: list[list[int]] = [[] for _ in range(K)]
    for idx, rp in enumerate(last_assoc):
        buckets[int(rp)].append(idx)

    selected: list[Individual] = []
    selected_mask = np.zeros(len(last_front), dtype=bool)

    while len(selected) < k:
        # Choose reference points with minimum niche count that still have candidates
        available_rps = [
            rp for rp in range(K) if any(not selected_mask[i] for i in buckets[rp])
        ]
        if not available_rps:
            break

        min_count = min(int(niche_count[rp]) for rp in available_rps)
        least_crowded = [
            rp for rp in available_rps if int(niche_count[rp]) == min_count
        ]
        rp = rng.choice(least_crowded)

        # Among candidates associated with rp, choose:
        # - if niche_count[rp] == 0: pick closest (min perpendicular distance)
        # - else: pick random (NSGA-III rule)
        candidates = [i for i in buckets[rp] if not selected_mask[i]]
        if not candidates:
            # Shouldn't happen due to available_rps filter
            niche_count[rp] += 1
            continue

        if int(niche_count[rp]) == 0:
            best_i = min(candidates, key=lambda i: float(last_dist[i]))
        else:
            best_i = rng.choice(candidates)

        selected_mask[best_i] = True
        selected.append(last_front[best_i])
        niche_count[rp] += 1

    return selected


# Reference points


def generate_reference_points(
    num_objectives: int, num_divisions: int
) -> NDArray[np.float64]:
    """
    Structured reference points (Das & Dennis).
    Returns shape (K, M) float array.
    """

    if num_objectives < 2:
        raise ValueError("NSGA-III requires at least 2 objectives")
    if num_objectives == 2 and num_divisions < 2:
        raise ValueError("For 2 objectives, num_divisions must be >= 2")

    def recursive_combinations(n: int, k: int) -> Generator[list[int], None, None]:
        for c in itertools.combinations_with_replacement(range(n + 1), k - 1):
            yield [c[0]] + [c[i] - c[i - 1] for i in range(1, len(c))] + [n - c[-1]]

    points: list[NDArray[np.float64]] = []
    for part in recursive_combinations(num_divisions, num_objectives):
        points.append(np.array(part, dtype=np.float64) / float(num_divisions))

    return np.stack(points, axis=0)
