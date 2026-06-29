import pytest

from finchge.core.individual import Individual
from finchge.operators.replacement import RandomElitistReplacement


def make_individual(fitness_value):
    ind = Individual(phenotype="x")
    ind.fitness = [fitness_value]
    return ind


def test_returns_correct_population_size():
    old = [make_individual(float(i)) for i in range(5)]
    new = [make_individual(float(i + 10)) for i in range(5)]
    r = RandomElitistReplacement(max_best=True, random_state=42)
    result = r.replace(
        old_population=old, new_population=new, elite_size=2, population_size=5
    )
    assert len(result) == 5


def test_negative_elite_size_raises():
    old = [make_individual(1.0)]
    r = RandomElitistReplacement(max_best=True)
    with pytest.raises(ValueError):
        r.replace(
            old_population=old, new_population=[], elite_size=-1, population_size=1
        )


def test_elite_size_exceeds_population_size_raises():
    old = [make_individual(1.0)]
    r = RandomElitistReplacement(max_best=True)
    with pytest.raises(ValueError):
        r.replace(
            old_population=old, new_population=[], elite_size=5, population_size=3
        )


def test_multi_objective_fitness_raises():
    ind = Individual(phenotype="x")
    ind.fitness = [1.0, 2.0]
    r = RandomElitistReplacement(max_best=True)
    with pytest.raises(ValueError):
        r.replace(
            old_population=[ind], new_population=[], elite_size=0, population_size=1
        )


def test_not_enough_eligible_individuals_raises():
    old = [make_individual(1.0)]
    r = RandomElitistReplacement(max_best=True)
    with pytest.raises(ValueError):
        r.replace(
            old_population=old, new_population=[], elite_size=0, population_size=5
        )


def test_maximization_elites_are_highest_fitness():
    old = [make_individual(1.0), make_individual(99.0), make_individual(50.0)]
    new = [make_individual(10.0), make_individual(20.0)]
    r = RandomElitistReplacement(max_best=True, random_state=42)
    result = r.replace(
        old_population=old, new_population=new, elite_size=1, population_size=4
    )
    # First element is always the elite
    assert result[0].fitness[0] == 99.0


def test_minimization_elites_are_lowest_fitness():
    old = [make_individual(100.0), make_individual(1.0), make_individual(50.0)]
    new = [make_individual(10.0), make_individual(20.0)]
    r = RandomElitistReplacement(max_best=False, random_state=42)
    result = r.replace(
        old_population=old, new_population=new, elite_size=1, population_size=4
    )
    assert result[0].fitness[0] == 1.0


def test_elites_always_present_in_result():
    old = [make_individual(float(i)) for i in range(5)]
    new = [make_individual(float(i + 10)) for i in range(3)]
    r = RandomElitistReplacement(max_best=True, random_state=42)
    result = r.replace(
        old_population=old, new_population=new, elite_size=2, population_size=5
    )
    # Top 2 from old (4.0 and 3.0) must be in result
    result_fitnesses = [ind.fitness[0] for ind in result]
    assert 4.0 in result_fitnesses
    assert 3.0 in result_fitnesses


def test_non_elite_slots_sampled_without_replacement():
    # rng.sample() guarantees no duplicate objects in the non-elite portion
    old = [make_individual(float(i)) for i in range(6)]
    new = [make_individual(float(i + 10)) for i in range(6)]
    r = RandomElitistReplacement(max_best=True, random_state=42)
    result = r.replace(
        old_population=old, new_population=new, elite_size=1, population_size=6
    )
    assert len(set(id(ind) for ind in result)) == 6


def test_deterministic_with_fixed_seed():
    old = [make_individual(float(i)) for i in range(5)]
    new = [make_individual(float(i + 10)) for i in range(5)]
    r1 = RandomElitistReplacement(max_best=True, random_state=7)
    r2 = RandomElitistReplacement(max_best=True, random_state=7)
    res1 = r1.replace(old, new, elite_size=1, population_size=5)
    res2 = r2.replace(old, new, elite_size=1, population_size=5)
    assert [id(ind) for ind in res1] == [id(ind) for ind in res2]


def test_random_sampling_varies_across_seeds():
    old = [make_individual(float(i)) for i in range(6)]
    new = [make_individual(float(i + 10)) for i in range(6)]
    results = set()
    for seed in range(20):
        r = RandomElitistReplacement(max_best=True, random_state=seed)
        res = r.replace(old, new, elite_size=1, population_size=5)
        results.add(
            tuple(id(ind) for ind in res[1:])
        )  # exclude elite, check non-elite variation
    assert len(results) > 1


def test_result_individuals_come_from_old_or_new():
    old = [make_individual(float(i)) for i in range(4)]
    new = [make_individual(float(i + 10)) for i in range(4)]
    r = RandomElitistReplacement(max_best=True, random_state=42)
    result = r.replace(old, new, elite_size=1, population_size=5)
    combined = old + new
    assert all(ind in combined for ind in result)
