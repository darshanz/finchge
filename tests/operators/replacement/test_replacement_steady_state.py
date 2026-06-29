from finchge.core.individual import Individual
from finchge.operators.replacement import SteadyStateReplacement


def make_individual(fitness_value):
    ind = Individual(phenotype="x")
    ind.fitness = [fitness_value]
    return ind


def test_returns_correct_population_size():
    old = [make_individual(float(i)) for i in range(5)]
    new = [make_individual(float(i + 10)) for i in range(5)]
    r = SteadyStateReplacement(max_best=True)
    result = r.replace(
        old_population=old, new_population=new, elite_size=2, population_size=5
    )
    assert len(result) == 5


def test_old_elite_always_preserved_even_when_offspring_are_better():
    # Hard elitism: preserved elites are not subject to competition with offspring.
    # This is the key behavioural difference from GenerationalReplacement.
    old_weak = make_individual(1.0)
    new = [make_individual(100.0), make_individual(99.0)]
    r = SteadyStateReplacement(max_best=True)
    result = r.replace(
        old_population=[old_weak],
        new_population=new,
        elite_size=1,
        population_size=2,
    )
    assert old_weak in result


def test_non_elite_slots_filled_by_best_offspring():
    old = [make_individual(5.0), make_individual(3.0)]
    new = [make_individual(10.0), make_individual(7.0), make_individual(1.0)]
    r = SteadyStateReplacement(max_best=True)
    result = r.replace(
        old_population=old,
        new_population=new,
        elite_size=1,
        population_size=3,
    )
    result_fitnesses = {ind.fitness[0] for ind in result}
    # old best (5.0) preserved; top 2 new (10.0, 7.0) fill remaining slots
    assert result_fitnesses == {5.0, 10.0, 7.0}


def test_weakest_offspring_dropped_when_slots_full():
    old = [make_individual(5.0)]
    new = [make_individual(10.0), make_individual(7.0), make_individual(0.5)]
    r = SteadyStateReplacement(max_best=True)
    result = r.replace(
        old_population=old,
        new_population=new,
        elite_size=1,
        population_size=3,
    )
    fitnesses = {ind.fitness[0] for ind in result}
    assert 0.5 not in fitnesses


def test_maximization_preserves_best_old_individual():
    old = [make_individual(1.0), make_individual(99.0), make_individual(50.0)]
    new = [make_individual(10.0), make_individual(20.0)]
    r = SteadyStateReplacement(max_best=True)
    result = r.replace(
        old_population=old,
        new_population=new,
        elite_size=1,
        population_size=3,
    )
    assert result[0].fitness[0] == 99.0


def test_minimization_preserves_lowest_fitness_old_individual():
    old = [make_individual(100.0), make_individual(1.0), make_individual(50.0)]
    new = [make_individual(10.0), make_individual(20.0)]
    r = SteadyStateReplacement(max_best=False)
    result = r.replace(
        old_population=old,
        new_population=new,
        elite_size=1,
        population_size=3,
    )
    assert result[0].fitness[0] == 1.0


def test_invalid_old_individuals_excluded_from_preserved():
    invalid = Individual()  # not mapped, has_usable_fitness() == False
    valid = make_individual(10.0)
    new = [make_individual(1.0), make_individual(2.0)]
    r = SteadyStateReplacement(max_best=True)
    result = r.replace(
        old_population=[invalid, valid],
        new_population=new,
        elite_size=1,
        population_size=3,
    )
    assert invalid not in result
    assert valid in result


def test_elite_size_zero_returns_only_best_offspring():
    old = [make_individual(99.0)]
    new = [make_individual(1.0), make_individual(5.0), make_individual(3.0)]
    r = SteadyStateReplacement(max_best=True)
    result = r.replace(
        old_population=old,
        new_population=new,
        elite_size=0,
        population_size=2,
    )
    # No elites preserved; only best 2 from new population
    assert all(ind in new for ind in result)
    fitnesses = {ind.fitness[0] for ind in result}
    assert fitnesses == {5.0, 3.0}
