from finchge.core.individual import Individual
from finchge.operators.replacement import GenerationalReplacement


def make_individual(fitness_value):
    ind = Individual(phenotype="x")
    ind.fitness = [fitness_value]
    return ind


def test_returns_correct_population_size():
    old = [make_individual(float(i)) for i in range(5)]
    new = [make_individual(float(i + 10)) for i in range(5)]
    r = GenerationalReplacement(max_best=True)
    result = r.replace(
        old_population=old, new_population=new, elite_size=2, population_size=5
    )
    assert len(result) == 5


def test_maximization_returns_highest_fitness_individuals():
    old = [make_individual(1.0), make_individual(2.0)]
    new = [make_individual(5.0), make_individual(3.0), make_individual(4.0)]
    r = GenerationalReplacement(max_best=True)
    result = r.replace(
        old_population=old, new_population=new, elite_size=1, population_size=3
    )
    fitnesses = [ind.fitness[0] for ind in result]
    assert sorted(fitnesses, reverse=True) == fitnesses
    assert 5.0 in fitnesses


def test_minimization_returns_lowest_fitness_individuals():
    old = [make_individual(10.0), make_individual(20.0)]
    new = [make_individual(1.0), make_individual(3.0), make_individual(2.0)]
    r = GenerationalReplacement(max_best=False)
    result = r.replace(
        old_population=old, new_population=new, elite_size=1, population_size=3
    )
    fitnesses = [ind.fitness[0] for ind in result]
    assert sorted(fitnesses) == fitnesses
    assert 1.0 in fitnesses


def test_old_elite_survives_when_offspring_are_weaker():
    old_best = make_individual(100.0)
    old_other = make_individual(1.0)
    new = [make_individual(5.0), make_individual(3.0)]
    r = GenerationalReplacement(max_best=True)
    result = r.replace(
        old_population=[old_best, old_other],
        new_population=new,
        elite_size=1,
        population_size=3,
    )
    assert old_best in result


def test_old_elite_can_be_displaced_by_superior_offspring():
    # Soft elitism: old elite enters the combined pool and competes.
    # If offspring are strictly better, the elite is displaced.
    old_elite = make_individual(5.0)
    new = [make_individual(10.0), make_individual(8.0)]
    r = GenerationalReplacement(max_best=True)
    result = r.replace(
        old_population=[old_elite],
        new_population=new,
        elite_size=1,
        population_size=2,
    )
    # combined = [new_a(10), new_b(8), elite(5)]
    # top 2 are the new individuals
    assert old_elite not in result


def test_invalid_old_individuals_excluded_from_elite_pool():
    invalid_ind = Individual()  # not mapped, has_usable_fitness() == False
    valid_ind = make_individual(10.0)
    new = [make_individual(1.0)]
    r = GenerationalReplacement(max_best=True)
    result = r.replace(
        old_population=[invalid_ind, valid_ind],
        new_population=new,
        elite_size=1,
        population_size=2,
    )
    assert invalid_ind not in result


def test_elite_size_zero_excludes_old_population():
    old = [make_individual(99.0)]
    new = [make_individual(1.0), make_individual(2.0)]
    r = GenerationalReplacement(max_best=True)
    result = r.replace(
        old_population=old,
        new_population=new,
        elite_size=0,
        population_size=2,
    )
    # Combined pool = new + [], so old individual never enters
    assert all(ind in new for ind in result)


def test_result_individuals_come_from_old_or_new_populations():
    old = [make_individual(float(i)) for i in range(5)]
    new = [make_individual(float(i + 10)) for i in range(5)]
    r = GenerationalReplacement(max_best=True)
    result = r.replace(
        old_population=old, new_population=new, elite_size=2, population_size=5
    )
    combined = old + new
    assert all(ind in combined for ind in result)


def test_all_individuals_returned_when_combined_fits_exactly():
    old = [make_individual(1.0), make_individual(2.0)]
    new = [make_individual(3.0)]
    r = GenerationalReplacement(max_best=True)
    # elite_size=1 adds one old
    # combined = new(1) + elite(1) = 2
    # population_size=2
    result = r.replace(
        old_population=old, new_population=new, elite_size=1, population_size=2
    )
    assert len(result) == 2
