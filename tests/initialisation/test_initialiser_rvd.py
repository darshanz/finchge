import pytest

from finchge.core.population import Population
from finchge.grammar import Grammar
from finchge.grammar.mapper import GenotypeMapper
from finchge.initialisation.initialisers import RVDInitialiser


@pytest.fixture
def grammar() -> Grammar:
    grammar_str = """
    <expr> ::= <expr> <op> <expr>
             | ( <expr> )
             | <var>
    <op> ::= + | - | * | /
    <var> ::= x | y | 1 | 2
    """
    gr = Grammar(grammar_str=grammar_str)
    gr.analyze()
    return gr


@pytest.fixture
def mapper(grammar: Grammar) -> GenotypeMapper:
    return GenotypeMapper(
        grammar=grammar,
        max_tree_depth=10,
        max_wraps=1,
        random_state=42,
    )


@pytest.fixture
def rvd_initialiser(mapper: GenotypeMapper) -> RVDInitialiser:
    init = RVDInitialiser(
        genome_length=30,
        codon_size=127,
        population_size=20,
        random_state=42,
    )
    init.set_mapper(mapper)
    return init


def test_rvd_creates_valid_individual(rvd_initialiser: RVDInitialiser) -> None:
    ind = rvd_initialiser.initialise()

    assert ind is not None
    assert ind.invalid is False
    assert ind.phenotype is not None
    assert ind.genotype is not None
    assert ind.tree is not None


def test_rvd_population_is_all_valid(rvd_initialiser: RVDInitialiser) -> None:
    population = Population(initialiser=rvd_initialiser, population_size=20)

    assert len(population.individuals) == 20
    assert all(not ind.invalid for ind in population.individuals)
    assert all(ind.phenotype is not None for ind in population.individuals)


def test_rvd_population_has_unique_phenotypes(rvd_initialiser: RVDInitialiser) -> None:
    population = Population(initialiser=rvd_initialiser, population_size=20)

    phenotypes = [ind.phenotype for ind in population.individuals]

    assert all(p is not None for p in phenotypes)
    assert len(set(phenotypes)) == len(phenotypes)


def test_rvd_individuals_have_used_genome_metadata(
    rvd_initialiser: RVDInitialiser,
) -> None:
    population = Population(initialiser=rvd_initialiser, population_size=20)

    for ind in population.individuals:
        assert ind.used_genome is not None
        assert ind.used_codon_count >= 0
        assert ind.genotype is not None
        assert ind.used_codon_count <= len(ind.genotype)


def test_rvd_is_reproducible_for_same_seed(
    grammar: Grammar,
) -> None:
    mapper1 = GenotypeMapper(
        grammar=grammar,
        max_tree_depth=10,
        max_wraps=1,
        random_state=42,
    )
    mapper2 = GenotypeMapper(
        grammar=grammar,
        max_tree_depth=10,
        max_wraps=1,
        random_state=42,
    )

    init1 = RVDInitialiser(
        genome_length=30,
        codon_size=127,
        population_size=15,
        random_state=42,
    )
    init1.set_mapper(mapper1)

    init2 = RVDInitialiser(
        genome_length=30,
        codon_size=127,
        population_size=15,
        random_state=42,
    )
    init2.set_mapper(mapper2)

    pop1 = Population(initialiser=init1, population_size=15)
    pop2 = Population(initialiser=init2, population_size=15)

    phenotypes1 = [ind.phenotype for ind in pop1.individuals]
    phenotypes2 = [ind.phenotype for ind in pop2.individuals]

    assert phenotypes1 == phenotypes2


def test_rvd_changes_with_different_seed(grammar: Grammar) -> None:
    mapper1 = GenotypeMapper(
        grammar=grammar,
        max_tree_depth=10,
        max_wraps=1,
        random_state=42,
    )
    mapper2 = GenotypeMapper(
        grammar=grammar,
        max_tree_depth=10,
        max_wraps=1,
        random_state=99,
    )

    init1 = RVDInitialiser(
        genome_length=30,
        codon_size=127,
        population_size=15,
        random_state=42,
    )
    init1.set_mapper(mapper1)

    init2 = RVDInitialiser(
        genome_length=30,
        codon_size=127,
        population_size=15,
        random_state=99,
    )
    init2.set_mapper(mapper2)

    pop1 = Population(initialiser=init1, population_size=15)
    pop2 = Population(initialiser=init2, population_size=15)

    phenotypes1 = [ind.phenotype for ind in pop1.individuals]
    phenotypes2 = [ind.phenotype for ind in pop2.individuals]

    assert phenotypes1 != phenotypes2


def test_rvd_reset_clears_internal_state(
    mapper: GenotypeMapper,
) -> None:
    init = RVDInitialiser(
        genome_length=30,
        codon_size=127,
        population_size=10,
        random_state=42,
    )
    init.set_mapper(mapper)

    pop1 = Population(initialiser=init, population_size=10)
    phenotypes1 = [ind.phenotype for ind in pop1.individuals]

    init.reset()

    pop2 = Population(initialiser=init, population_size=10)
    phenotypes2 = [ind.phenotype for ind in pop2.individuals]

    assert len(phenotypes1) == 10
    assert len(phenotypes2) == 10
    assert all(p is not None for p in phenotypes2)


def test_rvd_raises_when_mapper_is_missing() -> None:
    init = RVDInitialiser(
        genome_length=30,
        codon_size=127,
        population_size=10,
        random_state=42,
    )

    with pytest.raises(RuntimeError, match="Mapper must be set"):
        init.initialise()
