import pytest

from finchge.grammar import GenotypeMapper, Grammar
from finchge.initialisation import RandomGenomeInitialiser
from finchge.initialisation.initialisers import RVDInitialiser


@pytest.fixture
def grammar():
    grammar_str = """
        <expr> ::= <expr> <op> <expr> | <var>
        <op> ::= + | - | * | /
        <var> ::= x | y | 1 | 2
    """
    return Grammar(grammar_str=grammar_str)


@pytest.fixture
def random_genome_initialiser():
    return RandomGenomeInitialiser(genome_length=100, codon_size=127, random_state=42)


@pytest.fixture
def mapper(grammar):
    return GenotypeMapper(grammar=grammar, random_state=42)


@pytest.fixture
def rvd_initialiser(mapper):
    return RVDInitialiser(
        genome_length=100,
        codon_size=127,
        population_size=100,
        random_state=42,
        mapper=mapper,
    )


def test_random_genome_returns_individual(random_genome_initialiser):
    """Initializer returns valid Individual object."""
    ind = random_genome_initialiser.initialise()
    assert ind is not None
    assert ind.genotype is not None


def test_random_genome_length(random_genome_initialiser):
    """Generated genome has expected length."""
    ind = random_genome_initialiser.initialise()
    assert len(ind.genotype) == random_genome_initialiser.genome_length


def test_random_genome_deterministic_seed(mapper):
    """Same seed produces identical genomes."""
    init1 = RandomGenomeInitialiser(genome_length=10, codon_size=127, random_state=42)
    init2 = RandomGenomeInitialiser(genome_length=10, codon_size=127, random_state=42)
    g1 = init1.initialise().genotype
    g2 = init2.initialise().genotype

    assert g1 == g2


def test_random_genome_values_within_range(random_genome_initialiser):
    """Codon values stay within valid byte range."""
    ind = random_genome_initialiser.initialise()
    assert all(0 <= codon <= 127 for codon in ind.genotype)


def test_rvd_returns_valid_individual(rvd_initialiser):
    """RVD produces valid individual."""
    ind = rvd_initialiser.initialise()
    assert ind is not None
    assert not ind.invalid


def test_rvd_no_duplicate_phenotypes(rvd_initialiser):
    """RVD should not generate duplicate phenotypes."""
    phenotypes = set()

    for _ in range(5):
        ind = rvd_initialiser.initialise()
        assert ind.phenotype not in phenotypes
        phenotypes.add(ind.phenotype)


def test_rvd_respects_attempt_limit(rvd_initialiser):
    """RVD raises error if uniqueness cannot be satisfied."""

    rvd_initialiser.max_attempts = 1  # force failure quickly

    with pytest.raises(RuntimeError):
        for _ in range(5):
            rvd_initialiser.initialise()


def test_rvd_deterministic_seed(mapper):
    """Same seed should produce same phenotype sequence."""

    init1 = RVDInitialiser(
        genome_length=8, codon_size=127, population_size=5, random_state=99
    )
    init2 = RVDInitialiser(
        genome_length=8, codon_size=127, population_size=5, random_state=99
    )

    init1.set_mapper(mapper)
    init2.set_mapper(mapper)

    seq1 = [init1.initialise().phenotype for _ in range(5)]
    seq2 = [init2.initialise().phenotype for _ in range(5)]

    assert seq1 == seq2


def test_rvd_internal_memory_persists(rvd_initialiser):
    """RVD remembers previous phenotypes across calls."""
    p1 = rvd_initialiser.initialise().phenotype
    p2 = rvd_initialiser.initialise().phenotype

    assert p1 != p2
