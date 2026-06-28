import pytest

from finchge.core.individual import Individual
from finchge.operators.mutation import GaussianMutation


def make_individual(genome):
    return Individual(genotype=genome)


def test_raises_error_without_genotype():
    """Requires genotype; raises ValueError when missing"""
    ind = Individual()
    mutation = GaussianMutation(mutation_probability=0.5, random_state=42)
    with pytest.raises(ValueError):
        mutation.mutate(ind)


def test_probability_out_of_range_raises():
    """Probability outside [0, 1] raises ValueError"""
    with pytest.raises(ValueError):
        GaussianMutation(mutation_probability=2.0)


def test_returns_new_individual():
    """Mutate returns a new object, not the original"""
    ind = make_individual([10, 20, 30])
    mutation = GaussianMutation(mutation_probability=0.5, random_state=42)
    assert mutation.mutate(ind) is not ind


def test_does_not_modify_parent():
    """Parent genotype is unchanged after Gaussian mutation"""
    genome = [10, 20, 30]
    ind = make_individual(genome.copy())
    mutation = GaussianMutation(mutation_probability=1.0, std_dev=5.0, random_state=42)
    mutation.mutate(ind)
    assert ind.genotype == genome


def test_genome_length_preserved():
    """Genome length is unchanged after Gaussian mutation"""
    ind = make_individual([1, 2, 3, 4, 5])
    mutation = GaussianMutation(mutation_probability=1.0, std_dev=10.0, random_state=42)
    child = mutation.mutate(ind)
    assert len(child.genotype) == 5


def test_values_are_non_negative():
    """Gaussian noise is clamped; all output values are >= 0"""
    # Starting from 0 maximises the chance of negative values before clamping
    ind = make_individual([0] * 100)
    mutation = GaussianMutation(
        mutation_probability=1.0, std_dev=100.0, random_state=42
    )
    child = mutation.mutate(ind)
    assert all(v >= 0 for v in child.genotype)


def test_zero_probability_no_change():
    """Probability zero produces a clone of the parent"""
    genome = [10, 20, 30, 40]
    ind = make_individual(genome.copy())
    mutation = GaussianMutation(mutation_probability=0.0, random_state=42)
    child = mutation.mutate(ind)
    assert child.genotype == genome


def test_zero_std_dev_preserves_values_when_selected():
    """Adding zero-sigma Gaussian noise leaves all selected values unchanged"""
    genome = [5, 10, 15, 20]
    ind = make_individual(genome.copy())
    mutation = GaussianMutation(mutation_probability=1.0, std_dev=0.0, random_state=42)
    child = mutation.mutate(ind)
    assert child.genotype == genome


def test_all_output_values_are_integers():
    """Gaussian noise is rounded; all output values are integers"""
    ind = make_individual([50] * 30)
    mutation = GaussianMutation(mutation_probability=1.0, std_dev=10.0, random_state=42)
    child = mutation.mutate(ind)
    assert all(isinstance(v, int) for v in child.genotype)


def test_large_std_dev_changes_values():
    """With large std_dev and p=1.0, values typically diverge from original"""
    genome = [100] * 20
    ind = make_individual(genome.copy())
    mutation = GaussianMutation(
        mutation_probability=1.0, std_dev=1000.0, random_state=42
    )
    child = mutation.mutate(ind)
    assert child.genotype != genome


def test_deterministic_with_fixed_seed():
    """Same random_state produces identical mutation results"""
    ind = make_individual([10, 20, 30, 40])
    m1 = GaussianMutation(mutation_probability=0.5, std_dev=5.0, random_state=7)
    m2 = GaussianMutation(mutation_probability=0.5, std_dev=5.0, random_state=7)
    assert m1.mutate(ind).genotype == m2.mutate(ind).genotype


def test_higher_std_dev_produces_larger_deviations():
    """Greater std_dev leads to larger average absolute change in codon values"""
    genome = [100] * 50
    ind_low = make_individual(genome.copy())
    ind_high = make_individual(genome.copy())

    m_low = GaussianMutation(mutation_probability=1.0, std_dev=1.0, random_state=42)
    m_high = GaussianMutation(mutation_probability=1.0, std_dev=50.0, random_state=42)

    child_low = m_low.mutate(ind_low)
    child_high = m_high.mutate(ind_high)

    deviation_low = sum(abs(a - b) for a, b in zip(child_low.genotype, genome))
    deviation_high = sum(abs(a - b) for a, b in zip(child_high.genotype, genome))

    assert deviation_high > deviation_low
