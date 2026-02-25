import pytest

from finchge.core.individual import Individual
from finchge.operators.crossover import UniformCrossover


def make_individual(genotype, used_codon_count=None):
    return Individual(genotype=genotype, used_codon_count=used_codon_count)


def test_cross_raises_error_if_genotype_missing():
    # the crossover method raises a ValueError
    # if either parent does not have a genotype defined.
    parent1 = make_individual(None)
    parent2 = make_individual([1, 2, 3])
    crossover = UniformCrossover(crossover_proba=1.0, random_state=42)
    with pytest.raises(ValueError):
        crossover.cross(parent1, parent2)


def test_cross_returns_two_offspring():
    # the crossover operation always returns
    # exactly two offspring individuals.
    parent1 = make_individual([1, 2, 3, 4], used_codon_count=4)
    parent2 = make_individual([5, 6, 7, 8], used_codon_count=4)

    crossover = UniformCrossover(crossover_proba=1.0, random_state=42)
    child1, child2 = crossover.cross(parent1, parent2)
    assert child1 is not None
    assert child2 is not None


def test_no_crossover_when_probability_zero():
    # when crossover probability is zero,
    # offspring are exact copies of the parents.
    parent1 = make_individual([1, 2, 3, 4], used_codon_count=4)
    parent2 = make_individual([5, 6, 7, 8], used_codon_count=4)
    crossover = UniformCrossover(crossover_proba=0.0, random_state=42)
    child1, child2 = crossover.cross(parent1, parent2)
    assert child1.genotype == parent1.genotype
    assert child2.genotype == parent2.genotype


def test_uniform_crossover_swaps_genes_when_probability_one():
    # when crossover probability is one,
    # genes are independently swapped between parents.
    parent1 = make_individual([1, 2, 3, 4], used_codon_count=4)
    parent2 = make_individual([5, 6, 7, 8], used_codon_count=4)
    crossover = UniformCrossover(crossover_proba=1.0, random_state=1)
    child1, child2 = crossover.cross(parent1, parent2)
    assert child1.genotype != parent1.genotype or child2.genotype != parent2.genotype


def test_crossover_respects_within_used_true():
    # when within_used is True,
    # crossover only occurs within the used portion of each genome.
    parent1 = make_individual([1, 2, 3, 4, 99, 99], used_codon_count=3)
    parent2 = make_individual([5, 6, 7, 8, 99, 99], used_codon_count=3)

    crossover = UniformCrossover(crossover_proba=1.0, within_used=True, random_state=42)

    child1, child2 = crossover.cross(parent1, parent2)

    assert child1.genotype[3:] == parent1.genotype[3:]
    assert child2.genotype[3:] == parent2.genotype[3:]


def test_crossover_allows_full_genome_when_within_used_false():
    # when within_used is False,
    # crossover may occur anywhere within the full genome.
    parent1 = make_individual([1, 2, 3, 4, 10, 11], used_codon_count=2)
    parent2 = make_individual([5, 6, 7, 8, 12, 13], used_codon_count=2)

    crossover = UniformCrossover(
        crossover_proba=1.0, within_used=False, random_state=42
    )

    child1, child2 = crossover.cross(parent1, parent2)

    assert len(child1.genotype) == len(parent1.genotype)
    assert len(child2.genotype) == len(parent2.genotype)


def test_crossover_handles_zero_used_codon_count():
    # if used_codon_count is zero,
    # no gene positions are swapped and parents are copied. within uses is true by default
    parent1 = make_individual([1, 2, 3], used_codon_count=0)
    parent2 = make_individual([4, 5, 6], used_codon_count=0)

    crossover = UniformCrossover(crossover_proba=1.0, random_state=42)
    child1, child2 = crossover.cross(parent1, parent2)
    assert child1.genotype == parent1.genotype
    assert child2.genotype == parent2.genotype


def test_crossover_is_deterministic_with_fixed_random_state():
    # using the same random_state produces
    # identical offspring genotypes, ensuring reproducibility.
    parent1 = make_individual([1, 2, 3, 4], used_codon_count=4)
    parent2 = make_individual([5, 6, 7, 8], used_codon_count=4)

    crossover1 = UniformCrossover(crossover_proba=1.0, random_state=123)
    crossover2 = UniformCrossover(crossover_proba=1.0, random_state=123)
    child1_a, child2_a = crossover1.cross(parent1, parent2)
    child1_b, child2_b = crossover2.cross(parent1, parent2)
    assert child1_a.genotype == child1_b.genotype
    assert child2_a.genotype == child2_b.genotype


def test_crossover_does_not_modify_parent_genotypes():
    # crossover produces offspring without modifying
    # the original parent genotypes.
    parent1 = make_individual([1, 2, 3, 4], used_codon_count=4)
    parent2 = make_individual([5, 6, 7, 8], used_codon_count=4)
    original_p1 = parent1.genotype.copy()
    original_p2 = parent2.genotype.copy()
    crossover = UniformCrossover(crossover_proba=1.0, random_state=42)
    crossover.cross(parent1, parent2)
    assert parent1.genotype == original_p1
    assert parent2.genotype == original_p2
