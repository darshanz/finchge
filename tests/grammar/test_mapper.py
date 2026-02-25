import random

import pytest

from finchge.grammar import Grammar
from finchge.grammar.mapper import GenotypeMapper


def test_genotype_mapper_determinism():
    grammar_str = """
        <string> ::= <letter> | <letter> <string>
        <letter> ::= _ | [a-z]
        """
    # Create same grammar
    grammar = Grammar(grammar_str)

    # Create two mappers with same seed
    mapper1 = GenotypeMapper(grammar=grammar, random_state=42)
    mapper2 = GenotypeMapper(grammar=grammar, random_state=42)

    # Test with same genotype
    genotype = [1, 2, 3, 4, 5]

    result1 = mapper1.map(genotype)
    result2 = mapper2.map(genotype)

    # These might differ if mapper has non-deterministic internals
    assert result1.phenotype == result2.phenotype
    assert result1.used_genome == result2.used_genome

    print(f"Mapper test: {result1.phenotype == result2.phenotype}")


def test_genotype_mapper_round_trip_consistency() -> None:
    """
    Core GE invariant:

        map(genome) -> tree -> reverse_map(tree) -> genome' -> map(genome')
        must preserve phenotype and tree structure.

    Genome equality is NOT required.
    """

    random.seed(60)

    grammar_str = """
    <string> ::= <letter> | <letter> <string>
    <letter> ::= _ | [a-z]
    """

    grammar = Grammar(grammar_str)

    mapper = GenotypeMapper(
        grammar=grammar,
        max_recursion_depth=20,
        max_wraps=5,
    )

    genome = [random.randint(0, 127) for _ in range(20)]

    # First mapping
    result_1 = mapper.map(genome)
    assert not result_1.invalid, "Initial mapping unexpectedly invalid"

    # Reverse mapping
    genome_2 = mapper.reverse_map(
        result_1.tree,
        codon_size=127,
        pad_to_length=len(genome),
        pad_mode="zeros",
    )

    # Second mapping
    result_2 = mapper.map(genome_2)
    assert not result_2.invalid, "Reverse-mapped genome produced invalid mapping"

    # Invariants
    assert result_1.phenotype == result_2.phenotype, (
        "Phenotype mismatch after map -- reverse_map -- map.\n"
        f"Original phenotype: {result_1.phenotype!r}\n"
        f"Round-trip phenotype: {result_2.phenotype!r}"
    )

    assert result_1.tree.to_string() == result_2.tree.to_string(), (
        "Tree structure mismatch after round-trip.\n"
        f"Original tree:\n{result_1.tree.to_string()}\n"
        f"Round-trip tree:\n{result_2.tree.to_string()}"
    )


@pytest.mark.parametrize("genome_length", [10, 20, 50])
def test_genotype_mapper_round_trip_consistency_various_lengths(
    genome_length: int,
) -> None:
    """
    Round-trip consistency across different genome lengths.
    """

    random.seed(78)

    grammar_str = """
    <string> ::= <letter> | <letter> <string>
    <letter> ::= _ | [a-z]
    """

    grammar = Grammar(grammar_str)

    mapper = GenotypeMapper(
        grammar=grammar,
        max_recursion_depth=20,
        max_wraps=5,
    )

    genome = [random.randint(0, 127) for _ in range(genome_length)]

    # Decode
    result_1 = mapper.map(genome)
    assert not result_1.invalid, "Initial mapping unexpectedly invalid"

    # Encode
    genome_2 = mapper.reverse_map(
        result_1.tree,
        codon_size=127,
        pad_to_length=genome_length,
        pad_mode="zeros",
    )

    # Decode again
    result_2 = mapper.map(genome_2)
    assert not result_2.invalid, "Reverse-mapped genome produced invalid mapping"

    # GE invariants
    assert result_1.phenotype == result_2.phenotype, (
        "Phenotype mismatch after round-trip.\n"
        f"Original phenotype: {result_1.phenotype!r}\n"
        f"Round-trip phenotype: {result_2.phenotype!r}"
    )

    assert result_1.tree.to_string() == result_2.tree.to_string(), (
        "Tree structure mismatch after round-trip.\n"
        f"Original tree:\n{result_1.tree.to_string()}\n"
        f"Round-trip tree:\n{result_2.tree.to_string()}"
    )
