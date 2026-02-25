import random

import pytest

from finchge.initialisation.factory import make_initialiser


@pytest.fixture
def py_rng() -> random.Random:
    return random.Random(42)


def test_random_genome_init():
    """Test if individual is correctly intialized by RandomGenomeInitialiser"""
    from finchge.initialisation.initialisers import RandomGenomeInitialiser

    initialiser = RandomGenomeInitialiser(codon_size=120, genome_length=100)
    individual = initialiser.initialise()
    assert len(individual.genotype) == 100


def test_random_genome_init_from_conf():
    """Test if individual is correctly intialized by RandomGenomeInitialiser from config"""

    from finchge.initialisation.initialisers import RandomGenomeInitialiser

    init_config = {"codon_size": 120, "genome_length": 80}
    initialiser = RandomGenomeInitialiser.from_config(init_config)
    individual = initialiser.initialise()
    assert len(individual.genotype) == 80


def test_init_using_factory_random_genome():
    """
    Initializer made using factory for random genome initialisation
    """

    init_config_ = {
        "init_type": "random_genome",
        "codon_size": 127,
        "genome_length": 100,
    }
    initialiser_ = make_initialiser(init_config_)
    individual = initialiser_.initialise()

    assert len(individual.genotype) == 100


def test_invalid_init_type():
    """
    Should raise ValueError if unsupported intialization type is provided
    Test both that ValueError is raised AND the message is correct.
    """
    init_config_ = {
        "init_type": "wrong_init_type",
        "codon_size": 127,
        "genome_length": 100,
    }

    with pytest.raises(ValueError) as exc_info:
        make_initialiser(init_config_)

    assert "Unknown initialisation type" in str(exc_info.value)
