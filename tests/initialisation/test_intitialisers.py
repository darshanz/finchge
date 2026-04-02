import random

import pytest

from finchge.config import FinchConfig
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

    init_config = {
        "experiment": {"random_seed": 42},
        "ge": {"codon_size": 120, "genome_length": 80},
    }
    config = FinchConfig.from_dict(init_config)
    initialiser = RandomGenomeInitialiser.from_config(config)
    individual = initialiser.initialise()
    assert len(individual.genotype) == 80


def test_init_using_factory_random_genome():
    """
    Initializer made using factory for random genome initialisation
    """

    init_config_ = {
        "experiment": {"random_seed": 42},
        "ge": {
            "init_type": "random_genome",
            "codon_size": 127,
            "genome_length": 100,
        },
    }
    cfg = FinchConfig.from_dict(init_config_)
    initialiser_ = make_initialiser(cfg)
    individual = initialiser_.initialise()

    assert len(individual.genotype) == 100


def test_invalid_init_type():
    """
    Should raise ValueError if unsupported intialization type is provided
    Test both that ValueError is raised AND the message is correct.
    """
    init_config_ = {
        "experiment": {"random_seed": 42},
        "ge": {
            "init_type": "wrong_init_type",
            "codon_size": 127,
            "genome_length": 100,
        },
    }
    cfg = FinchConfig.from_dict(init_config_)

    with pytest.raises(ValueError) as exc_info:
        make_initialiser(cfg)

    assert "Unknown initialisation type" in str(exc_info.value)
