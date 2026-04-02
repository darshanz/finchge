from finchge.config import FinchConfig, Keys
from finchge.initialisation.base import GEInitialiser, GETreeInitialiser
from finchge.initialisation.initialisers import (
    FullTreeInitialiser,
    GrowTreeInitialiser,
    PIGrowInitialiser,
    PTC2Initialiser,
    RampedPTC2Initialiser,
    RandomGenomeInitialiser,
    RHHInitialiser,
    RVDInitialiser,
)


def make_initialiser(
    cfg: FinchConfig,
) -> GEInitialiser | GETreeInitialiser:
    """
    Create an initialiser instance based on configuration.

    Supported initialisation algorithms:

        - "random_genome" : Canonical GE random genome initialisation
        - "rvd"           : Canonical GE random genome initialisation with all valid and no duplicates
        - "full"          : Full tree initialisation
        - "grow"          : Grow tree initialisation
        - "pi_grow"       : Position-independent Grow initialisation
        - "rhh"           : RHH Initializer or Ramped Half-and-Half initialisation for GE
        - "ptc2"          : Probabilistic Tree Creation 2
        - "rampedptc2"     : Ramped Probabilistic Tree Creation 2

    Args:
        cfg:
            Configuration instance (FinchConfig).
        random_state:
            Optional RNG seed.

    Returns:
        GEInitializer or GETreeInitializer
    """

    init_type = cfg.ge.get(
        Keys.INIT_TYPE, "random_genome"
    ).lower()  # use random_genome as default init type

    # Registry mapping initialiser names to classes
    initialiser_registry: dict[str, type[GEInitialiser | GETreeInitialiser]] = {
        "random_genome": RandomGenomeInitialiser,
        "rvd": RVDInitialiser,
        "full": FullTreeInitialiser,
        "grow": GrowTreeInitialiser,
        "pi_grow": PIGrowInitialiser,
        "rhh": RHHInitialiser,
        "ptc2": PTC2Initialiser,
        "ramped_ptc2": RampedPTC2Initialiser,
    }

    if init_type not in initialiser_registry:
        raise ValueError(
            f"Unknown initialisation type '{init_type}'. "
            f"Valid options are: {sorted(initialiser_registry.keys())}. "
            f"For config-based initialisation is not supported for custom intialisation method"
        )

    print(f"Initialisation type: {init_type}")

    initialiser_cls = initialiser_registry[init_type]

    return initialiser_cls.from_config(cfg)
