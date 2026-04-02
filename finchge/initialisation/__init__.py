from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import GEInitialiser
    from .initialisers import (
        FullTreeInitialiser,
        GrowTreeInitialiser,
        PIGrowInitialiser,
        PTC2Initialiser,
        RampedPTC2Initialiser,
        RandomGenomeInitialiser,
        RHHInitialiser,
        RVDInitialiser,
    )


def __getattr__(name: str) -> Any:
    if name in {
        "GEInitialiser",
        "RandomGenomeInitialiser",
        "RVDInitialiser",
        "FullTreeInitialiser",
        "GrowTreeInitialiser",
        "PIGrowInitialiser",
        "SensibleInitialiser",
        "PTC2Initialiser",
        "RampedPTC2Initialiser",
    }:
        from .base import GEInitialiser
        from .initialisers import (
            FullTreeInitialiser,
            GrowTreeInitialiser,
            PIGrowInitialiser,
            PTC2Initialiser,
            RampedPTC2Initialiser,
            RandomGenomeInitialiser,
            RHHInitialiser,
            RVDInitialiser,
        )

        return locals()[name]
    raise AttributeError(name)


__all__ = [
    "GEInitialiser",
    "RandomGenomeInitialiser",
    "RVDInitialiser",
    "FullTreeInitialiser",
    "GrowTreeInitialiser",
    "PIGrowInitialiser",
    "RHHInitialiser",
    "PTC2Initialiser",
    "RampedPTC2Initialiser",
]
