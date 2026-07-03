from typing import Optional

from finchge.config.config import ConfigError, FinchConfig, Keys
from finchge.core.individual import Individual
from finchge.grammar import GenotypeMapper
from finchge.grammar.tree_generator import TreeGenerator
from finchge.initialisation.base import GEInitialiser, GETreeInitialiser


class RandomGenomeInitialiser(GEInitialiser):
    """
    Random genome initialiser for Grammatical Evolution.

    Generates Individuals with a fixed-length integer genome where
    each codon is sampled uniformly from [0, codon_size].

    This initialiser requires genome_length and codon_size to be
    provided explicitly via configuration or constructor.
    """

    def __init__(
        self, genome_length: int, codon_size: int, random_state: Optional[int] = None
    ) -> None:
        super().__init__(random_state=random_state)
        if genome_length <= 0:
            raise ValueError(f"genome_length must be > 0, got {genome_length}")
        if codon_size < 1:
            raise ValueError(f"codon_size must be >= 1, got {codon_size}")

        self.genome_length = genome_length
        self.codon_size = codon_size

    @classmethod
    def from_config(cls, config: FinchConfig) -> "RandomGenomeInitialiser":
        """
        Create a RandomGenomeInitialiser using FinchConfig object

        Args:
            config: FinchConfig instance

        Returns:
            RandomGenomeInitialiser

        Raises:
            ConfigError: If required configuration keys are missing or invalid.
        """
        try:
            random_state = config.experiment[Keys.RANDOM_SEED]
            genome_length = config.ge[Keys.GENOME_LENGTH]
            codon_size = config.ge[Keys.CODON_SIZE]
        except KeyError as e:
            raise ConfigError(
                f"Missing required GE config key for RandomGenomeInitialiser: {e.args[0]}"
            )

        return cls(
            genome_length=genome_length,
            codon_size=codon_size,
            random_state=random_state,
        )

    def initialise(self) -> Individual:
        genome = [
            self.rng.randint(0, self.codon_size) for _ in range(self.genome_length)
        ]
        return Individual(genotype=genome)


class RVDInitialiser(GEInitialiser):
    """
    Random Valid Distinct (RVD) initialiser.

    Generates individuals by sampling random genomes while rejecting
    invalid mappings and duplicate phenotypes.
    - RVD initialisation [[Nicolau, 2017](https://link.springer.com/article/10.1007/s10710-017-9309-9#Bib1)].

    Note:
        RVD initializer instance is supposed to be used once per run or the stored RVD state may cause issues.
        To reset the stored state call reset() , if same RVDInitializer instance has to be reused.
    """

    def __init__(
        self,
        genome_length: int,
        codon_size: int,
        population_size: int,
        random_state: Optional[int] = None,
        mapper: GenotypeMapper | None = None,
    ) -> None:
        super().__init__(random_state=random_state)

        if genome_length <= 0:
            raise ValueError(f"genome_length must be > 0, got {genome_length}")
        if codon_size < 1:
            raise ValueError(f"codon_size must be >= 1, got {codon_size}")

        self.genome_length = genome_length
        self.codon_size = codon_size
        self.population_size = population_size
        self.max_attempts = population_size * 42  # arbitrarily chosen for now
        self.mapper = mapper
        # RVD state
        self._seen_phenotypes: set[str] = set()
        self._attempts = 0

    def reset(self) -> None:
        """
        If same RVD instance is used for multiple runs. must reset

        """
        self._seen_phenotypes.clear()
        self._attempts = 0

    @classmethod
    def from_config(cls, config: FinchConfig) -> "RVDInitialiser":
        """
        Create a RVDInitializer from the [ge] config section.

        Args:
            config: FinchConfig instance

        Returns:
            RandomGenomeInitialiser

        Raises:
            ConfigError: If required configuration keys are missing or invalid.
        """
        try:
            random_state = config.experiment[Keys.RANDOM_SEED]
            genome_length = config.ge[Keys.GENOME_LENGTH]
            codon_size = config.ge[Keys.CODON_SIZE]
            population_size = config.ge[Keys.POPULATION_SIZE]
        except KeyError as e:
            raise ConfigError(
                f"Missing required GE config key for RandomGenomeInitialiser: {e.args[0]}"
            )

        return cls(
            genome_length=genome_length,
            codon_size=codon_size,
            population_size=population_size,
            random_state=random_state,
        )

    def set_mapper(self, mapper: "GenotypeMapper") -> None:
        self.mapper = mapper

    def initialise(self) -> Individual:
        if self.mapper is None:
            raise RuntimeError("Mapper must be set before initialisation.")

        while True:
            if self._attempts > self.max_attempts:
                raise RuntimeError(
                    f"Exceeded {self.max_attempts} attempts during RVD initialisation. "
                    "Grammar may not support enough unique valid individuals."
                )

            genome = [
                self.rng.randint(0, self.codon_size) for _ in range(self.genome_length)
            ]

            result = self.mapper.map(genome)

            self._attempts += 1

            if result.invalid:
                continue

            if result.phenotype in self._seen_phenotypes:
                continue

            self._seen_phenotypes.add(str(result.phenotype))

            return Individual(
                genotype=genome,
                phenotype=result.phenotype,
                used_genome=result.used_genome,
                used_codon_count=result.used_codon_count,
                invalid=result.invalid,
                tree=result.tree.to_string(),
            )


class FullTreeInitialiser(GETreeInitialiser):
    """
    Full tree initialiser as defined by Koza (1992).

    Produces trees where all internal nodes expand until max depth.
    Terminals are only allowed at the maximum depth.

    """

    def __init__(
        self,
        init_min_depth: int,
        init_max_depth: int,
        strict_full: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)

        if init_min_depth <= 0 or init_max_depth <= 0:
            raise ValueError("Depth values must be > 0")

        if init_min_depth > init_max_depth:
            raise ValueError("init_min_depth cannot be greater than init_max_depth")

        self.init_min_depth = init_min_depth
        self.init_max_depth = init_max_depth
        self.strict_full = strict_full

        self.tree_generator: TreeGenerator | None = None
        self._index = 0

    @classmethod
    def from_config(cls, config: FinchConfig) -> "FullTreeInitialiser":
        try:
            random_state = config.experiment[Keys.RANDOM_SEED]
            min_depth = config.ge[Keys.INIT_MIN_DEPTH]
            max_depth = config.ge[Keys.INIT_MAX_DEPTH]
            strict_full = config.ge.get(Keys.INIT_TREE_STRICT_FULL, True)
        except KeyError as e:
            raise ConfigError(
                "Missing required GE config keys for Full initialiser"
            ) from e

        return cls(
            init_min_depth=min_depth,
            init_max_depth=max_depth,
            strict_full=strict_full,
            random_state=random_state,
        )

    def set_tree_generator(self, tree_generator: TreeGenerator) -> None:
        self.tree_generator = tree_generator

    def _pick_depth(self) -> int:
        if self.init_min_depth >= self.init_max_depth:
            return self.init_max_depth

        depth_range = self.init_max_depth - self.init_min_depth + 1
        depth = self.init_min_depth + (self._index % depth_range)
        self._index += 1
        return depth

    def initialise(self) -> Individual:
        if self.tree_generator is None:
            raise RuntimeError(
                "TreeGenerator has not been set for FullTreeInitializer."
            )

        depth = self._pick_depth()

        tree = self.tree_generator.generate_tree_full(
            max_depth=depth,
            strict=self.strict_full,
            rng=self.rng,
        )

        return Individual.from_tree(tree)


class GrowTreeInitialiser(GETreeInitialiser):
    """
    Grow tree initialiser as defined by Koza (1992).

    Allows productions to terminate early below max depth,
    resulting in irregular tree shapes.

    """

    def __init__(
        self,
        init_min_depth: int,
        init_max_depth: int,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)

        if init_min_depth <= 0 or init_max_depth <= 0:
            raise ValueError("Depth values must be > 0")

        if init_min_depth > init_max_depth:
            raise ValueError("init_min_depth cannot be greater than init_max_depth")

        self.init_min_depth = init_min_depth
        self.init_max_depth = init_max_depth

        self.tree_generator: TreeGenerator | None = None
        self._index = 0

    @classmethod
    def from_config(cls, config: FinchConfig) -> "GrowTreeInitialiser":
        try:
            random_state = config.experiment[Keys.RANDOM_SEED]
            min_depth = config.ge[Keys.INIT_MIN_DEPTH]
            max_depth = config.ge[Keys.INIT_MAX_DEPTH]
        except KeyError as e:
            raise ConfigError(
                "Missing required GE config keys for Grow initialiser"
            ) from e

        return cls(
            init_min_depth=min_depth,
            init_max_depth=max_depth,
            random_state=random_state,
        )

    def set_tree_generator(self, tree_generator: TreeGenerator) -> None:
        self.tree_generator = tree_generator

    def _pick_depth(self) -> int:
        if self.init_min_depth >= self.init_max_depth:
            return self.init_max_depth

        depth_range = self.init_max_depth - self.init_min_depth + 1
        depth = self.init_min_depth + (self._index % depth_range)
        self._index += 1
        return depth

    def initialise(self) -> Individual:
        if self.tree_generator is None:
            raise RuntimeError(
                "TreeGenerator has not been set for GrowTreeInitializer."
            )

        depth = self._pick_depth()

        tree = self.tree_generator.generate_tree_grow(
            max_depth=depth,
            rng=self.rng,
        )

        return Individual.from_tree(tree)


class RHHInitialiser(GETreeInitialiser):
    """
    Ramped Half-and-Half (RHH) Initialisation for Grammatical Evolution.

    This implementation of RHH Initialisation is the Grammatical Evolution
    is based on "Sensible Initialization" approach as an analogue of Koza’s
    Ramped Half-and-Half (RHH) method originally developed for tree-based
    Genetic Programming. It generates an initial population of derivation
    trees using a combination of Full and Grow construction strategies
    across a range of depth limits, promoting structural diversity while
    respecting grammar constraints.

    In this approach, grammar production rules assume the role of GP
    functions in controlling tree growth. Recursive production rules are
    preferentially selected during Full initialisation to ensure tree
    expansion until the specified depth limit is reached. During Grow
    initialisation, both recursive and terminating productions may be
    selected, allowing trees of irregular shape to be generated.

    Production feasibility is guided by grammar analysis, including
    recursion detection and minimum derivation depth calculation, ensuring
    that generated trees can terminate within the allowed depth budget.


    References:
        Koza, J. R. (1992).
        Genetic Programming: On the Programming of Computers by Means of Natural Selection.
        MIT Press.

        Ryan, C., Collins, J. J., & O’Neill, M. (1998).
        Grammatical Evolution: Evolving Programs for an Arbitrary Language.
        In Proceedings of the First European Workshop on Genetic Programming.

        Ryan, C., & Azad, R. M. A. (2003).
        Sensible Initialisation in Grammatical Evolution.
        In Grammatical Evolution: Evolutionary Automatic Programming in an Arbitrary Language.
        Springer.

        Fenton, M., McDermott, J., Fagan, D., Forstenlechner, S., Hemberg, E., & O’Neill, M. (2017).
        PonyGE2: Grammatical Evolution in Python.
        arXiv:1703.08535.
    """

    def __init__(
        self,
        init_max_depth: int,
        population_size: int,
        strict_full: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)

        if init_max_depth <= 0:
            raise ValueError(f"init_max_depth must be > 0, got {init_max_depth}")

        self.init_max_depth = init_max_depth
        self.population_size = population_size
        self.strict_full = strict_full

        self._index = 0  # Tracks number of individuals generated
        self.tree_generator: TreeGenerator | None = None

        # Only used in stochastic mode
        self._schedule: list[tuple[int, bool]] | None = None

        # inject rng
        if self.tree_generator:
            self.inject_tree_generator_rng()

    @classmethod
    def from_config(cls, config: FinchConfig) -> "RHHInitialiser":
        """
        Create a RampedHalfAndHalfInitializer from configuration.

        Args:
            config: FinchConfig instance

        Returns:
            RampedHalfAndHalfInitializer

        Raises:
            ConfigError: If required configuration keys are missing or invalid.
        """
        try:
            random_state = config.experiment[Keys.RANDOM_SEED]
            init_max_depth = config.ge[Keys.INIT_MAX_DEPTH]
            population_size = config.ge[Keys.POPULATION_SIZE]
            strict_full = config.ge.get(Keys.INIT_TREE_STRICT_FULL, True)
        except KeyError as e:
            raise ConfigError(
                f"Missing required GE config key for RHH initialiser: {e.args[0]}"
            )

        return cls(
            init_max_depth=init_max_depth,
            strict_full=strict_full,
            population_size=population_size,
            random_state=random_state,
        )

    def _get_depths(self) -> list[int]:
        if self.tree_generator is None:
            raise RuntimeError(
                "TreeGenerator must be set before building RHH schedule."
            )

        grammar = self.tree_generator.grammar
        grammar.analyze()

        min_ramp = grammar.compute_min_ramp(
            population_size=self.population_size,
            max_init_depth=self.init_max_depth,
        )

        if min_ramp is None:
            raise RuntimeError("Grammar cannot compute a valid minimum ramp depth.")

        # ramping starts from min_ramp
        depths = list(range(min_ramp, self.init_max_depth + 1))
        if not depths:
            raise RuntimeError(
                f"No valid RHH ramp depths available. Computed min_ramp={min_ramp} and init_max_depth={self.init_max_depth}. "
                "Increase init_max_depth or check grammar ramp feasibility."
            )
        return depths

    def _build_koza_schedule(self) -> None:
        """
        Build schedule for Ramped Half-and-Half behaviour.
        """

        # depth definition based on min and max depth params
        depths = self._get_depths()

        if self.population_size < 2:
            raise RuntimeError("Population size too small for RHH.")

        #  RHH needs an even population
        size = self.population_size
        if size % 2:
            size += 1

        # If pop too small to cover all depths
        if size // 2 < len(depths):
            depths = depths[: size // 2]

        # Allocation calculation
        times = (size // 2) // len(depths)
        remainder = (size // 2) - (times * len(depths))

        schedule: list[tuple[int, bool]] = []

        # Main allocation
        for depth in depths:
            for _ in range(times):
                schedule.append((depth, False))  # Grow
                schedule.append((depth, True))  # Full

        # Remainder allocation
        if remainder:
            shuffled_depths = depths[:]
            self.rng.shuffle(shuffled_depths)

            for i in range(remainder):
                depth = shuffled_depths.pop()
                schedule.append((depth, False))  # Grow
                schedule.append((depth, True))  # Full

        self.rng.shuffle(schedule)

        self._schedule = schedule

    def _pick_params(self) -> tuple[int, bool]:
        if self._schedule is None:
            self._build_koza_schedule()

        assert self._schedule
        if self._index >= len(self._schedule):
            self._index = 0

        params = self._schedule[self._index]
        self._index += 1

        return params

    def set_tree_generator(self, tree_generator: TreeGenerator) -> None:
        self.tree_generator = tree_generator
        self.inject_tree_generator_rng()

    def initialise(self) -> Individual:
        if self.tree_generator is None:
            raise RuntimeError(
                "TreeGenerator has not been set for "
                f"{self.__class__.__name__}. "
                "Call set_tree_generator() before initialise()."
            )

        depth, force_full = self._pick_params()

        if force_full:
            tree = self.tree_generator.generate_tree_full(
                max_depth=depth,
                strict=self.strict_full,
                rng=self.rng,
            )
        else:
            tree = self.tree_generator.generate_tree_grow(
                max_depth=depth,
                rng=self.rng,
            )

        return Individual.from_tree(tree)

    def inject_tree_generator_rng(self) -> None:
        if self.tree_generator is not None:
            if hasattr(self.tree_generator, "_rng"):
                self.tree_generator._rng = self._rng
            if hasattr(self.tree_generator, "_np_rng") and hasattr(self, "_np_rng"):
                self.tree_generator._np_rng = self._np_rng


class PIGrowInitialiser(GETreeInitialiser):
    """
    Position-Independent Grow (PI-Grow) tree initialiser.

    Generates derivation trees using grow-style initialisation with
    position-independent expansion.
    """

    def __init__(
        self,
        init_max_depth: int,
        population_size: int,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)

        if init_max_depth <= 0:
            raise ValueError("Depth values must be > 0")

        # configuration for intiializaers
        self.init_max_depth = init_max_depth
        self.population_size = population_size

        # pre run mutable state
        self._index = 0
        self.tree_generator: TreeGenerator | None = None

    @classmethod
    def from_config(cls, config: FinchConfig) -> "PIGrowInitialiser":
        """
        Create a PI-Grow initialiser from GE configuration.

        Args:
            config: FinchConfig instance

        Returns:
            PIGrowInitializer

        Raises:
            ConfigError: If required configuration keys are missing or invalid.
        """
        try:
            random_state = config.experiment[Keys.RANDOM_SEED]
            init_max_depth = config.ge[Keys.INIT_MAX_DEPTH]
            population_size = config.ge[Keys.POPULATION_SIZE]
        except KeyError as e:
            raise ConfigError(
                f"Missing required GE config key for PI-Grow initialiser: {e.args[0]}"
            )

        return cls(
            init_max_depth=init_max_depth,
            population_size=population_size,
            random_state=random_state,
        )

    def set_tree_generator(self, tree_generator: TreeGenerator) -> None:
        self.tree_generator = tree_generator

    def initialise(self) -> Individual:
        if self.tree_generator is None:
            raise RuntimeError(
                "TreeGenerator has not been set for "
                f"{self.__class__.__name__}. "
                "Call set_tree_generator() before initialize()."
            )
        depth = self._pick_depth()

        tree = self.tree_generator.generate_tree_pi_grow(
            max_depth=depth,
            rng=self.rng,
        )

        return Individual.from_tree(tree)

    def _pick_depth(self) -> int:
        if self.tree_generator is None:
            raise RuntimeError(
                "TreeGenerator must be set before PI-Grow initialisation."
            )

        grammar = self.tree_generator.grammar
        grammar.analyze()

        min_ramp = grammar.compute_min_ramp(
            population_size=self.population_size,
            max_init_depth=self.init_max_depth,
        )

        if min_ramp is None:
            raise RuntimeError("Grammar cannot compute a valid minimum ramp depth.")

        depths = list(range(min_ramp, self.init_max_depth + 1))
        if not depths:
            raise RuntimeError("No valid PI-Grow ramp depths available.")

        depth = depths[self._index % len(depths)]
        self._index += 1
        return depth


class PTC2Initialiser(GETreeInitialiser):
    def __init__(
        self,
        target_size: int,
        random_state: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)

        if target_size <= 0:
            raise ValueError("target_size must be > 0")

        self.target_size = target_size
        self.tree_generator: TreeGenerator | None = None
        self.max_depth = max_depth

    @classmethod
    def from_config(
        cls,
        config: FinchConfig,
        max_depth: Optional[int] = None,
    ) -> "PTC2Initialiser":
        try:
            random_state = config.experiment[Keys.RANDOM_SEED]
            target_size = config.ge[Keys.PTC2_TARGET_SIZE]
        except KeyError:
            raise ConfigError("Missing required config key 'ptc2_target_size'")

        return cls(
            target_size=target_size, random_state=random_state, max_depth=max_depth
        )

    def set_tree_generator(self, tree_generator: TreeGenerator) -> None:
        self.tree_generator = tree_generator

    def initialise(self) -> Individual:
        if self.tree_generator is None:
            raise RuntimeError("TreeGenerator not set")

        tree = self.tree_generator.generate_tree_ptc2(
            target_size=self.target_size, rng=self.rng, max_depth=self.max_depth
        )

        return Individual.from_tree(tree)


class RampedPTC2Initialiser(GETreeInitialiser):
    """
    Ramped size-based population initialiser using PTC2 family algorithms.

    This initialiser generates individuals using Probabilistic Tree Creation 2
    across a ramped distribution of target tree sizes.

    Behaviour:
        • If max_depth is None:
            Uses PTC2 (size-controlled only).

        • If max_depth is provided:
            Uses PTC2D (size + depth constrained).

    """

    def __init__(
        self,
        init_min_size: int,
        init_max_size: int,
        population_size: int,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)

        if init_min_size <= 1:
            raise ValueError("init_min_size must be > 1")

        if init_max_size < init_min_size:
            raise ValueError("init_max_size must be >= init_min_size")

        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be > 0 or None")

        self.init_min_size = init_min_size
        self.init_max_size = init_max_size
        self.population_size = population_size
        self.max_depth = max_depth

        self._index = 0
        self._schedule: list[int] | None = None

        self.tree_generator: TreeGenerator | None = None

    @classmethod
    def from_config(cls, config: FinchConfig) -> "RampedPTC2Initialiser":
        try:
            random_state = config.experiment[Keys.RANDOM_SEED]
            min_size = config.ge[Keys.INIT_TREE_MIN_SIZE]
            max_size = config.ge[Keys.INIT_TREE_MAX_SIZE]
            pop_size = config.ge[Keys.POPULATION_SIZE]
        except KeyError as e:
            raise ConfigError(
                f"Missing config key for RampedPTC2Initializer: {e.args[0]}"
            )

        max_depth = config.ge.get(Keys.INIT_MAX_DEPTH, None)

        return cls(
            init_min_size=min_size,
            init_max_size=max_size,
            population_size=pop_size,
            max_depth=max_depth,
            random_state=random_state,
        )

    def _build_schedule(self) -> None:
        """
        Build size ramp schedule.

        Distributes target tree sizes approximately uniformly across
        the population and shuffles ordering.
        """

        sizes = list(range(self.init_min_size, self.init_max_size + 1))
        bins = len(sizes)

        schedule: list[int] = []

        base = self.population_size // bins
        remainder = self.population_size % bins

        for i, size in enumerate(sizes):
            bin_size = base + (1 if i < remainder else 0)
            schedule.extend([size] * bin_size)

        self.rng.shuffle(schedule)
        self._schedule = schedule

    def _pick_size(self) -> int:
        if self._schedule is None:
            self._build_schedule()

        assert self._schedule
        if self._index >= len(self._schedule):
            self._index = 0

        assert self._schedule
        size = self._schedule[self._index]
        self._index += 1
        return size

    def set_tree_generator(self, tree_generator: TreeGenerator) -> None:
        self.tree_generator = tree_generator
        self.inject_tree_generator_rng()

    def initialise(self) -> Individual:
        if self.tree_generator is None:
            raise RuntimeError("TreeGenerator must be set before initialisation")

        target_size = self._pick_size()

        tree = self.tree_generator.generate_tree_ptc2(
            target_size=target_size,
            max_depth=self.max_depth,
            rng=self.rng,
        )

        return Individual.from_tree(tree)

    def inject_tree_generator_rng(self) -> None:
        if self.tree_generator is not None:
            if hasattr(self.tree_generator, "_rng"):
                self.tree_generator._rng = self._rng

            if hasattr(self.tree_generator, "_np_rng") and hasattr(self, "_np_rng"):
                self.tree_generator._np_rng = self._np_rng
