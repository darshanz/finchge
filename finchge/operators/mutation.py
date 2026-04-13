from typing import Literal, Optional

from finchge.core.individual import Individual
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.tree_generator import TreeGenerator
from finchge.operators.base import GEMutationStrategy


class IntFlipMutation(GEMutationStrategy):
    """
    Integer flip mutation strategy.

    Args:
        mutation_probability (float): Probability of mutation per gene.
        codon_size (int): Maximum codon value (inclusive).
    """

    def __init__(
        self,
        mutation_probability: float,
        codon_size: int,
        mode: Literal["per_codon", "per_ind"] = "per_codon",
        mutation_events: int = 1,
        within_used: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        self.mutation_probability = mutation_probability
        self.codon_size = codon_size
        self.mode = mode
        self.mutation_events = mutation_events
        self.within_used = within_used

        if self.mode not in {"per_codon", "per_ind"}:
            raise ValueError(
                f"Unsupported mutation mode: {self.mode!r}. "
                "Expected 'per_codon' or 'per_ind'."
            )

        if self.mode == "per_codon":
            if self.mutation_probability is not None:
                if not (0.0 <= self.mutation_probability <= 1.0):
                    raise ValueError("mutation_probability must be between 0 and 1.")

        if self.mode == "per_ind":
            if self.mutation_events < 0:
                raise ValueError("mutation_events must be >= 0.")

    def mutate(self, individual: "Individual") -> "Individual":
        """
        Perform integer flip mutation on an Individual.

        Args:
            individual (Individual): Individual to apply mutation on.

        Returns:
            Individual: Mutated Individual.
        """
        if not individual.genotype:
            raise ValueError(
                "IntFlipMutation requires individuals with a genotype, but none was found.\n"
                "This usually happens when using tree-based initialization, which does not create genotypes.\n"
                "To use this operator, enable genotype encoding by setting encode_trees=True in FitnessEvaluator class."
            )

        # avoiding mutating the original individual in place.
        genome: list[int] = individual.genotype.copy()

        # calculate effective length if it is within_used
        eff_length = self._get_effective_length(individual=individual, genome=genome)

        if eff_length == 0:
            return Individual.from_genotype(genome)

        if self.mode == "per_codon":
            self._mutate_per_codon(genome, eff_length)
        elif self.mode == "per_ind":
            self._mutate_per_ind(genome, eff_length)

        return Individual.from_genotype(genome)

    def _get_effective_length(
        self,
        individual: "Individual",
        genome: list[int],
    ) -> int:
        """
        Determine how much of the genome is eligible for mutation.
        """
        if not self.within_used:
            return len(genome)
        if individual.invalid:
            return len(genome)
        effective_length = (
            len(genome) if not individual.used_genome else len(individual.used_genome)
        )
        return max(0, min(effective_length, len(genome)))

    def _mutate_per_codon(self, genome: list[int], mutation_length: int) -> None:
        """
        Mutate each gene independently with some probability.
        """
        if self.mutation_probability is not None:
            p_mut = self.mutation_probability
        else:
            p_mut = 1.0 / mutation_length

        for i in range(mutation_length):
            if self.rng.random() < p_mut:
                genome[i] = self.rng.randint(0, self.codon_size)

    def _mutate_per_ind(self, genome: list[int], mutation_length: int) -> None:
        """
        Perform a fixed number of mutation events.
        """
        for _ in range(self.mutation_events):
            idx = self.rng.randint(0, mutation_length - 1)
            genome[idx] = self.rng.randint(0, self.codon_size)


class SwapMutation(GEMutationStrategy):
    """
    Swap mutation strategy.
    Randomly swaps pairs of genes in the genotype based on
    a per gene mutation probability.
    """

    def __init__(
        self, mutation_probability: float, random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the swap mutation strategy.

        Args:
            mutation_probability: Probability in [0.0, 1.0] that a given
                gene position is selected for swapping.

        Raises:
            ValueError: If mutation_probability is not in [0.0, 1.0].
        """
        super().__init__(random_state=random_state)

        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be in [0, 1]")

        self.mutation_probability = mutation_probability

    def mutate(self, individual: "Individual") -> "Individual":
        """
        Apply swap mutation to an individual.

        Each gene position is independently selected with
        `mutation_probability`. Selected positions are randomly
        paired and their values swapped.

        Args:
            individual: Individual to mutate.

        Returns:
            A new Individual with a mutated genotype.
        """
        if not individual.genotype:
            raise ValueError(
                "SwapMutation requires individuals with a genotype, but none was found.\n"
                "This usually happens when using tree-based initialization, which does not create genotypes.\n"
                "To use this operator, enable genotype encoding by setting encode_trees=True in FitnessEvaluator class."
            )
        original_genome: list[int] = individual.genotype

        # Copy genome to avoid mutating the original individual
        mutated_genome: list[int] = original_genome.copy()

        # Collect indices selected for mutation
        swap_positions: list[int] = [
            i
            for i in range(len(mutated_genome))
            if self.rng.random() < self.mutation_probability
        ]

        # Randomize pairing order
        self.rng.shuffle(swap_positions)

        # Swap values in pairs
        for i in range(0, len(swap_positions) - 1, 2):
            p1 = swap_positions[i]
            p2 = swap_positions[i + 1]

            mutated_genome[p1], mutated_genome[p2] = (
                mutated_genome[p2],
                mutated_genome[p1],
            )

        return Individual.from_genotype(mutated_genome)


class GaussianMutation(GEMutationStrategy):
    """
    Gaussian noise mutation strategy.

    Adds Gaussian noise to selected genes and clamps results
    to non-negative integers.
    """

    def __init__(
        self,
        mutation_probability: float,
        std_dev: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the Gaussian mutation strategy.

        Args:
            mutation_probability: Probability in [0.0, 1.0] that a given
                gene is mutated.
            std_dev: Standard deviation of the Gaussian noise.

        Raises:
            ValueError: If mutation_probability is not in [0.0, 1.0].
        """
        super().__init__(random_state=random_state)

        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be in [0, 1]")

        self.mutation_probability = mutation_probability
        self.std_dev = std_dev

    def mutate(self, individual: "Individual") -> "Individual":
        """
        Apply Gaussian mutation to an individual.

        Each gene is independently selected with
        `mutation_probability`. Selected genes receive
        Gaussian noise, are rounded to integers, and
        clamped to be non-negative.

        Args:
            individual: Individual to mutate.

        Returns:
            A new Individual with a mutated genotype.
        """
        if not individual.genotype:
            raise ValueError(
                "GaussianMutation requires individuals with a genotype, but none was found.\n"
                "This usually happens when using tree-based initialization, which does not create genotypes.\n"
                "To use this operator, enable genotype encoding by setting encode_trees=True in FitnessEvaluator class."
            )
        original_genome: list[int] = individual.genotype

        # Copy genome to avoid mutating the original individual
        mutated_genome: list[int] = original_genome.copy()

        for i, value in enumerate(mutated_genome):
            if self.rng.random() < self.mutation_probability:
                # Add Gaussian noise, round to nearest int, clamp to >= 0
                noisy_value = int(round(value + self.rng.gauss(0.0, self.std_dev)))
                mutated_genome[i] = max(0, noisy_value)

        return Individual.from_genotype(mutated_genome)


class InversionMutation(GEMutationStrategy):
    """Inversion mutation strategy.

    With a given probability, selects a random contiguous segment
    of the genome and reverses its order.
    """

    def __init__(
        self, segment_probability: float, random_state: Optional[int] = None
    ) -> None:
        """Initialize the inversion mutation strategy.

        Args:
            segment_probability: Probability in [0.0, 1.0] of applying
                the inversion mutation.

        Raises:
            ValueError: If segment_probability is not in [0.0, 1.0].
        """
        super().__init__(random_state=random_state)

        if not 0.0 <= segment_probability <= 1.0:
            raise ValueError("segment_probability must be in [0, 1]")

        self.segment_probability = segment_probability

    def mutate(self, individual: "Individual") -> "Individual":
        """Apply inversion mutation to an individual.

        With probability `segment_probability`, a random contiguous
        segment [start:end) of the genome is reversed.

        Args:
            individual: Individual to mutate.

        Returns:
            A new Individual with a mutated genotype.
        """
        if not individual.genotype:
            raise ValueError(
                "InversionMutation requires individuals with a genotype, but none was found.\n"
                "This usually happens when using tree-based initialization, which does not create genotypes.\n"
                "To use this operator, enable genotype encoding by setting encode_trees=True in FitnessEvaluator class."
            )
        original_genome: list[int] = individual.genotype
        length = len(original_genome)

        # Decide whether to mutate
        if self.rng.random() >= self.segment_probability:
            return Individual.from_genotype(original_genome.copy())

        start = self.rng.randrange(0, length - 1)
        end = self.rng.randrange(start + 1, length)

        # Copy genome to avoid mutating the original individual
        mutated_genome: list[int] = original_genome.copy()

        # Reverse the selected segment
        mutated_genome[start:end] = reversed(mutated_genome[start:end])

        return Individual.from_genotype(mutated_genome)


class CyclicMutation(GEMutationStrategy):
    """
    Cyclic mutation strategy.

    Rotates fixed-size contiguous segments of the genome
    with a per-gene mutation probability.
    """

    def __init__(
        self,
        mutation_probability: float,
        segment_size: int = 3,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize the cyclic mutation strategy.

        Args:
            mutation_probability: Probability in [0.0, 1.0] that a given
                position triggers a cyclic rotation.
            segment_size: Size of the segment to rotate.

        Raises:
            ValueError: If mutation_probability is not in [0.0, 1.0].
            ValueError: If segment_size is less than 2.
        """
        super().__init__(random_state=random_state)

        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be in [0, 1]")

        if segment_size < 2:
            raise ValueError("segment_size must be >= 2")

        self.mutation_probability = mutation_probability
        self.segment_size = segment_size

    def mutate(self, individual: "Individual") -> "Individual":
        """
        Apply cyclic mutation to an individual.

        Each genome position is independently selected with
        `mutation_probability`. When selected, the contiguous
        segment starting at that position is rotated by one step.

        Args:
            individual: Individual to mutate.

        Returns:
            A new Individual with a mutated genotype.
        """
        if not individual.genotype:
            raise ValueError(
                "CyclicMutation requires individuals with a genotype, but none was found.\n"
                "This usually happens when using tree-based initialization, which does not create genotypes.\n"
                "To use this operator, enable genotype encoding by setting encode_trees=True in FitnessEvaluator class."
            )
        original_genome: list[int] = individual.genotype
        genome_length = len(original_genome)

        # Copy genome to avoid mutating the original individual
        mutated_genome: list[int] = original_genome.copy()

        for pos in range(genome_length):
            if self.rng.random() < self.mutation_probability:
                end = pos + self.segment_size
                if end <= genome_length:
                    # Rotate segment right by one position
                    segment = mutated_genome[pos:end]
                    mutated_genome[pos:end] = segment[-1:] + segment[:-1]

        return Individual.from_genotype(mutated_genome)


class DuplicationMutation(GEMutationStrategy):
    """Duplication mutation strategy.

    With a given probability, duplicates a contiguous segment of fixed size
    and copies it into another non-overlapping location in the genome.
    """

    def __init__(
        self,
        mutation_probability: float,
        segment_size: int = 2,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize the duplication mutation strategy.

        Args:
            mutation_probability: Probability in [0.0, 1.0] of applying
                the duplication mutation.
            segment_size: Length of the segment to duplicate, must be > 0.

        Raises:
            ValueError: If mutation_probability is not in [0.0, 1.0].
            ValueError: If segment_size is not a positive integer.
        """
        super().__init__(random_state=random_state)

        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be in [0, 1]")

        if segment_size <= 0:
            raise ValueError("segment_size must be a positive integer")

        self.mutation_probability = mutation_probability
        self.segment_size = segment_size

    def mutate(self, individual: "Individual") -> "Individual":
        """Apply duplication mutation to an individual.

        A segment of length `segment_size` is copied from one position and
        overwrites another non-overlapping segment with probability
        `mutation_probability`.

        Args:
            individual: Individual to mutate.

        Returns:
            A new Individual with a mutated genotype.
        """
        if not individual.genotype:
            raise ValueError(
                "DuplicationMutation requires individuals with a genotype, but none was found.\n"
                "This usually happens when using tree-based initialization, which does not create genotypes.\n"
                "To use this operator, enable genotype encoding by setting encode_trees=True in FitnessEvaluator class."
            )
        original_genome: list[int] = individual.genotype
        length = len(original_genome)

        # Copy genome to avoid mutating the original individual
        mutated_genome: list[int] = original_genome.copy()

        # Decide whether to mutate
        if self.rng.random() >= self.mutation_probability:
            return Individual.from_genotype(mutated_genome)

        # Ensure space for a non-overlapping source and target segment
        if length < 2 * self.segment_size:
            return Individual.from_genotype(mutated_genome)

        source_start = self.rng.randrange(0, length - self.segment_size + 1)
        source_end = source_start + self.segment_size

        # Compute valid non-overlapping target start indices
        possible_targets: list[int] = list(
            range(0, max(0, source_start - self.segment_size + 1))
        ) + list(range(source_end, length - self.segment_size + 1))

        if not possible_targets:
            return Individual.from_genotype(mutated_genome)

        target_start = self.rng.choice(possible_targets)
        target_end = target_start + self.segment_size

        # Duplicate the source segment into the target location
        segment = original_genome[source_start:source_end]
        mutated_genome[target_start:target_end] = segment

        return Individual.from_genotype(mutated_genome)


class MultipleMutation(GEMutationStrategy):
    """
    Multiple mutation strategies combiner.

    Randomly selects and applies one mutation strategy
    according to provided probabilities.
    """

    def __init__(
        self,
        strategies: list[GEMutationStrategy],
        probabilities: Optional[list[float]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the multiple mutation strategy.

        Args:
            strategies: List of mutation strategies to choose from.
            probabilities: Optional selection probabilities for each strategy.
                If None, all strategies are selected uniformly.

        Raises:
            ValueError: If probabilities length does not match strategies
                or if probabilities sum to zero.
        """
        super().__init__(random_state=random_state)

        if not strategies:
            raise ValueError("strategies must be a non-empty list")

        self.strategies = strategies

        if probabilities is None:
            # Uniform selection if no probabilities are provided
            self.probabilities: list[float] = [1.0 / len(strategies)] * len(strategies)
        else:
            if len(probabilities) != len(strategies):
                raise ValueError(
                    "probabilities must have the same length as strategies"
                )

            total = sum(probabilities)
            if total <= 0.0:
                raise ValueError("sum of probabilities must be > 0")

            # Normalize probabilities
            self.probabilities = [p / total for p in probabilities]

    def mutate(self, individual: "Individual") -> "Individual":
        """Apply a randomly selected mutation strategy.

        One mutation strategy is selected according to
        the configured probabilities and applied to
        the individual.

        Args:
            individual: Individual to mutate.

        Returns:
            A mutated Individual.
        """
        selected_strategy = self.rng.choices(
            self.strategies,
            weights=self.probabilities,
            k=1,
        )[0]

        return selected_strategy.mutate(individual)


class SubtreeMutation(GEMutationStrategy):
    """
    Subtree mutation replaces randomly selected subtrees within a program tree
    with newly generated random subtrees.
    Mutation targets nodes representing non-terminal grammar symbols. The
    subtree rooted at the selected node is replaced with a newly generated
    subtree that satisfies grammar constraints. Subtree mutation is a core mutation
    mechanism in genetic programming and was introduced alongside subtree crossover in early GP research.

    Characteristics:

    - Maintains syntactic validity of individuals.
    - Introduces structural novelty.
    - Controls search-space exploration via subtree depth limits.
    """

    def __init__(
        self,
        mutation_probability: float,
        non_terminals: list[str],
        tree_generator: TreeGenerator,
        mutation_max_depth: int,
        mutation_events: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        self.mutation_probability = mutation_probability
        self.non_terminals = non_terminals
        self.tree_generator = tree_generator
        self.mutation_events = mutation_events
        self.mutation_max_depth = mutation_max_depth

    def mutate(self, individual: "Individual") -> "Individual":
        """
        Mutate an individual using subtree mutation.

        Args:
            individual: Parent individual.

        Returns:
            A newly constructed mutated individual.
        """
        if not individual.tree:
            raise ValueError("The tree must exist.")
        tree: TreeNode = TreeNode.from_string(individual.tree)
        # Decide whether mutation happens at all
        if self.rng.random() > self.mutation_probability:
            return Individual.from_tree(tree)

        for _ in range(self.mutation_events):
            # Collect grammar-valid mutation points
            symbols = sorted(tree.collect_symbols(self.non_terminals))
            if not symbols:
                break  # nothing to mutate

            symbol = self.rng.choice(symbols)
            target: TreeNode = self.rng.choice(tree.find_by_symbol(symbol))

            # Generate a fresh subtree rooted at the same symbol
            new_subtree = self.tree_generator.generate_subtree(
                symbol=symbol, max_depth=self.mutation_max_depth, rng=self.rng
            )

            # Replace target subtree
            tree = target.replace_subtree_with(new_subtree)

        return Individual.from_tree(tree)
