show_source: false# API Reference


The finchGE API consists of:


::: finchge.core.Population
    options:
      show_root_heading: true
      show_source: false

::: finchge.core.Individual
    options:
      show_root_heading: true
      show_source: false
      filters:
        - "!^__.*__$"
        - "!^_.*"

::: finchge.grammar.Grammar
    options:
      show_root_heading: true
      show_source: false
      filters:
          - "!^__.*__$"
          - "!^_.*"

::: finchge.grammar.parser
    options:
      show_root_heading: true
      show_source: false
      members_order: source

::: finchge.grammar.rule
    options:
      show_root_heading: true
      show_source: false
      filters:
          - "!^__.*__$"
          - "!^_.*"


::: finchge.grammar.GenotypeMapper
    options:
      show_root_heading: true
      show_source: false
      filters:
          - "!^__.*__$"
          - "!^_.*"

::: finchge.grammar.derivation_tree.TreeNode
    options:
      show_root_heading: true
      show_source: false
      filters:
          - "!^__.*__$"
          - "!^_.*"

::: finchge.algorithm
    options:
      show_root_heading: true
      show_source: false
      members:
        - BaseAlgorithm
        - GeneticAlgorithm
        - NSGA2
        - NSGA3


::: finchge.initialisation
    options:
      show_root_heading: true
      show_source: false
      members:
        - RandomGenomeInitialiser
        - RVDInitialiser
        - FullTreeInitialiser
        - GrowTreeInitialiser
        - PIGrowInitialiser
        - RHHInitialiser
        - PTC2Initialiser
        - RampedPTC2Initialiser

::: finchge.operators.selection
    options:
      show_root_heading: true
      show_source: false
      members:
        - GESelectionStrategy
        - TournamentSelection
        - RouletteWheelSelection
        - RankSelection
        - TruncationSelection
        - NSGA2TournamentSelection
        - NSGA3TournamentSelection

::: finchge.operators.crossover
    options:
      show_root_heading: true
      show_source: false

::: finchge.operators.mutation
    options:
      show_root_heading: true
      show_source: false

::: finchge.operators.replacement
    options:
      show_root_heading: true
      show_source: false
      filters:
          - "!^__.*__$"
          - "!^_.*"

::: finchge.config
    options:
      show_root_heading: true
      show_source: false
      members:
        - FinchConfig
        - ConfigValidator
      filters:
        - "!^__.*__$"
        - "!^_.*"


::: finchge.fitness
    options:
      show_root_heading: true
      show_source: false
      filters:
        - "!.*FitnessEvaluator$"
      members:
        - GEFitnessFunction
        - AccuracyFitness

::: finchge.grammar.repair_strategy
    options:
      show_root_heading: true
      show_source: false


 
::: finchge.core.GrammaticalEvolution
    options:
      show_root_heading: true
      show_source: false

::: finchge.fitness.FitnessEvaluator
    options:
      show_root_heading: true
      show_source: false

::: finchge.utils.cache
    options:
      show_root_heading: true
      show_source: false


::: finchge.utils.logger
    options:
      show_root_heading: true
      show_source: false
      members:
        - BaseLogger
        - FileLogger
        - ExperimentLogger
        - setup_logging
        - get_log_dir

::: finchge.utils.checkpoint
    options:
      show_root_heading: true
      show_source: false

::: finchge.symbolic
    options:
      show_root_heading: true
      show_source: false


::: finchge.grammar.range_handlers
    options:
      show_root_heading: true
      show_source: false
