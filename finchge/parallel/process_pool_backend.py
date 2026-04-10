import asyncio
import concurrent
import logging
import os
import random
from typing import Any, Optional

import numpy as np
from cloudpickle import cloudpickle

from finchge.config import Keys
from finchge.fitness.fitness_functions import GEFitnessFunction
from finchge.fitness.fitness_types import (
    EvaluationRecord,
    Fitness,
    merge_fitness_results,
)
from finchge.parallel.base import BaseParallelBackend


def seed_everything(seed: int, use_torch: bool = False) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    if use_torch:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For determinism (optional; can slow down):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def process_train_and_evaluate(
    context: dict[str, Any], fitness_functions: list[GEFitnessFunction]
) -> EvaluationRecord:
    """Process a single individual in a process pool."""
    try:
        phenotype = context.get("phenotype", "")
        seed = context.get("seed", 0)

        # Set seed for reproducibility
        use_torch = False
        if "runner" in context:
            runner = context["runner"]
            use_torch = hasattr(runner, "train_dataset")

        seed_everything(seed, use_torch=use_torch)

        eval_context: dict[str, Any] = {}

        if "runner" in context:
            # Classic case: use runner
            runner = context["runner"]
            eval_context = runner.run(
                phenotype=phenotype, context_hints=context.get("required_keys", {})
            )
            # Add any additional context the executor might provide
            if hasattr(runner, "get_context"):
                extra_context = runner.get_context()
                if isinstance(extra_context, dict):
                    eval_context.update(extra_context)
        else:
            # Direct evaluation: phenotype is the value, Fitness function should handle that. eg. stringmatch
            eval_context = {
                "phenotype": phenotype,
            }

        # Calculate fitness
        results = [
            fitness_fn.evaluate(eval_context) for fitness_fn in fitness_functions
        ]
        return merge_fitness_results(results)

    except Exception as e:
        logging.error(f"Exception in worker: {e}", exc_info=True)
        import traceback

        traceback.print_exc()
        # Return worst possible fitness
        fallback = [
            Fitness(value=float("-inf") if fn.maximize else float("inf"))
            for fn in fitness_functions
        ]
        return merge_fitness_results(fallback)


class ProcessPoolBackend(BaseParallelBackend):
    """Process-based parallel backend using cloudpickle for serialization."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.max_workers: int = config.get(Keys.MAX_WORKERS) or (os.cpu_count() or 1)
        self.batch_size: int = config.get(Keys.BATCH_SIZE, 10)
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None

    async def evaluate_batch(
        self,
        contexts_: list[dict[str, Any]],
        fitness_functions: list[GEFitnessFunction],
    ) -> list[EvaluationRecord]:
        """Evaluate a batch of individuals using process pool."""
        contexts: list[dict[str, Any]] = cloudpickle.loads(contexts_)
        fitness_funcs: list[GEFitnessFunction] = cloudpickle.loads(fitness_functions)

        if self.executor is None:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            )

        all_results: list[EvaluationRecord] = []

        # Process in batches
        for batch_start in range(0, len(contexts), self.batch_size):
            batch = contexts[batch_start : batch_start + self.batch_size]

            futures: list[Any] = []

            # ONE task per individual
            for context in batch:
                future = self.executor.submit(
                    process_train_and_evaluate,
                    context,
                    fitness_funcs,
                )
                futures.append(future)

            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=60)
                    # Ensure result is a list of EvaluationRecords
                    if not isinstance(result, EvaluationRecord):
                        fallback = [
                            Fitness(
                                value=float("-inf") if fn.maximize else float("inf")
                            )
                            for fn in fitness_funcs
                        ]
                        result = merge_fitness_results(fallback)
                except concurrent.futures.TimeoutError:
                    logging.warning("Process task timed out")
                    fallback = [
                        Fitness(value=float("-inf") if fn.maximize else float("inf"))
                        for fn in fitness_funcs
                    ]
                    result = merge_fitness_results(fallback)
                except Exception as e:
                    logging.error(f"Worker failed: {e}")
                    fallback = [
                        Fitness(value=float("-inf") if fn.maximize else float("inf"))
                        for fn in fitness_funcs
                    ]
                    result = merge_fitness_results(fallback)

                all_results.append(result)

        return all_results

    async def shutdown(self) -> None:
        """Shutdown the process pool gracefully."""
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.executor.shutdown, True)
            self.executor = None
