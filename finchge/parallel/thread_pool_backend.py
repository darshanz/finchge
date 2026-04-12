import concurrent.futures
import logging
import random
import threading
from typing import Any, Dict, Optional

import numpy as np

from finchge.config import Keys
from finchge.fitness.fitness_functions import GEFitnessFunction
from finchge.fitness.fitness_types import (
    EvaluationRecord,
    Fitness,
    merge_fitness_results,
)
from finchge.parallel.base import BaseParallelBackend


def seed_everything(seed: int, use_torch: bool = False) -> None:
    """
    Set seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    if use_torch:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ThreadPoolBackend(BaseParallelBackend):
    """
    Thread-based parallel backend.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.max_workers: Optional[int] = config.get(Keys.MAX_WORKERS, None)
        self.batch_size: int = config.get(Keys.BATCH_SIZE, 10)
        self.executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._local: threading.local = threading.local()

    def _evaluate_single(
        self,
        runner: Any,
        phenotype: str,
        fitness_functions: list[GEFitnessFunction],
        required_keys: Dict[str, Any],
        seed: int,
    ) -> EvaluationRecord:
        """
        Evaluate a single individual in a thread.
        """
        try:
            # Set seed for reproducibility
            use_torch = False
            if runner is not None:
                use_torch = hasattr(runner, "train_dataset")

            seed_everything(seed, use_torch=use_torch)

            eval_context: dict[str, Any] = {"phenotype": phenotype}

            if runner is not None:
                # Run the phenotype
                runner_context = runner.run(
                    phenotype=phenotype, context_hints=required_keys
                )

                eval_context.update(runner_context)

                if hasattr(runner, "get_context"):
                    extra_context = runner.get_context()
                    if isinstance(extra_context, dict):
                        eval_context.update(extra_context)

            results = [fn.evaluate(eval_context) for fn in fitness_functions]

            return merge_fitness_results(results)

        except Exception as e:
            logging.error(f"Thread worker failed: {e}")
            import traceback

            traceback.print_exc()

            fallback = []
            for fn in fitness_functions:
                bad = float("-inf") if fn.maximize else float("inf")
                fallback.append(Fitness(value=bad))

            return merge_fitness_results(fallback)

    async def evaluate_batch(
        self,
        contexts: list[dict[str, Any]],
        fitness_functions: list[GEFitnessFunction],
    ) -> list[EvaluationRecord]:
        """
        Evaluate a batch of individuals using thread pool.

        Args:
            contexts: list of context dicts with 'runner', 'phenotype', 'seed'
            fitness_functions: list of fitness functions

        Returns:
            list of fitness values for each individual
        """
        if self.executor is None:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="FinchGE_Thread"
            )

        all_results: list[EvaluationRecord] = []
        total_items = len(contexts)

        # Process in batches
        for batch_start in range(0, total_items, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_items)
            batch_futures: list[concurrent.futures.Future[EvaluationRecord]] = []
            # Submit batch to thread pool
            for i in range(batch_start, batch_end):
                ctx = contexts[i]
                phenotype = ctx.get("phenotype", "")
                if phenotype is None:
                    raise ValueError("ThreadPoolBackend received None phenotype")

                future = self.executor.submit(
                    self._evaluate_single,
                    ctx.get("runner"),
                    phenotype,
                    fitness_functions,
                    ctx.get("required_keys", {}),
                    ctx.get("seed", 0),
                )
                batch_futures.append(future)

            # Collect results with timeout
            for future in batch_futures:
                try:
                    result = future.result(timeout=60)
                    # Ensure result is EvaluationRecord

                    if not isinstance(result, EvaluationRecord):
                        fallback = [
                            Fitness(
                                value=float("-inf") if fn.maximize else float("inf")
                            )
                            for fn in fitness_functions
                        ]
                        result = merge_fitness_results(fallback)
                except concurrent.futures.TimeoutError:
                    logging.warning("Thread task timed out")
                    fallback = [
                        Fitness(value=float("-inf") if fn.maximize else float("inf"))
                        for fn in fitness_functions
                    ]
                    result = merge_fitness_results(fallback)
                except Exception as e:
                    logging.error(f"Thread worker failed: {e}")
                    fallback = [
                        Fitness(value=float("-inf") if fn.maximize else float("inf"))
                        for fn in fitness_functions
                    ]
                    result = merge_fitness_results(fallback)

                all_results.append(result)

        return all_results

    async def shutdown(self) -> None:
        """Shutdown the thread pool gracefully."""
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None
