from typing import Any

from finchge.config import Keys
from finchge.parallel.base import BaseParallelBackend
from finchge.parallel.process_pool_backend import ProcessPoolBackend
from finchge.parallel.thread_pool_backend import ThreadPoolBackend


class ParallelBackendFactory:
    """Factory that creates parallel execution backends from configuration.

    Supports:
    - 'thread': Thread-based parallelism
    - 'process': Process-based parallelism
    """

    @staticmethod
    def create_backend(config: dict[str, Any]) -> BaseParallelBackend:
        """
        Create backend from config dict.

        Args:
            config: Dictionary with parallel configuration.
                   Must contain 'executor_type' key ('thread' or 'process').
                   Optional 'max_workers' and 'batch_size'.

        Returns:
            BaseParallelBackend instance
        """
        executor_type = config.get(Keys.EXECUTOR_TYPE, "process").lower()

        if executor_type == "thread":
            return ThreadPoolBackend(config)
        elif executor_type == "process":
            return ProcessPoolBackend(config)
        else:
            raise ValueError(
                f"Unknown executor_type: {executor_type}. "
                "Must be one of: 'thread' or 'process'"
            )
