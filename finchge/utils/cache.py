import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Generic, Optional, TypeVar, cast

import diskcache as dc

from finchge.config import FinchConfig
from finchge.config.config import Keys

K = TypeVar("K")
V = TypeVar("V")


class CacheInterface(ABC, Generic[K, V]):
    @abstractmethod
    def get(self, key: K) -> Optional[V]: ...

    @abstractmethod
    def set(self, key: K, value: V) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


class LRUCache(CacheInterface[K, V]):
    def __init__(self, cache_size: int = 128) -> None:
        self.cache: OrderedDict[K, V] = OrderedDict()
        self.cache_size = cache_size

    def get(self, key: K) -> Optional[V]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: K, value: V) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()


class DiskCache(CacheInterface[K, V]):
    def __init__(self, cache_dir: str = "cache", size_limit: int = 2**30) -> None:
        self.cache: Any = dc.Cache(cache_dir, size_limit=size_limit)

    def get(self, key: K) -> Optional[V]:
        return cast(Optional[V], self.cache.get(key))

    def set(self, key: K, value: V) -> None:
        self.cache.set(key, value)

    def clear(self) -> None:
        self.cache.clear()


class CacheManager(Generic[V]):
    """
    Central cache manager.

    Responsibilities:
    - key construction (hashing, namespacing)
    - backend selection (LRU / disk )
    - fitness cache access

    """

    def __init__(
        self,
        *,
        cache_type: str = "lru",
        experiment_id: Optional[str] = None,
        evaluator_namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if cache_type == "lru":
            self.cache: CacheInterface[str, V] = LRUCache(**kwargs)
        elif cache_type == "disk":
            self.cache = DiskCache(**kwargs)
        else:
            raise ValueError(f"Unsupported cache type: {cache_type}")

        # Namespacing (optional but recommended)
        self.experiment_id = experiment_id or "default"
        self.evaluator_namespace = evaluator_namespace or "fitness"

    @classmethod
    def from_config(
        cls,
        cfg: Optional[FinchConfig],
        *,
        experiment_id: Optional[str] = None,
        evaluator_namespace: Optional[str] = None,
    ) -> "CacheManager[V]":
        cache_type = "lru" if not cfg else cfg.experiment.get(Keys.CACHE_TYPE, "lru")
        cache_size = 128 if not cfg else cfg.experiment.get(Keys.CACHE_SIZE, 128)

        return cls(
            cache_type=cache_type,
            cache_size=cache_size,
            experiment_id=experiment_id,
            evaluator_namespace=evaluator_namespace,
        )

    def get_fitness(
        self,
        *,
        phenotype: str,
        env_version: str | int,
    ) -> Optional[V]:
        key = self._make_fitness_key(phenotype, env_version)
        return self.cache.get(key)

    def set_fitness(
        self,
        *,
        phenotype: str,
        env_version: str | int,
        fitness: V,
    ) -> None:
        key = self._make_fitness_key(phenotype, env_version)
        self.cache.set(key, fitness)

    def _make_fitness_key(
        self,
        phenotype: str,
        env_version: str | int,
    ) -> str:
        """
        Build a deterministic, parallel-safe cache key.
        """
        phenotype_hash = hashlib.sha256(phenotype.encode("utf-8")).hexdigest()

        return (
            f"fitness::"
            f"{self.experiment_id}::"
            f"{self.evaluator_namespace}::"
            f"{env_version}::"
            f"{phenotype_hash}"
        )

    def clear(self) -> None:
        self.cache.clear()
