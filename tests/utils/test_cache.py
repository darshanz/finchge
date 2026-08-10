import pytest

from finchge.utils.cache import CacheManager, LRUCache


def test_lru_cache_get_set_roundtrip():
    cache: LRUCache[str, float] = LRUCache(cache_size=10)
    cache.set("a", 1.0)
    assert cache.get("a") == 1.0


def test_lru_cache_miss_returns_none():
    cache: LRUCache[str, float] = LRUCache(cache_size=10)
    assert cache.get("missing") is None


def test_lru_cache_evicts_lru_when_full():
    cache: LRUCache[str, int] = LRUCache(cache_size=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    # access "a" to make it recently used
    cache.get("a")
    # adding "d" should evict "b" (LRU)
    cache.set("d", 4)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4


def test_lru_cache_capacity_never_exceeded():
    cache: LRUCache[int, int] = LRUCache(cache_size=5)
    for i in range(20):
        cache.set(i, i)
    assert len(cache.cache) <= 5


def test_lru_cache_clear_empties():
    cache: LRUCache[str, int] = LRUCache(cache_size=5)
    cache.set("x", 10)
    cache.clear()
    assert cache.get("x") is None


# CacheManager


def test_cache_manager_set_and_get_fitness():
    mgr: CacheManager[list[float]] = CacheManager(cache_type="lru", cache_size=32)
    mgr.set_fitness(phenotype="x + 1", env_version=1, fitness=[0.5])
    result = mgr.get_fitness(phenotype="x + 1", env_version=1)
    assert result == [0.5]


def test_cache_manager_miss_returns_none():
    mgr: CacheManager[list[float]] = CacheManager(cache_type="lru", cache_size=32)
    assert mgr.get_fitness(phenotype="never_set", env_version=1) is None


def test_cache_manager_env_version_isolates_entries():
    mgr: CacheManager[list[float]] = CacheManager(cache_type="lru", cache_size=32)
    mgr.set_fitness(phenotype="x", env_version=1, fitness=[1.0])
    mgr.set_fitness(phenotype="x", env_version=2, fitness=[2.0])
    assert mgr.get_fitness(phenotype="x", env_version=1) == [1.0]
    assert mgr.get_fitness(phenotype="x", env_version=2) == [2.0]


def test_cache_manager_clear():
    mgr: CacheManager[list[float]] = CacheManager(cache_type="lru", cache_size=32)
    mgr.set_fitness(phenotype="x", env_version=1, fitness=[3.0])
    mgr.clear()
    assert mgr.get_fitness(phenotype="x", env_version=1) is None


def test_cache_manager_unknown_type_raises():
    with pytest.raises(ValueError, match="Unsupported cache type"):
        CacheManager(cache_type="redis")


def test_cache_manager_same_phenotype_different_namespace():
    mgr1: CacheManager[list[float]] = CacheManager(
        cache_type="lru", evaluator_namespace="fitness_a"
    )
    mgr2: CacheManager[list[float]] = CacheManager(
        cache_type="lru", evaluator_namespace="fitness_b"
    )
    mgr1.set_fitness(phenotype="x", env_version=1, fitness=[1.0])
    # mgr2 has a different namespace so it should not see mgr1's entry
    assert mgr2.get_fitness(phenotype="x", env_version=1) is None
