from __future__ import annotations

import hashlib
import json
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path

# to avoid circular imports while keeping mypy working for type checking
from typing import TYPE_CHECKING, Any, Optional, Protocol

import numpy as np

from finchge.config import FinchConfig

if TYPE_CHECKING:
    from finchge.algorithm.base import BaseAlgorithm
    from finchge.core.population import Population


FUNCHGE_CHECKPOINT_VERSION = "1"


def stable_config_hash(config: dict[str, Any] | FinchConfig) -> str:
    """
    Create a hash of config for checkpoint compatibility checks.
    """
    # hash after converting FinchConfig or dict to json.
    ge_config: str = ""
    if isinstance(config, FinchConfig):
        ge_config = config.to_json()
    elif isinstance(config, dict):
        ge_config = FinchConfig.from_dict(config).to_json()

    payload = json.dumps(ge_config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RNGState:
    """
    Serializable RNG states for deterministic resume.
    """

    py_state: tuple[Any, ...]
    np_state: dict[str, Any]


@dataclass
class CheckpointState:
    """
    All state required to resume a GE run deterministically.
    """

    version: str
    config_hash: str
    generation: int
    population: "Population"
    algorithm: "BaseAlgorithm"
    rng_state: RNGState


class CheckpointManager(Protocol):
    """
    Checkpoint manager interface.
    """

    def exists(self) -> bool:
        """
        Return True if at least one checkpoint exists.
        """
        ...

    def save(
        self,
        *,
        generation: int,
        population: "Population",
        algorithm: "BaseAlgorithm",
        config: dict[str, Any] | FinchConfig,
        py_rng_state: tuple[Any, ...],
        np_rng_state: dict[str, Any],
    ) -> Path:
        """Persist a checkpoint and return the path written."""
        ...

    def load_latest(
        self,
        *,
        expected_config_hash: Optional[str] = None,
    ) -> CheckpointState:
        """Load and return the latest checkpoint."""
        ...

    def should_save(self, generation: int) -> bool:
        """
        Return True if this generation should be checkpointed.
        Based on 'every' parameter's value, decide whether to save or not.
        """
        ...


def capture_rng_state() -> RNGState:
    return RNGState(
        py_state=random.getstate(),
        np_state=np.random.get_state(),
    )


def restore_rng_state(state: RNGState) -> None:
    random.setstate(state.py_state)
    np.random.set_state(state.np_state)


class FileCheckpointManager(CheckpointManager):
    """
    Filesystem checkpoint manager using pickle.
    Population and Algorithm must be pickle-serializable.
    RNG state is stored to guarantee deterministic resume.
    Files are written as: `checkpoint_gen_[GEN].pkl`

    """

    def __init__(
        self,
        directory: str | Path,
        *,
        every: int = 10,
        keep_last: int = 5,
        filename_prefix: str = "checkpoint_gen_",
    ) -> None:
        if every <= 0:
            raise ValueError("every must be >= 1")
        if keep_last <= 0:
            raise ValueError("keep_last must be >= 1")

        self.dir = Path(directory)
        self.every = every
        self.keep_last = keep_last
        self.filename_prefix = filename_prefix
        self.dir.mkdir(parents=True, exist_ok=True)

        # precompile pattern for fast scanning
        self._pattern = re.compile(rf"^{re.escape(self.filename_prefix)}(\d+)\.pkl$")

    def exists(self) -> bool:
        return self._latest_checkpoint_path() is not None

    def should_save(self, generation: int) -> bool:
        """
        Return True if this generation should be checkpointed.
        Based on 'every' parameter's value, decide whether to save or not.
        """
        return generation % self.every == 0

    def save(
        self,
        *,
        generation: int,
        population: "Population",
        algorithm: "BaseAlgorithm",
        config: dict[str, Any] | FinchConfig,
        py_rng_state: tuple[Any, ...],
        np_rng_state: dict[str, Any],
    ) -> Path:
        config_hash = stable_config_hash(config)
        state = CheckpointState(
            version=FUNCHGE_CHECKPOINT_VERSION,
            config_hash=config_hash,
            generation=generation,
            population=population,
            algorithm=algorithm,
            rng_state=RNGState(py_state=py_rng_state, np_state=np_rng_state),
        )

        path = self.dir / f"{self.filename_prefix}{generation}.pkl"
        tmp = path.with_suffix(".pkl.tmp")

        # Atomic-ish write: write temp then rename
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

        self._prune_old()
        return path

    def load_latest(
        self,
        *,
        expected_config_hash: Optional[str] = None,
    ) -> CheckpointState:
        """
        Load latest checkpoint and verify config hash.
        """
        path = self._latest_checkpoint_path()
        if path is None:
            raise FileNotFoundError(f"No checkpoints found in {self.dir}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        if not isinstance(state, CheckpointState):
            raise TypeError(f"Invalid checkpoint type in {path}")

        if state.version != FUNCHGE_CHECKPOINT_VERSION:
            raise RuntimeError(
                f"Checkpoint version mismatch: found {state.version}, "
                f"expected {FUNCHGE_CHECKPOINT_VERSION}"
            )

        if (
            expected_config_hash is not None
            and state.config_hash != expected_config_hash
        ):
            raise RuntimeError(
                "Checkpoint config hash does not match current config. "
                "Refusing to resume to prevent inconsistent runs."
            )

        return state

    def _latest_checkpoint_path(self) -> Path | None:
        """
        Return path to latest checkpoint by generation number.
        """
        best_gen = None
        best_path: Path | None = None

        for p in self.dir.iterdir():
            if not p.is_file():
                continue
            m = self._pattern.match(p.name)
            if not m:
                continue
            gen = int(m.group(1))
            if best_gen is None or gen > best_gen:
                best_gen = gen
                best_path = p

        return best_path

    def _prune_old(self) -> None:
        """
        Keep only the newest keep_last checkpoints.
        """
        checkpoints: list[tuple[int, Path]] = []

        for p in self.dir.iterdir():
            if not p.is_file():
                continue
            m = self._pattern.match(p.name)
            if not m:
                continue
            gen = int(m.group(1))
            checkpoints.append((gen, p))

        checkpoints.sort(key=lambda t: t[0])  # ascending by gen

        # remove oldest if too many
        to_remove = max(0, len(checkpoints) - self.keep_last)
        for i in range(to_remove):
            _, path = checkpoints[i]
            try:
                path.unlink()
            except OSError:
                # If delete fails, do not crash the run.
                pass
