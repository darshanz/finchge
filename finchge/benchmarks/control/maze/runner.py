from typing import Any, Callable, Dict, List, Optional

from finchge.benchmarks.control.maze.environment import MazeEnvironment
from finchge.benchmarks.control.maze.interpreter import MazeInterpreter
from finchge.runners import ControlRunner


class MazeRunner(ControlRunner[MazeEnvironment]):
    def __init__(
        self,
        env_factory: Callable[[], MazeEnvironment],
        n_episodes: int = 1,
        interpreter: Optional[MazeInterpreter] = None,
        optimal_fitness: Optional[float] = None,
        random_state: Optional[Any] = None,
    ) -> None:
        super().__init__(
            random_state=random_state,
            env_factory=env_factory,
            n_episodes=n_episodes,
            optimal_fitness=optimal_fitness,
        )
        self.interpreter = interpreter or MazeInterpreter()
        self._last_rewards: List[float] = []

    def _run_episode(self, phenotype: str, env: MazeEnvironment) -> float:
        tokens = self.interpreter.tokenize(phenotype)
        env.reset()
        program_counter = 0
        total_reward = 0.0

        while not env.is_done():
            observation = env.get_observation()
            action, program_counter = self.interpreter.evaluate(
                tokens, observation, program_counter
            )

            if action is None:
                actions = env.get_available_actions()
                action = self.rng.choice(actions) if actions else None
                program_counter = 0

            if action is not None:
                _, reward, _, _ = env.step(action)
                total_reward += reward
            if program_counter >= len(tokens):
                program_counter = 0

        return total_reward

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        if state is not None:
            state.update(
                {
                    "interpreter": self.interpreter,
                    "_last_rewards": self._last_rewards,
                }
            )
        return state or {}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    @property
    def provided_context_keys(self) -> set[str]:
        return super().provided_context_keys | {"y_pred"}
