"""Two-phase, multi-client driver for SYNC multiplayer.

A server-side barrier only ticks after every client has sent its action, so a blocking
per-env step() in sequence deadlocks. This wrapper sends every action first, then reads
every observation.
"""

from collections.abc import Sequence
from typing import Any, Generic

from .base_env import ActType, MCioBaseEnv, ObsType, ResetOptions


class MCioMultiEnv(Generic[ObsType, ActType]):
    def __init__(self, envs: Sequence[MCioBaseEnv[ObsType, ActType]]) -> None:
        assert len(envs) > 0
        self.envs = envs

    def __len__(self) -> int:
        return len(self.envs)

    def send_actions(
        self,
        actions: Sequence[ActType],
        options: Sequence[ResetOptions] | None = None,
    ) -> None:
        assert len(actions) == len(self.envs)
        assert options is None or len(options) == len(self.envs)
        options = options or [ResetOptions() for _ in self.envs]
        assert not any(e.terminated for e in self.envs)
        for env, action, opt in zip(self.envs, actions, options):
            env.begin_step(action, opt)

    def recv_steps(
        self,
        actions: Sequence[ActType],
    ) -> list[tuple[ObsType, int, bool, bool, dict[Any, Any]]]:
        assert len(actions) == len(self.envs)
        return [env.end_step(action) for env, action in zip(self.envs, actions)]

    def step(
        self,
        actions: Sequence[ActType],
        options: Sequence[ResetOptions] | None = None,
    ) -> list[tuple[ObsType, int, bool, bool, dict[Any, Any]]]:
        self.send_actions(actions, options)
        return self.recv_steps(actions)

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def skip_steps(
        self, n_steps: int
    ) -> list[tuple[ObsType, int, bool, bool, dict[Any, Any]]]:
        """Advance n server ticks with empty actions on every env, in lockstep"""
        assert n_steps > 0
        for _ in range(n_steps):
            for env in self.envs:
                env.send_noop()
            observations = [env.recv_observation() for env in self.envs]
        return [
            (obs, 0, env.terminated, False, {})
            for env, obs in zip(self.envs, observations, strict=True)
        ]

    def reset(
        self,
        seeds: Sequence[int | None] | None = None,
        options: Sequence[ResetOptions] | None = None,
        max_respawn_steps: int = 100,
    ) -> list[tuple[ObsType, dict[Any, Any]]]:
        """Reset every env together, then tick all of them until everyone has respawned."""
        seeds = seeds or [None] * len(self.envs)
        options = options or [ResetOptions() for _ in self.envs]
        for env, seed, opt in zip(self.envs, seeds, options, strict=True):
            env.begin_reset(seed, opt)
        results = [env.end_reset() for env in self.envs]

        # Lockstep replacement for per-env _reset_terminated_hack()
        for _ in range(max_respawn_steps):
            if not any(env.terminated for env in self.envs):
                break
            steps = self.skip_steps(1)
            for env in self.envs:
                if env.health > 0.0:
                    env.terminated = False
            results = [
                (obs, env._get_info()) for env, (obs, *_) in zip(self.envs, steps)
            ]
        else:
            raise RuntimeError(
                f"Environments remained terminated after {max_respawn_steps} steps."
            )
        return results
