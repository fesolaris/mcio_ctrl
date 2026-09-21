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
        for env, action, opt in zip(self.envs, actions, options, strict=True):
            env.begin_step(action, opt)

    def recv_steps(
        self,
        actions: Sequence[ActType],
    ) -> list[tuple[ObsType, int, bool, bool, dict[Any, Any]]]:
        """Receive every env's step. If any agent terminated, the episode ends for all:
        every env is marked terminated, and info["agent_terminated"] says who died."""
        assert len(actions) == len(self.envs)
        results = [
            env.end_step(action) for env, action in zip(self.envs, actions, strict=True)
        ]
        episode_over = any(term for _, _, term, _, _ in results)
        if episode_over:
            for env in self.envs:
                env.terminated = True
        return [
            (obs, reward, episode_over, trunc, {**info, "agent_terminated": term})
            for obs, reward, term, trunc, info in results
        ]

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
            (obs, 0, env.terminated, False, env.get_info())
            for env, obs in zip(self.envs, observations, strict=True)
        ]

    @property
    def any_terminated(self) -> bool:
        return any(env.terminated for env in self.envs)

    def reset(
        self,
        seeds: Sequence[int | None] | None = None,
        options: Sequence[ResetOptions] | None = None,
        max_respawn_steps: int = 100,
    ) -> list[tuple[ObsType, dict[Any, Any]]]:
        """Reset every env together. The first call launches/connects; later calls
        reset over the existing connections (relaunching the LAN host drops the guest).
        """
        relaunch = any(env.ctrl is None for env in self.envs)
        seeds = seeds or [None] * len(self.envs)
        options = options or [ResetOptions() for _ in self.envs]
        if not relaunch:
            # Reset commands don't apply to a dead player
            self._wait_for_respawn(max_respawn_steps)
        for env, seed, opt in zip(self.envs, seeds, options, strict=True):
            env.begin_reset(seed, opt, relaunch=relaunch)
        results = [env.end_reset() for env in self.envs]

        return self._wait_for_respawn(max_respawn_steps) or results

    def _wait_for_respawn(
        self, max_respawn_steps: int
    ) -> list[tuple[ObsType, dict[Any, Any]]] | None:
        """Tick all envs until every player is alive.
        Returns the latest observations if any ticks were needed."""
        results = None
        for _ in range(max_respawn_steps):
            for env in self.envs:
                if env.terminated and env.health > 0.0:
                    env.terminated = False
            if not self.any_terminated:
                return results
            results = [(obs, info) for obs, _, _, _, info in self.skip_steps(1)]
        raise RuntimeError(
            f"Environments remained terminated after {max_respawn_steps} steps."
        )
