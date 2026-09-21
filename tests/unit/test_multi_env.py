from typing import Any
from unittest.mock import MagicMock

import pytest

from mcio_ctrl import types
from mcio_ctrl.envs import base_env, mcio_env, multi_env

# All envs share the same mocked controller instance, so its mock_calls record
# the interleaving of sends and receives across envs.
SEND = "send_action"
RECV = "recv_observation"


def _make_multi(n_envs: int = 2) -> multi_env.MCioMultiEnv[Any, Any]:
    envs: list[base_env.MCioBaseEnv[Any, Any]] = [
        mcio_env.MCioEnv(types.RunOptions(mcio_mode=types.MCioMode.SYNC))
        for _ in range(n_envs)
    ]
    return multi_env.MCioMultiEnv(envs)


def _io_calls(ctrl: MagicMock) -> list[str]:
    return [c[0] for c in ctrl.mock_calls if c[0] in (SEND, RECV)]


def test_step_sends_all_before_receiving(
    mock_controller: dict[str, MagicMock], action_space_sample1: mcio_env.MCioAction
) -> None:
    ctrl = mock_controller["ctrl_sync"].return_value
    multi = _make_multi()
    multi.reset()
    ctrl.reset_mock()

    multi.step([action_space_sample1] * 2)

    assert _io_calls(ctrl) == [SEND, SEND, RECV, RECV]


def test_step_length_mismatch(
    mock_controller: dict[str, MagicMock], action_space_sample1: mcio_env.MCioAction
) -> None:
    multi = _make_multi()
    multi.reset()

    with pytest.raises(AssertionError):
        multi.step([action_space_sample1])
    with pytest.raises(AssertionError):
        multi.step([action_space_sample1] * 2, options=[base_env.ResetOptions()])


def test_step_terminated_sends_nothing(
    mock_controller: dict[str, MagicMock], action_space_sample1: mcio_env.MCioAction
) -> None:
    ctrl = mock_controller["ctrl_sync"].return_value
    multi = _make_multi()
    multi.reset()
    multi.envs[1].terminated = True
    ctrl.reset_mock()

    with pytest.raises(AssertionError):
        multi.step([action_space_sample1] * 2)
    ctrl.send_action.assert_not_called()


def test_skip_steps_lockstep(mock_controller: dict[str, MagicMock]) -> None:
    ctrl = mock_controller["ctrl_sync"].return_value
    multi = _make_multi()
    multi.reset()
    ctrl.reset_mock()

    results = multi.skip_steps(2)

    assert _io_calls(ctrl) == [SEND, SEND, RECV, RECV] * 2
    assert len(results) == 2


def test_skip_steps_zero(mock_controller: dict[str, MagicMock]) -> None:
    multi = _make_multi()
    multi.reset()
    with pytest.raises(AssertionError):
        multi.skip_steps(0)


def test_reset_sends_all_before_receiving(
    mock_controller: dict[str, MagicMock],
) -> None:
    ctrl = mock_controller["ctrl_sync"].return_value
    multi = _make_multi()

    multi.reset()

    assert _io_calls(ctrl) == [SEND, SEND, RECV, RECV]


def test_reset_respawns_in_lockstep(mock_controller: dict[str, MagicMock]) -> None:
    ctrl = mock_controller["ctrl_sync"].return_value
    # fmt: off
    healths = [
        0.0, 20.0,   # reset: env 0 dead
        0.0, 20.0,   # respawn tick 1: env 0 still dead
        20.0, 20.0,  # respawn tick 2: env 0 back
    ]
    # fmt: on
    ctrl.recv_observation.side_effect = [MagicMock(health=h) for h in healths]
    multi = _make_multi()

    multi.reset()

    # Both envs are ticked on every respawn step, not just the dead one
    assert _io_calls(ctrl) == [SEND, SEND, RECV, RECV] * 3
    assert not any(env.terminated for env in multi.envs)


def test_reset_respawn_gives_up(mock_controller: dict[str, MagicMock]) -> None:
    ctrl = mock_controller["ctrl_sync"].return_value
    ctrl.recv_observation.side_effect = lambda: MagicMock(health=0.0)
    multi = _make_multi()

    with pytest.raises(RuntimeError):
        multi.reset(max_respawn_steps=3)
