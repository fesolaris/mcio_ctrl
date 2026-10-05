from unittest.mock import MagicMock

import cbor2
import numpy as np

from mcio_ctrl import network, types
from mcio_ctrl.envs import mcio_env


def test_send_frame_absent_from_wire_when_unset() -> None:
    packed = network.ActionPacket().pack()
    raw = cbor2.loads(packed)
    assert "send_frame" not in raw


def test_send_frame_encoded_when_set() -> None:
    packed = network.ActionPacket(send_frame=False).pack()
    raw = cbor2.loads(packed)
    assert raw["send_frame"] is False
    decoded = network.ActionPacket.unpack(packed)
    assert decoded is not None
    assert decoded.send_frame is False


def test_observation_roundtrip_without_frame() -> None:
    packet = network.ObservationPacket(sequence=3, frame=b"", health=20.0)
    decoded = network.ObservationPacket.unpack(packet.pack())
    assert decoded is not None
    assert not decoded.has_frame()
    assert decoded.sequence == 3


def obs_with_frame(frame: bytes, height: int = 2, width: int = 2) -> network.ObservationPacket:
    return network.ObservationPacket(
        sequence=1,
        frame=frame,
        frame_height=height,
        frame_width=width,
        health=20.0,
    )


def test_update_state_keeps_previous_frame(
    mock_controller: dict[str, MagicMock], action_space_sample1: mcio_env.MCioAction
) -> None:
    frame = bytes(range(12))
    env = mcio_env.MCioEnv(types.RunOptions(mcio_mode=types.MCioMode.SYNC))
    assert isinstance(env, mcio_env.MCioEnv)
    ctrl = mock_controller["ctrl_sync"].return_value
    ctrl.recv_observation.return_value = obs_with_frame(frame)
    env.reset()
    first = env.last_frame
    assert first is not None and first.shape == (2, 2, 3)

    ctrl.recv_observation.return_value = obs_with_frame(b"")
    env.step(action_space_sample1)
    assert env.last_frame is first or np.array_equal(env.last_frame, first)


def test_update_state_empty_frame_before_first_real_frame(
    mock_controller: dict[str, MagicMock],
) -> None:
    env = mcio_env.MCioEnv(types.RunOptions(mcio_mode=types.MCioMode.SYNC))
    assert env.last_frame is None
    ctrl = mock_controller["ctrl_sync"].return_value
    ctrl.recv_observation.return_value = obs_with_frame(b"")
    env.reset()
    assert env.last_frame is not None
    assert env.last_frame.shape == (0, 0, 3)


def test_step_send_frame_option(
    mock_controller: dict[str, MagicMock], action_space_sample1: mcio_env.MCioAction
) -> None:
    env = mcio_env.MCioEnv(types.RunOptions(mcio_mode=types.MCioMode.SYNC))
    ctrl = mock_controller["ctrl_sync"].return_value
    ctrl.recv_observation.return_value = obs_with_frame(bytes(range(12)))
    env.reset()
    send_action = ctrl.send_action

    env.step(action_space_sample1, options={"send_frame": False})
    packet = send_action.call_args.args[0]
    assert isinstance(packet, network.ActionPacket)
    assert packet.send_frame is False

    env.step(action_space_sample1)
    packet = send_action.call_args.args[0]
    assert packet.send_frame is None


def test_step_send_frame_can_be_true(
    mock_controller: dict[str, MagicMock], action_space_sample1: mcio_env.MCioAction
) -> None:
    env = mcio_env.MCioEnv(types.RunOptions(mcio_mode=types.MCioMode.SYNC))
    ctrl = mock_controller["ctrl_sync"].return_value
    ctrl.recv_observation.return_value = obs_with_frame(bytes(range(12)))
    env.reset()

    env.step(action_space_sample1, options={"send_frame": True})
    packet = ctrl.send_action.call_args.args[0]
    assert packet.send_frame is True
