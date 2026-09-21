from unittest.mock import MagicMock

import numpy as np
import pytest

from mcio_ctrl import types
from mcio_ctrl.envs import mcio_env


@pytest.fixture
def default_mcio_env() -> mcio_env.MCioEnv:
    return mcio_env.MCioEnv(types.RunOptions())


@pytest.fixture
def action_space_sample1(default_mcio_env: mcio_env.MCioEnv) -> mcio_env.MCioAction:
    act = default_mcio_env.get_noop_action()
    act.update(
        {
            "cursor_delta": np.array([827, 22], dtype=np.int32),
            "A": 1,
            "W": 0,
            "LEFT_SHIFT": 1,
            "LEFT_BUTTON": 1,
        }
    )
    return act


@pytest.fixture
def mock_controller(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    mock_ctrl_sync = MagicMock()
    mock_ctrl_async = MagicMock()
    monkeypatch.setattr("mcio_ctrl.controller.ControllerSync", mock_ctrl_sync)
    monkeypatch.setattr("mcio_ctrl.controller.ControllerAsync", mock_ctrl_async)

    # Return objects that tests might need to access
    return {
        "ctrl_sync": mock_ctrl_sync,
        "ctrl_async": mock_ctrl_async,
    }
