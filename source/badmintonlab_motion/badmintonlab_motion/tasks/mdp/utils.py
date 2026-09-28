# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_FORWARD_AXIS_B = (1.0, 0.0, 0.0)


def _has_hit(env: ManagerBasedRLEnv) -> torch.Tensor:
    return getattr(env, "has_hit", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))


def _robot_forward_w(robot: Articulation) -> torch.Tensor:
    forward_b = torch.tensor(_FORWARD_AXIS_B, device=robot.data.root_quat_w.device, dtype=robot.data.root_quat_w.dtype)
    forward_b = forward_b.expand(robot.data.root_quat_w.shape[0], 3)
    return quat_apply(robot.data.root_quat_w, forward_b)


def pre_hit_only(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Float mask that is 1 before the first valid hit, 0 afterwards."""
    return (~_has_hit(env)).float()
