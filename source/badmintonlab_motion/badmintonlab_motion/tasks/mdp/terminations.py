# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import _has_hit


def no_hit_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode timeout that only applies before the first valid hit."""
    return (env.episode_length_buf >= env.max_episode_length) & (~_has_hit(env))


def delay_success_hit(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Boolean termination triggered 1 second after a successful hit."""
    hit_window_steps = int(round(1.0 / float(env.step_dt)))
    return _has_hit(env) & (env.post_hit_steps >= hit_window_steps)


def shuttle_landed(env: ManagerBasedRLEnv, minimum_height: float = 0.03) -> torch.Tensor:
    """End the episode once a validly hit shuttle has dropped to the court."""
    shuttle = env.scene["shuttle"]
    return _has_hit(env) & (shuttle.data.root_pos_w[:, 2] < minimum_height)


def net_fault_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End the episode the step the shuttle is judged to have gone into the net."""
    return getattr(env, "net_fault", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))


def out_of_bounds_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End the episode the step the shuttle is judged to be out of bounds."""
    return getattr(env, "out_fault", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))


def root_vertical_deviation_exceed_limit(
    env: ManagerBasedRLEnv,
    max_angle_deg: float = 45.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot root tilts too far away from the world vertical."""
    robot: Articulation = env.scene[asset_cfg.name]

    up_axis_b = torch.tensor((0.0, 0.0, 1.0), device=robot.data.root_quat_w.device, dtype=robot.data.root_quat_w.dtype)
    up_axis_b = up_axis_b.expand(robot.data.root_quat_w.shape[0], 3)
    up_axis_w = quat_apply(robot.data.root_quat_w, up_axis_b)

    cos_angle = torch.clamp(up_axis_w[:, 2], -1.0, 1.0)
    angle = torch.acos(cos_angle)
    angle_limit = torch.deg2rad(torch.tensor(max_angle_deg, device=angle.device, dtype=angle.dtype))
    return angle > angle_limit
