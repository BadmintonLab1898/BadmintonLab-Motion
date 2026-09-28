from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_heading_court(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot heading as a unit vector in the court/world xy frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    forward_b = torch.tensor((1.0, 0.0, 0.0), device=robot.data.root_quat_w.device, dtype=robot.data.root_quat_w.dtype)
    forward_b = forward_b.expand(robot.data.root_quat_w.shape[0], 3)
    forward_w = quat_apply(robot.data.root_quat_w, forward_b)
    heading_xy = forward_w[:, :2]
    return heading_xy / torch.clamp(torch.linalg.norm(heading_xy, dim=-1, keepdim=True), min=1e-6)


def robot_pos_court(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot root position in court-local coordinates."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_pos_w - env.scene.env_origins


def shuttle_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    shuttle_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Shuttle position expressed in the robot base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    shuttle: RigidObject = env.scene[shuttle_cfg.name]
    rel_w = shuttle.data.root_pos_w - robot.data.root_pos_w
    return quat_apply_inverse(robot.data.root_quat_w, rel_w)


def shuttle_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    shuttle_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Shuttle linear velocity expressed in the robot base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    shuttle: RigidObject = env.scene[shuttle_cfg.name]
    rel_vel_w = shuttle.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return quat_apply_inverse(robot.data.root_quat_w, rel_vel_w)


def racket_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body position expressed in the robot base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    rel_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :][:, 0] - robot.data.root_pos_w
    return quat_apply_inverse(robot.data.root_quat_w, rel_w)


def racket_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body linear velocity expressed in the robot base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    rel_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :][:, 0] - robot.data.root_lin_vel_w
    return quat_apply_inverse(robot.data.root_quat_w, rel_vel_w)


def racket_ang_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body angular velocity expressed in the robot base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    rel_ang_vel_w = robot.data.body_ang_vel_w[:, asset_cfg.body_ids, :][:, 0] - robot.data.root_ang_vel_w
    return quat_apply_inverse(robot.data.root_quat_w, rel_ang_vel_w)


def shuttle_pos_in_sweet_spot_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    shuttle_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Shuttle position expressed in the sweet-spot body frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    shuttle: RigidObject = env.scene[shuttle_cfg.name]
    sweet_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :][:, 0]
    sweet_quat = robot.data.body_quat_w[:, asset_cfg.body_ids, :][:, 0]
    return quat_apply_inverse(sweet_quat, shuttle.data.root_pos_w - sweet_pos)


def shuttle_vel_in_sweet_spot_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    shuttle_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Shuttle linear velocity expressed in the sweet-spot body frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    shuttle: RigidObject = env.scene[shuttle_cfg.name]
    sweet_quat = robot.data.body_quat_w[:, asset_cfg.body_ids, :][:, 0]
    sweet_lin_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :][:, 0]
    return quat_apply_inverse(sweet_quat, shuttle.data.root_lin_vel_w - sweet_lin_vel_w)
