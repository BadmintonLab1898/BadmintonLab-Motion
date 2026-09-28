# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import _has_hit


def _event_scale(env: ManagerBasedRLEnv) -> float:
    """Undo RewardManager's global dt scaling for one-shot event rewards."""
    return 1.0 / max(float(env.step_dt), 1e-6)

def root_height_above_minimum_reward(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    height_deficit = torch.clamp(minimum_height - robot.data.root_pos_w[:, 2], min=0.0)
    return torch.exp(-height_deficit)

def root_rpy_limit_reward(
    env: ManagerBasedRLEnv,
    max_rp_angle_deg: float = 45.0,
    max_yaw_angle_deg: float = 90.0,
    rad_sigma: float = 10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    rp_limit = torch.deg2rad(torch.tensor(max_rp_angle_deg, device=roll.device, dtype=roll.dtype))
    roll_excess = torch.clamp(torch.abs(roll) - rp_limit, min=0.0)
    pitch_excess = torch.clamp(torch.abs(pitch) - rp_limit, min=0.0)
    yaw_limit = torch.deg2rad(torch.tensor(max_yaw_angle_deg, device=yaw.device, dtype=yaw.dtype))
    yaw_excess = torch.clamp(torch.abs(yaw) - yaw_limit, min=0.0)
    total_excess = roll_excess + pitch_excess + yaw_excess
    return torch.exp(-total_excess*rad_sigma)

def root_state_valid_factor(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    minimum_height: float = 0.5,
    max_rp_angle_deg: float = 45.0,
    max_yaw_angle_deg: float = 90.0,
) -> torch.Tensor:
    robot: RigidObject = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)

    rp_limit = torch.deg2rad(torch.tensor(max_rp_angle_deg, device=roll.device, dtype=roll.dtype))
    yaw_limit = torch.deg2rad(torch.tensor(max_yaw_angle_deg, device=yaw.device, dtype=yaw.dtype))

    height_valid = (robot.data.root_pos_w[:, 2] >= minimum_height).float()
    roll_valid = (roll.abs() <= rp_limit).float()
    pitch_valid = (pitch.abs() <= rp_limit).float()
    yaw_valid = (yaw.abs() <= yaw_limit).float()
    return height_valid * roll_valid * pitch_valid * yaw_valid

def hit_bonus(
    env: ManagerBasedRLEnv,
    required_hit_side: str = "forehand",
    local_axis: int = 0,
    side_margin: float = 0.0,
) -> torch.Tensor:
    has_hit = _has_hit(env)
    just_hit = has_hit & (env.hit_step == env.episode_length_buf)

    robot: Articulation = env.scene["robot"]
    sweet_pos_w = robot.data.body_pos_w[:, env._sweet_spot_body_id, :]
    sweet_quat_w = robot.data.body_quat_w[:, env._sweet_spot_body_id, :]
    hit_pos_local = quat_apply_inverse(sweet_quat_w, env.hit_pos - sweet_pos_w)
    side_value = hit_pos_local[:, int(local_axis)]
    margin = max(float(side_margin), 0.0)
    if required_hit_side == "forehand":
        # For this racket model, positive local-X (the racket-face normal)
        # is the forehand side.
        side_valid = side_value > margin
    elif required_hit_side == "backhand":
        side_valid = side_value < -margin
    else:
        raise ValueError("required_hit_side must be 'forehand' or 'backhand'")

    root_valid_factor = root_state_valid_factor(env)
    return just_hit.float() * side_valid.float() * root_valid_factor * _event_scale(env)

def post_hit_robot_stable_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    follow_time_s: float = 0.35,
    position_sigma: float = 0.35,
    heading_zero_angle_deg: float = 90.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reward returning to the nominal court position and heading after a hit."""
    robot: Articulation = env.scene[asset_cfg.name]
    has_hit = _has_hit(env)
    follow_steps = max(int(round(follow_time_s / float(env.step_dt))), 0)
    after_hit_follow_time = has_hit & (env.post_hit_steps > follow_steps)

    court = env.cfg.court
    robot_pos_local = robot.data.root_pos_w - env.scene.env_origins
    center = torch.zeros_like(robot_pos_local[:, :2])
    center[:, 0] = -0.5 * court.half_length
    center_error = torch.linalg.norm(robot_pos_local[:, :2] - center, dim=-1)
    position_reward = torch.exp(-(center_error / max(position_sigma, eps)))

    forward_b = torch.tensor((1.0, 0.0, 0.0), device=robot.device, dtype=robot.data.root_quat_w.dtype)
    forward_b = forward_b.expand(robot.data.root_quat_w.shape[0], 3)
    forward_w = quat_apply(robot.data.root_quat_w, forward_b)
    heading_xy = forward_w[:, :2]
    heading_norm = torch.clamp(torch.linalg.norm(heading_xy, dim=-1), min=eps)
    heading_cos = heading_xy[:, 0] / heading_norm
    zero_heading_cos = torch.cos(
        torch.deg2rad(torch.tensor(heading_zero_angle_deg, device=robot.device, dtype=heading_cos.dtype))
    )
    heading_reward = torch.clamp(
        (heading_cos - zero_heading_cos) / torch.clamp(1.0 - zero_heading_cos, min=eps),
        0.0,
        1.0,
    )

    recovery_reward = torch.sqrt(torch.clamp(position_reward * heading_reward, min=0.0))
    active = after_hit_follow_time.float()

    active_count = active.sum().clamp_min(1.0)
    log = env.extras.setdefault("log", {})
    log["Metrics/Recovery/post_hit_score"] = torch.sum(recovery_reward * active) / active_count
    log["Metrics/Recovery/post_hit_position"] = torch.sum(position_reward * active) / active_count
    log["Metrics/Recovery/post_hit_heading"] = torch.sum(heading_reward * active) / active_count
    log["Metrics/Recovery/post_hit_active_fraction"] = active.mean()

    return active * recovery_reward


def foot_slip_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    foot_height_threshold: float = 0.08,
) -> torch.Tensor:
    """Penalize horizontal foot motion while the feet are close to the ground."""
    robot: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    foot_height = foot_pos_w[..., 2]
    foot_speed_xy = torch.linalg.norm(foot_vel_w[..., :2], dim=-1)
    foot_speed_xy = torch.clamp(foot_speed_xy, min=0.0, max=5.0)
    near_ground = (foot_height < foot_height_threshold).float()

    return torch.sum(foot_speed_xy * near_ground, dim=1)

def racket_ground_support_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 3.0,
    force_scale: float = 10.0,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """Penalize sustained, high-force racket contact with the ground.

    This reads the contact sensor's filtered force matrix, so contacts with
    the shuttlecock or other robot bodies are excluded. The contact must stay
    above ``force_threshold`` throughout the configured sensor history before
    the penalty is activated.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history = sensor.data.force_matrix_w_history[:, :, sensor_cfg.body_ids]
    force_history = torch.linalg.norm(force_history, dim=-1).amax(dim=-1)

    sustained_contact = torch.all(force_history > max(force_threshold, 0.0), dim=1).any(dim=-1)
    current_force = force_history[:, 0].amax(dim=-1)
    penalty = torch.clamp(current_force / max(force_scale, 1e-6), min=0.0, max=max_penalty)
    return sustained_contact.float() * penalty

def net_cross_bonus(
    env: ManagerBasedRLEnv,
    net_full_reward_height: float = 1.55,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reward outgoing net-crossing height as ``clamp(h / full_height, 0, 1)``."""
    prev_pos = getattr(env, "prev_shuttle_pos_w", None)
    curr_pos = getattr(env, "curr_shuttle_pos_w", None)
    prev_valid = getattr(env, "_prev_pos_valid", None)
    if prev_pos is None or curr_pos is None or prev_valid is None:
        return torch.zeros(env.num_envs, device=env.device)

    court = env.cfg.court
    env_origins = env.scene.env_origins
    prev_local = prev_pos - env_origins
    curr_local = curr_pos - env_origins

    # Only reward travel from the robot half (-X) into the opponent half (+X).
    crossed_net = (prev_local[:, 0] <= court.net_x) & (curr_local[:, 0] > court.net_x)
    dx = curr_local[:, 0] - prev_local[:, 0]
    safe_dx = torch.where(dx.abs() > eps, dx, torch.ones_like(dx))
    net_t = (court.net_x - prev_local[:, 0]) / safe_dx
    h = prev_local[:, 2] + net_t * (curr_local[:, 2] - prev_local[:, 2])
    height_reward = torch.clamp(h / max(net_full_reward_height, eps), 0.0, 1.0)
    net_event = prev_valid & _has_hit(env) & crossed_net
    return height_reward * net_event.float() * root_state_valid_factor(env) * _event_scale(env)

def backcourt_landing_reward(
    env: ManagerBasedRLEnv,
    landing_height: float = 0.03,
    backcourt_depth: float = 1.50,
    target_half_width: float = 0.75,
    distance_sigma: float = 0.50,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reward landing near the central target area of the opponent backcourt."""
    prev_pos = getattr(env, "prev_shuttle_pos_w", None)
    curr_pos = getattr(env, "curr_shuttle_pos_w", None)
    prev_valid = getattr(env, "_prev_pos_valid", None)
    if prev_pos is None or curr_pos is None or prev_valid is None:
        return torch.zeros(env.num_envs, device=env.device)

    court = env.cfg.court
    env_origins = env.scene.env_origins
    prev_local = prev_pos - env_origins
    curr_local = curr_pos - env_origins
    # The episode ends at landing_height, so extrapolate the final segment to
    # z=0 instead of requiring the simulator to produce a sample below zero.
    landed = (prev_local[:, 2] >= landing_height) & (curr_local[:, 2] < landing_height)
    dz = curr_local[:, 2] - prev_local[:, 2]
    safe_dz = torch.where(dz.abs() > eps, dz, -torch.ones_like(dz))
    ground_t = (0.0 - prev_local[:, 2]) / safe_dz
    ground_xy = prev_local[:, :2] + ground_t.unsqueeze(-1) * (curr_local[:, :2] - prev_local[:, :2])

    target_x_min = court.half_length - max(backcourt_depth, 0.0)
    target_x_max = court.half_length
    half_width = max(target_half_width, 0.0)
    dx_to_region = torch.clamp(target_x_min - ground_xy[:, 0], min=0.0) + torch.clamp(
        ground_xy[:, 0] - target_x_max, min=0.0
    )
    dy_to_region = torch.clamp(ground_xy[:, 1].abs() - half_width, min=0.0)
    distance = torch.linalg.norm(torch.stack((dx_to_region, dy_to_region), dim=-1), dim=-1)
    landing_reward = torch.exp(-(distance / max(distance_sigma, eps)))
    landing_event = prev_valid & _has_hit(env) & landed

    reward = landing_reward * landing_event.float()
    return reward * root_state_valid_factor(env) * _event_scale(env)

def interval_approach_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float = 1.5,
    max_height: float = 2.0,
    incoming_speed_epsilon: float = 0.05,
    full_reward_radius: float = 0.1,
    distance_sigma: float = 0.45,
) -> torch.Tensor:
    """Reward the closest racket approach during the incoming height interval."""
    del min_height, max_height, incoming_speed_epsilon
    distance = getattr(env, "approach_event_distance", None)
    valid = getattr(env, "approach_reward_event", None)
    if distance is None or valid is None:
        return torch.zeros(env.num_envs, device=env.device)
    outside = torch.clamp(distance - max(full_reward_radius, 0.0), min=0.0)
    proximity = torch.exp(-outside / max(distance_sigma, 1e-6))
    return valid.float() * proximity * root_state_valid_factor(env, asset_cfg) * _event_scale(env)

def dynamic_outgoing_velocity_reward(
    env: ManagerBasedRLEnv,
    shuttle_cfg: SceneEntityCfg,
    target_speed: float = 12.0,
    minimum_net_clearance_height: float = 1.55,
    direction_sigma: float = 0.15,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reward initial shuttle speed only when its direction can clear the net.

    The valid directions form an open-topped pyramid bounded by the net cord
    and the two net posts. Directions inside it receive a factor of one;
    outside it, the factor decays smoothly with the normalized violation of
    its four bounding half-spaces.
    """
    shuttle: RigidObject = env.scene[shuttle_cfg.name]

    shuttle_vel = shuttle.data.root_lin_vel_w
    shuttle_speed = torch.linalg.norm(shuttle_vel, dim=-1)
    speed_reward = torch.clamp(shuttle_speed / max(target_speed, eps), 0.0, 1.0)

    court = env.cfg.court
    hit_pos_local = env.hit_pos - env.scene.env_origins
    direction = shuttle_vel / torch.clamp(shuttle_speed.unsqueeze(-1), min=eps)
    net_distance = torch.clamp(court.net_x - hit_pos_local[:, 0], min=eps)
    half_width = court.singles_half_width if court.use_singles_bounds else court.doubles_half_width

    # Each row describes a half-space n . direction <= 0. Dividing by
    # ||n|| makes its positive violation comparable across hit positions.
    normals = torch.stack(
        (
            torch.stack((-torch.ones_like(net_distance), torch.zeros_like(net_distance), torch.zeros_like(net_distance)), dim=-1),
            torch.stack((hit_pos_local[:, 1] - half_width, net_distance, torch.zeros_like(net_distance)), dim=-1),
            torch.stack((-hit_pos_local[:, 1] - half_width, -net_distance, torch.zeros_like(net_distance)), dim=-1),
            torch.stack((minimum_net_clearance_height - hit_pos_local[:, 2], torch.zeros_like(net_distance), -net_distance), dim=-1),
        ),
        dim=1,
    )
    signed_violations = torch.sum(normals * direction.unsqueeze(1), dim=-1)
    normalized_violations = torch.relu(signed_violations) / torch.clamp(
        torch.linalg.norm(normals, dim=-1), min=eps
    )
    direction_error = torch.linalg.norm(normalized_violations, dim=-1)
    valid_direction = torch.exp(-(direction_error / max(direction_sigma, eps)))

    evaluate_outgoing_velocity = _has_hit(env) & (env.hit_step + 1 == env.episode_length_buf)

    return (
        speed_reward
        * valid_direction
        * evaluate_outgoing_velocity.float()
        * root_state_valid_factor(env)
        * _event_scale(env)
    )



def target_landing_reward(
    env: ManagerBasedRLEnv,
    target_x_range: tuple[float, float],
    target_y_range: tuple[float, float],
    landing_height: float = 0.03,
    distance_sigma: float = 0.50,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reward a valid shot for landing in a rectangular court target."""
    previous = getattr(env, "prev_shuttle_pos_w", None)
    current = getattr(env, "curr_shuttle_pos_w", None)
    previous_valid = getattr(env, "_prev_pos_valid", None)
    if previous is None or current is None or previous_valid is None:
        return torch.zeros(env.num_envs, device=env.device)

    local_previous = previous - env.scene.env_origins
    local_current = current - env.scene.env_origins
    descended_to_landing = (local_previous[:, 2] >= landing_height) & (local_current[:, 2] < landing_height)

    dz = local_current[:, 2] - local_previous[:, 2]
    safe_dz = torch.where(dz.abs() > eps, dz, -torch.ones_like(dz))
    ground_t = (0.0 - local_previous[:, 2]) / safe_dz
    ground_xy = local_previous[:, :2] + ground_t.unsqueeze(-1) * (
        local_current[:, :2] - local_previous[:, :2]
    )

    x_min, x_max = sorted((float(target_x_range[0]), float(target_x_range[1])))
    y_min, y_max = sorted((float(target_y_range[0]), float(target_y_range[1])))
    dx = torch.clamp(x_min - ground_xy[:, 0], min=0.0) + torch.clamp(ground_xy[:, 0] - x_max, min=0.0)
    dy = torch.clamp(y_min - ground_xy[:, 1], min=0.0) + torch.clamp(ground_xy[:, 1] - y_max, min=0.0)
    distance = torch.linalg.vector_norm(torch.stack((dx, dy), dim=-1), dim=-1)
    proximity = torch.exp(-(distance / max(distance_sigma, eps)))

    landing_event = previous_valid & _has_hit(env) & descended_to_landing
    return (
        proximity
        * landing_event.float()
        * root_state_valid_factor(env)
        * (1.0 / max(float(env.step_dt), 1e-6))
    )


def hit_bonus_any_side(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Reward a valid hit on either side of the racket."""
    has_hit = _has_hit(env)
    just_hit = has_hit & (env.hit_step == env.episode_length_buf)
    return just_hit.float() * root_state_valid_factor(env) * (1.0 / max(float(env.step_dt), 1e-6))
