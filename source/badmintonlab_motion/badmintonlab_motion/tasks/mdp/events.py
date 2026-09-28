from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files import spawn_from_urdf
from isaaclab.sim.utils import bind_physics_material
from isaacsim.core.utils.stage import get_current_stage

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

_SHUTTLE_RESET_POOL: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_SHUTTLE_RESET_POOL_SIZE = 4096


def spawn_urdf_with_racket_material(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    material_path = f"/World/PhysicsMaterials/{prim.GetPath().pathString.rsplit('/', 1)[-1].lower()}_racket"
    material_cfg = sim_utils.RigidBodyMaterialCfg(static_friction=0.2, dynamic_friction=0.15,
        restitution=0.75, friction_combine_mode="average", restitution_combine_mode="max")
    material_cfg.func(material_path, material_cfg)
    stage = get_current_stage()
    racket_path = f"{prim.GetPath().pathString}/racket_link"
    pending = [stage.GetPrimAtPath(racket_path)]
    while pending:
        child = pending.pop()
        if child.IsInstance(): child.SetInstanceable(False)
        pending.extend(child.GetChildren())
    bind_physics_material(racket_path, material_path, stage=stage)
    return prim

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv,ManagerBasedRLEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos

def reset_shuttle_root_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    position_range: dict[str, tuple[float, float]],
    flight_time_range: tuple[float, float],
    target_x_range: tuple[float, float],
    target_y_range: tuple[float, float],
    target_height_range: tuple[float, float],
    gravity_z: float = -9.81,
    shuttle_mass: float = 0.00519,
    shuttle_drag_k: float = 0.00110,
    solver_steps: int = 32,
    solver_iterations: int = 5,
) -> None:
    """Reset using a batched shooting solve for vector quadratic drag."""

    shuttle: RigidObject = env.scene[asset_cfg.name]
    device = shuttle.data.root_pos_w.device
    dtype = shuttle.data.root_pos_w.dtype

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    else:
        env_ids = env_ids.to(device=device)

    num_resets = len(env_ids)
    pool_key = (str(device), dtype, tuple(position_range.items()), flight_time_range,
                target_x_range, target_y_range, target_height_range,
                float(gravity_z), float(shuttle_mass), float(shuttle_drag_k),
                int(solver_steps), int(solver_iterations))
    cached = _SHUTTLE_RESET_POOL.get(pool_key)
    if cached is not None and cached[0].shape[0] >= num_resets:
        start_local, linear_velocity = cached
        indices = torch.randperm(start_local.shape[0], device=device)[:num_resets]
        start_local = start_local[indices]
        linear_velocity = linear_velocity[indices]
        world_position = start_local + env.scene.env_origins[env_ids]
        root_state = torch.zeros((num_resets, 13), device=device, dtype=dtype)
        root_state[:, :3] = world_position
        root_state[:, 3] = 1.0
        root_state[:, 7:10] = linear_velocity
        shuttle.write_root_state_to_sim(root_state, env_ids=env_ids)
        return

    num_resets = max(num_resets, _SHUTTLE_RESET_POOL_SIZE)

    def sample(value_range: tuple[float, float]) -> torch.Tensor:
        low, high = value_range
        return low + (high - low) * torch.rand(
            num_resets,
            device=device,
            dtype=dtype,
        )

    start_local = torch.stack(
        [
            sample(position_range["x"]),
            sample(position_range["y"]),
            sample(position_range["z"]),
        ],
        dim=-1,
    )
    env_origins = env.scene.env_origins[env_ids]
    target_local = torch.stack(
        [
            sample(target_x_range),
            sample(target_y_range),
            sample(target_height_range),
        ],
        dim=-1,
    )

    flight_time = sample(flight_time_range).unsqueeze(-1)

    delta_xy = target_local[:, :2] - start_local[:, :2]
    horizontal_distance = torch.linalg.norm(delta_xy, dim=-1)
    horizontal_direction = delta_xy / torch.clamp(horizontal_distance.unsqueeze(-1), min=1e-8)
    duration = flight_time.squeeze(-1)
    drag = float(shuttle_drag_k) / float(shuttle_mass)
    steps = max(int(solver_steps), 1)
    def rollout(vh, vz):
        # Keep caller-owned Newton iterates immutable; rollout is evaluated
        # repeatedly at the same baseline and finite-difference perturbations.
        vh = vh.clone()
        vz = vz.clone()
        h = torch.zeros_like(vh); z = start_local[:, 2].clone(); dt = duration / steps
        for _ in range(steps):
            def f(u, w):
                speed = torch.sqrt(u * u + w * w)
                return u, w, -drag * speed * u, float(gravity_z) - drag * speed * w
            k1 = f(vh, vz)
            k2 = f(vh + 0.5 * dt * k1[2], vz + 0.5 * dt * k1[3])
            k3 = f(vh + 0.5 * dt * k2[2], vz + 0.5 * dt * k2[3])
            k4 = f(vh + dt * k3[2], vz + dt * k3[3])
            h += dt * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6.0
            z += dt * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6.0
            vh += dt * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6.0
            vz += dt * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6.0
        return h, z
    vh = torch.full((num_resets,), 10.0, device=device, dtype=dtype)
    vz = torch.full((num_resets,), 8.0, device=device, dtype=dtype)
    for _ in range(max(int(solver_iterations), 1)):
        h0, z0 = rollout(vh, vz); hh, zh = rollout(vh + 1e-2, vz); hv, zv = rollout(vh, vz + 1e-2)
        j00, j10 = (hh-h0)/1e-2, (zh-z0)/1e-2; j01, j11 = (hv-h0)/1e-2, (zv-z0)/1e-2
        rh, rz = h0-horizontal_distance, z0-target_local[:, 2]; det = j00*j11-j01*j10
        det = torch.where(det.abs() < 1e-8, torch.where(det >= 0, 1e-8, -1e-8), det)
        vh -= ((j11*rh-j01*rz)/det).clamp(-5.0, 5.0); vz -= ((-j10*rh+j00*rz)/det).clamp(-5.0, 5.0)
    velocity_xy = horizontal_direction * vh.unsqueeze(-1)
    velocity_xy = torch.where(horizontal_distance.unsqueeze(-1) > 1e-8, velocity_xy, 0.0)
    linear_velocity = torch.cat((velocity_xy, vz.unsqueeze(-1)), dim=-1)
    _SHUTTLE_RESET_POOL[pool_key] = (start_local.detach(), linear_velocity.detach())
    if len(env_ids) < num_resets:
        indices = torch.randperm(num_resets, device=device)[:len(env_ids)]
        start_local = start_local[indices]
        linear_velocity = linear_velocity[indices]
        env_origins = env.scene.env_origins[env_ids]
        num_resets = len(env_ids)
    world_position = start_local + env_origins
    orientation = torch.zeros(
        (num_resets, 4),
        device=device,
        dtype=dtype,
    )
    orientation[:, 0] = 1.0
    angular_velocity = torch.zeros(
        (num_resets, 3),
        device=device,
        dtype=dtype,
    )
    root_state = torch.cat(
        [
            world_position,
            orientation,
            linear_velocity,
            angular_velocity,
        ],
        dim=-1,
    )
    shuttle.write_root_state_to_sim(root_state, env_ids=env_ids)
