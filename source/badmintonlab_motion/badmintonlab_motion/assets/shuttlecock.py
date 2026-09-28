"""Visual shuttlecock support with the existing spherical physics proxy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.sim.spawners.shapes import shapes
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply


SHUTTLECOCK_ASSET_DIR = Path(__file__).resolve().parent / "shuttlecock"
SHUTTLECOCK_VISUAL_USD = str(SHUTTLECOCK_ASSET_DIR / "shuttlecock_visual.usd")
# The imported FBX origin is on the skirt end. Its cork center is 6.5 cm in
# local +Y, so this translation makes the sphere proxy's origin the cork center.
SHUTTLECOCK_VISUAL_TRANSLATION = (0.0056, -0.0650, 0.0046)
SHUTTLECOCK_CORK_AXIS_B = (0.0, 1.0, 0.0)


@clone
def spawn_shuttlecock(
    prim_path: str,
    cfg: ShuttlecockSphereCfg,
    translation: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    **kwargs,
):
    """Spawn the existing sphere proxy plus an FBX-derived visual child."""
    sphere = shapes.spawn_sphere(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    prim_utils.create_prim(
        prim_path=f"{prim_path}/visual",
        usd_path=cfg.visual_usd_path,
        translation=cfg.visual_translation,
    )
    return sphere


@configclass
class ShuttlecockSphereCfg(sim_utils.SphereCfg):
    """Sphere collider with a no-collision shuttlecock visual child."""

    func: Callable = spawn_shuttlecock
    visual_usd_path: str = SHUTTLECOCK_VISUAL_USD
    visual_translation: tuple[float, float, float] = SHUTTLECOCK_VISUAL_TRANSLATION


def alignment_torque(
    root_quat_w: torch.Tensor,
    linear_velocity_w: torch.Tensor,
    angular_velocity_w: torch.Tensor,
    gain: float,
    damping: float,
    speed_threshold: float,
    max_torque: float,
) -> torch.Tensor:
    """Return a bounded torque that smoothly turns the cork toward velocity.

    This affects only the rigid body's orientation. Translational drag remains
    the existing isotropic quadratic model.
    """
    speed = torch.linalg.norm(linear_velocity_w, dim=-1, keepdim=True)
    desired_axis_w = linear_velocity_w / speed.clamp_min(1.0e-6)
    local_axis = torch.tensor(
        SHUTTLECOCK_CORK_AXIS_B, device=root_quat_w.device, dtype=root_quat_w.dtype
    ).expand_as(linear_velocity_w)
    current_axis_w = quat_apply(root_quat_w, local_axis)
    turn_axis = torch.cross(current_axis_w, desired_axis_w, dim=-1)

    # A perfect 180-degree reversal has zero cross product. Pick a stable
    # perpendicular turn direction so a hard racket reversal can recover.
    opposite = (torch.sum(current_axis_w * desired_axis_w, dim=-1, keepdim=True) < -0.999)
    world_x = torch.tensor((1.0, 0.0, 0.0), device=root_quat_w.device, dtype=root_quat_w.dtype).expand_as(
        current_axis_w
    )
    world_y = torch.tensor((0.0, 1.0, 0.0), device=root_quat_w.device, dtype=root_quat_w.dtype).expand_as(
        current_axis_w
    )
    fallback = torch.cross(current_axis_w, world_x, dim=-1)
    alternate = torch.cross(current_axis_w, world_y, dim=-1)
    fallback = torch.where(
        torch.linalg.norm(fallback, dim=-1, keepdim=True) > 1.0e-4, fallback, alternate
    )
    fallback = fallback / torch.linalg.norm(fallback, dim=-1, keepdim=True).clamp_min(1.0e-6)
    turn_axis = torch.where(opposite, fallback, turn_axis)

    torque = float(gain) * speed.square() * turn_axis - float(damping) * angular_velocity_w
    torque = torch.where(speed >= float(speed_threshold), torque, torch.zeros_like(torque))
    magnitude = torch.linalg.norm(torque, dim=-1, keepdim=True)
    return torque * (float(max_torque) / magnitude.clamp_min(float(max_torque))).clamp(max=1.0)
