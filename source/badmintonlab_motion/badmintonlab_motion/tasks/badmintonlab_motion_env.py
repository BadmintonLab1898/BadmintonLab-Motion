# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor

from badmintonlab_motion.assets.court import is_net_fault, is_out_of_bounds
from badmintonlab_motion.assets.shuttlecock import alignment_torque

from .amp_motion import (
    AMP_TRANSITION_STEPS,
    T1_DOF_NAMES,
    T2_DOF_NAMES,
    build_amp_state,
    get_amp_state_dim,
)

if TYPE_CHECKING:
    from .badmintonlab_motion_env_cfg import BadmintonLabMotionEnvCfg


class BadmintonLabMotionEnv(ManagerBasedRLEnv):
    """Badminton environment with shuttle dynamics and AMP state transitions.

    Sweet-spot pose / linear / angular state is read directly from the
    ``racket_sweet_spot`` dummy body — no manual offset, ``quat_apply`` or
    cross product. Contact detection still uses the ``racket_link`` contact
    sensor for stability (the sweet-spot dummy has no collision shape).

    Pre-hit the shuttle is held static (gravity disabled in spawn cfg). On the
    first valid hit the natural PhysX collision provides the initial shuttle
    velocity; every physics sub-step thereafter we inject world-frame gravity
    and a quadratic drag force ``F_drag = -k |v| v``.
    """

    cfg: "BadmintonLabMotionEnvCfg"

    def __init__(self, cfg: "BadmintonLabMotionEnvCfg", render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        device = self.device
        N = self.num_envs

        self.has_hit = torch.zeros(N, dtype=torch.bool, device=device)
        self.hit_step = torch.zeros(N, dtype=torch.long, device=device)
        self.post_hit_steps = torch.zeros(N, dtype=torch.long, device=device)
        self.hit_pos = torch.zeros((N, 3), device=device)
        self.hit_root_pos = torch.zeros((N, 3), device=device)
        self.hit_sweetspot_shuttle_distance = torch.zeros(N, device=device)
        # The interval-approach reward is an episode event, rather than a
        # dense per-step distance reward. Keep the closest distance observed
        # in the configured incoming height window and emit it exactly once.
        self.approach_min_distance = torch.full((N,), torch.inf, device=device)
        self.approach_event_distance = torch.zeros(N, device=device)
        self.approach_reward_event = torch.zeros(N, dtype=torch.bool, device=device)
        self.approach_reward_pending = torch.ones(N, dtype=torch.bool, device=device)

        # Once the shuttle first reaches the floor, keep its pose fixed. This
        # suppresses the rigid sphere's non-physical post-landing bounce while
        # preserving its restitution for racket contact.
        self.shuttle_landed = torch.zeros(N, dtype=torch.bool, device=device)
        self._landed_shuttle_state_w = torch.zeros((N, 13), device=device)

        # shuttle world-frame position one env-step ago (for net-crossing logic)
        self.prev_shuttle_pos_w = torch.zeros((N, 3), device=device)
        self.curr_shuttle_pos_w = torch.zeros((N, 3), device=device)
        # latched per-step fault flags consumed by reward / termination terms
        self.net_fault = torch.zeros(N, dtype=torch.bool, device=device)
        self.out_fault = torch.zeros(N, dtype=torch.bool, device=device)
        # whether a fault has already terminated this episode (suppresses re-firing)
        self._fault_latched = torch.zeros(N, dtype=torch.bool, device=device)
        self._prev_pos_valid = torch.zeros(N, dtype=torch.bool, device=device)

        self._shuttle: RigidObject = self.scene["shuttle"]
        self._robot: Articulation = self.scene["robot"]
        self._racket_contact: ContactSensor = self.scene.sensors["racket_contact"]
        self._sweet_spot_body_id = self._robot.find_bodies("racket_sweet_spot")[0][0]

        # self._court = self.scene["court"]

        self._drag_k = float(cfg.shuttle_drag_k)
        self._mass = float(cfg.shuttle_mass)
        self._g_vec = torch.tensor([0.0, 0.0, float(cfg.gravity_z)], device=device)
        self._shuttle_ground_height = float(cfg.shuttle_ground_height)
        self._shuttle_landing_detection_height = float(cfg.shuttle_landing_detection_height)

        self._hit_contact_thresh = float(cfg.hit_contact_threshold)

        self._amp_joint_ids = None
        if cfg.enable_amp:
            requested_names = T2_DOF_NAMES if self._robot.joint_names[0].islower() else T1_DOF_NAMES
            self._amp_joint_ids, amp_joint_names = self._robot.find_joints(list(requested_names), preserve_order=True)
            if tuple(amp_joint_names) != tuple(requested_names):
                raise RuntimeError(
                    "AMP joint order does not match the reference motion schema: "
                    f"expected {requested_names}, got {tuple(amp_joint_names)}"
                )
            self._amp_state_dim = get_amp_state_dim(len(requested_names))
            self._amp_state_history = torch.zeros(
                (self.num_envs, AMP_TRANSITION_STEPS, self._amp_state_dim), device=self.device
            )

    def step(self, action: torch.Tensor):
        # Mirror ManagerBasedRLEnv.step but inject shuttle external forces inside the
        # decimation loop and update hit state at the end.
        self.action_manager.process_action(action.to(self.device))
        self._pre_step_hook()
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()

            self._apply_shuttle_forces()
            previous_shuttle_pos_w = self._shuttle.data.root_pos_w.clone()
            previous_shuttle_vel_w = self._shuttle.data.root_lin_vel_w.clone()

            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self._update_shuttle_landing_state(previous_shuttle_pos_w, previous_shuttle_vel_w)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._update_hit_state()
        self._update_fault_state()
        self._update_approach_state()
        self._post_physics_step_hook()

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        # print(self.reward_buf, self.has_hit)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _pre_step_hook(self) -> None:
        self.approach_reward_event.zero_()
        if self.cfg.enable_amp:
            self._amp_state_history = torch.roll(self._amp_state_history, shifts=-1, dims=1)
            self._amp_state_history[:, -1] = self._compute_amp_state()

    def _post_physics_step_hook(self) -> None:
        if not self.cfg.enable_amp:
            return
        self._amp_state_history = torch.roll(self._amp_state_history, shifts=-1, dims=1)
        self._amp_state_history[:, -1] = self._compute_amp_state()
        amp_transition = self._amp_state_history.flatten(1)
        if amp_transition.shape[-1] != AMP_TRANSITION_STEPS * self._amp_state_dim:
            raise RuntimeError(f"Invalid AMP transition shape: {tuple(amp_transition.shape)}")
        # Capture terminal poses before the environment resets them.
        self.extras["amp_transition"] = amp_transition

    def _compute_amp_state(self) -> torch.Tensor:
        if self._amp_joint_ids is None:
            raise RuntimeError("AMP state requested while cfg.enable_amp is false")
        return build_amp_state(
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.joint_pos[:, self._amp_joint_ids],
            self._robot.data.joint_vel[:, self._amp_joint_ids],
        )

    def _apply_shuttle_forces(self) -> None:
        v = self._shuttle.data.root_lin_vel_w
        speed = torch.linalg.norm(v, dim=-1, keepdim=True)
        drag_force = -self._drag_k * speed * v
        grav_force = self._mass * self._g_vec
        total = drag_force + grav_force
        total[self.shuttle_landed] = 0.0

        forces = total.unsqueeze(1)
        torques = alignment_torque(
            self._shuttle.data.root_quat_w,
            v,
            self._shuttle.data.root_ang_vel_w,
            self.cfg.shuttle_orientation_gain,
            self.cfg.shuttle_orientation_damping,
            self.cfg.shuttle_orientation_speed_threshold,
            self.cfg.shuttle_orientation_max_torque,
        ).unsqueeze(1)
        torques[self.shuttle_landed] = 0.0
        self._shuttle.set_external_force_and_torque(forces, torques, is_global=True)

        # print(mask, "pos", self._shuttle.data.root_pos_w[0].tolist(), "speed", v.tolist(), "drag", drag_force.tolist(), "grav", grav_force.tolist())

    def _update_shuttle_landing_state(
        self,
        previous_pos_w: torch.Tensor,
        previous_vel_w: torch.Tensor,
    ) -> None:
        """Latch the first downward floor contact and freeze the shuttle there."""
        env_origins = self.scene.env_origins
        current_pos_w = self._shuttle.data.root_pos_w
        previous_z = previous_pos_w[:, 2] - env_origins[:, 2]
        current_z = current_pos_w[:, 2] - env_origins[:, 2]

        just_landed = (
            (~self.shuttle_landed)
            & (previous_vel_w[:, 2] < 0.0)
            & (current_z <= self._shuttle_landing_detection_height)
        )
        if just_landed.any():
            root_state = self._shuttle.data.root_state_w[just_landed].clone()

            # Interpolate the final airborne segment to the resting height so
            # horizontal placement does not depend on solver penetration.
            dz = current_z[just_landed] - previous_z[just_landed]
            interpolation = torch.where(
                dz.abs() > 1e-6,
                (self._shuttle_ground_height - previous_z[just_landed]) / dz,
                torch.ones_like(dz),
            ).clamp(0.0, 1.0)
            root_state[:, :2] = previous_pos_w[just_landed, :2] + interpolation.unsqueeze(-1) * (
                current_pos_w[just_landed, :2] - previous_pos_w[just_landed, :2]
            )
            root_state[:, 2] = env_origins[just_landed, 2] + self._shuttle_ground_height
            root_state[:, 7:] = 0.0

            self._landed_shuttle_state_w[just_landed] = root_state
            self.shuttle_landed[just_landed] = True

        landed_ids = self.shuttle_landed.nonzero(as_tuple=False).squeeze(-1)
        if len(landed_ids) > 0:
            self._shuttle.write_root_state_to_sim(
                self._landed_shuttle_state_w[landed_ids],
                env_ids=landed_ids,
            )

    def _update_hit_state(self) -> None:
        force_hist = self._racket_contact.data.force_matrix_w_history
        contact_force = torch.norm(force_hist, dim=-1).amax(dim=1).amax(dim=-1).amax(dim=-1)
        contact_mask = contact_force > self._hit_contact_thresh
        first_hit = contact_mask & (~self.has_hit)

        if first_hit.any():
            self.hit_step[first_hit] = self.episode_length_buf[first_hit]
            self.hit_pos[first_hit] = self._shuttle.data.root_pos_w[first_hit]
            self.hit_root_pos[first_hit] = self._robot.data.root_pos_w[first_hit]
            sweet_pos_w = self._robot.data.body_pos_w[:, self._sweet_spot_body_id, :]
            self.hit_sweetspot_shuttle_distance[first_hit] = torch.linalg.norm(
                self._shuttle.data.root_pos_w[first_hit] - sweet_pos_w[first_hit],
                dim=-1,
            )

        self.has_hit = self.has_hit | first_hit
        self.post_hit_steps = self.post_hit_steps + self.has_hit.long()

    def _update_approach_state(self) -> None:
        """Track the closest racket distance during the incoming height interval."""
        params = self.cfg.rewards.interval_approach.params
        min_height = float(params["min_height"])
        max_height = float(params["max_height"])
        incoming_speed_epsilon = max(float(params["incoming_speed_epsilon"]), 0.0)

        local = self.curr_shuttle_pos_w - self.scene.env_origins
        previous_local = self.prev_shuttle_pos_w - self.scene.env_origins
        descending = self._shuttle.data.root_lin_vel_w[:, 2] < -incoming_speed_epsilon
        in_robot_half = local[:, 0] < 0.0
        in_window = (local[:, 2] >= min_height) & (local[:, 2] <= max_height)
        tracking = (
            self._prev_pos_valid
            & self.approach_reward_pending
            & descending
            & in_robot_half
            & in_window
            & (~self.has_hit)
        )

        sweet_pos = self._robot.data.body_pos_w[:, self._sweet_spot_body_id]
        distance = torch.linalg.norm(self._shuttle.data.root_pos_w - sweet_pos, dim=-1)
        self.approach_min_distance = torch.where(
            tracking,
            torch.minimum(self.approach_min_distance, distance),
            self.approach_min_distance,
        )

        crossed_interval = (previous_local[:, 2] >= max_height) & (local[:, 2] < min_height)
        missed = (
            self._prev_pos_valid
            & self.approach_reward_pending
            & descending
            & in_robot_half
            & (previous_local[:, 2] >= min_height)
            & (local[:, 2] < min_height)
            & ((self.approach_min_distance < torch.inf) | crossed_interval)
        )
        completed = self.approach_reward_pending & (self.has_hit | missed)
        event_distance = torch.where(
            self.approach_min_distance < torch.inf,
            self.approach_min_distance,
            distance,
        )
        self.approach_event_distance[completed] = event_distance[completed]
        self.approach_reward_event[completed] = True
        self.approach_reward_pending[completed] = False
        self.approach_min_distance[completed] = torch.inf

    def _update_fault_state(self) -> None:
        """Detect net / out-of-bounds faults using shuttle motion across a step.

        Operates in env-local coordinates (world position minus env origin) so
        the same court geometry is reused per env. Only fires for envs that
        had a valid previous position recorded (i.e. not the first step after
        a reset) and that have not already latched a fault this episode.
        """
        env_origins = self.scene.env_origins
        self.prev_shuttle_pos_w = self.curr_shuttle_pos_w.clone()
        self.curr_shuttle_pos_w = self._shuttle.data.root_pos_w.clone()

        curr_local = self.curr_shuttle_pos_w - env_origins
        prev_local = self.prev_shuttle_pos_w - env_origins

        active = self._prev_pos_valid & (~self._fault_latched)

        net_f = is_net_fault(prev_local, curr_local, self.cfg.court) & active
        # only count out-of-bounds after the shuttle has been struck — otherwise
        # the spawn / pre-hit state can register as out by being held above the
        # court boundary
        out_f = is_out_of_bounds(curr_local, self.cfg.court) & active & self.has_hit

        self.net_fault = net_f
        self.out_fault = out_f
        self._fault_latched = self._fault_latched | net_f | out_f
        # after this step every env has a valid prev/curr pair; _reset_idx clears
        # the flag again for envs that get reset at the end of this step.
        self._prev_pos_valid.fill_(True)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        hit_success_rate = torch.mean(self.has_hit[env_ids].float())
        self.extras["log"]["Metrics/Hit/success_rate"] = hit_success_rate

        hit_mask = self.has_hit[env_ids]
        if torch.any(hit_mask):
            hit_distance = torch.mean(self.hit_sweetspot_shuttle_distance[env_ids][hit_mask])
        else:
            hit_distance = torch.tensor(0.0, device=self.device)
        self.extras["log"]["Metrics/Hit/sweetspot_shuttle_distance"] = hit_distance

        self.has_hit[env_ids] = False
        self.hit_step[env_ids] = 0
        self.post_hit_steps[env_ids] = 0
        self.hit_pos[env_ids] = 0.0
        self.hit_root_pos[env_ids] = 0.0
        self.hit_sweetspot_shuttle_distance[env_ids] = 0.0
        self.approach_min_distance[env_ids] = torch.inf
        self.approach_event_distance[env_ids] = 0.0
        self.approach_reward_event[env_ids] = False
        self.approach_reward_pending[env_ids] = True
        self.shuttle_landed[env_ids] = False
        self._landed_shuttle_state_w[env_ids] = 0.0
        self.net_fault[env_ids] = False
        self.out_fault[env_ids] = False
        self._fault_latched[env_ids] = False
        self._prev_pos_valid[env_ids] = False
        # re-seed prev/curr to current shuttle pose so the next step doesn't see
        # an artificial jump across the net
        new_pos = self._shuttle.data.root_pos_w[env_ids]
        self.prev_shuttle_pos_w[env_ids] = new_pos
        self.curr_shuttle_pos_w[env_ids] = new_pos

        if self.cfg.enable_amp and hasattr(self, "_amp_state_history"):
            initial_amp_state = self._compute_amp_state()[env_ids]
            self._amp_state_history[env_ids] = initial_amp_state.unsqueeze(1)
