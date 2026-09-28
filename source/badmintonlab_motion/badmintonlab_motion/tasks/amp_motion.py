"""Reference-motion features used by the badminton AMP discriminator."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


T1_DOF_NAMES = (
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)
T2_DOF_NAMES = (
    "aa_head_yaw_joint", "head_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "left_wrist_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_pitch_joint",
    "right_elbow_yaw_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint",
    "waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_hip_yaw_joint", "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_pitch_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
AMP_TRANSITION_STEPS = 4


def get_amp_state_dim(num_joints: int) -> int:
    """Root linear/angular velocity plus every joint's position and velocity."""
    if num_joints not in (len(T1_DOF_NAMES), len(T2_DOF_NAMES)):
        raise ValueError(f"Expected {len(T1_DOF_NAMES)} (T1) or {len(T2_DOF_NAMES)} (T2) joints, got {num_joints}")
    return 3 + 3 + 2 * num_joints


def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quaternion_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def _quaternion_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Multiply scalar-first (wxyz) quaternions."""
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quaternion_rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    vector_quaternion = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return _quaternion_multiply(
        _quaternion_multiply(_quaternion_conjugate(quaternion), vector_quaternion), quaternion
    )[..., 1:]


def _make_quaternion_sequence_continuous(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    for frame in range(1, result.shape[0]):
        if torch.dot(result[frame - 1], result[frame]) < 0:
            result[frame] = -result[frame]
    return result


def _finite_difference(values: torch.Tensor, dt: float) -> torch.Tensor:
    if values.shape[0] < 2:
        raise ValueError("A reference motion needs at least two frames")
    velocity = torch.empty_like(values)
    velocity[0] = (values[1] - values[0]) / dt
    velocity[-1] = (values[-1] - values[-2]) / dt
    if values.shape[0] > 2:
        velocity[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    return velocity


def _body_angular_velocity(quaternion: torch.Tensor, dt: float) -> torch.Tensor:
    relative = _quaternion_multiply(_quaternion_conjugate(quaternion[:-1]), quaternion[1:])
    relative = _normalize_quaternion(relative)
    relative = torch.where(relative[:, :1] < 0, -relative, relative)
    sin_half_angle = torch.linalg.vector_norm(relative[:, 1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half_angle, relative[:, :1].clamp_min(1.0e-8))
    scale = torch.where(sin_half_angle > 1.0e-7, angle / sin_half_angle, torch.full_like(angle, 2.0))
    pair_velocity = relative[:, 1:] * scale / dt
    return torch.cat((pair_velocity, pair_velocity[-1:]), dim=0)


def build_amp_state(
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
) -> torch.Tensor:
    """Build the robot-only state shared by simulation and reference clips."""
    state_dim = get_amp_state_dim(joint_pos.shape[-1])
    if joint_vel.shape != joint_pos.shape:
        raise ValueError("Joint position and velocity shapes must match")
    root_shape = (*joint_pos.shape[:-1], 3)
    if root_lin_vel_b.shape != root_shape or root_ang_vel_b.shape != root_shape:
        raise ValueError(f"Root velocities must have shape {root_shape}")
    state = torch.cat(
        (root_lin_vel_b, root_ang_vel_b, joint_pos, joint_vel), dim=-1
    )
    if state.shape[-1] != state_dim:
        raise ValueError(f"Expected {state_dim} AMP state features, got {state.shape[-1]}")
    return state


def reference_clip_to_amp_states(
    base_pos: torch.Tensor,
    root_rot_xyzw: torch.Tensor,
    dof_pos: torch.Tensor,
    fps: float,
) -> torch.Tensor:
    """Convert a T1 or T2 reference clip into the simulator's AMP feature convention."""
    frames = base_pos.shape[0]
    if base_pos.shape != (frames, 3):
        raise ValueError(f"base_pos must have shape [frames, 3], got {tuple(base_pos.shape)}")
    if root_rot_xyzw.shape != (frames, 4):
        raise ValueError(f"root_rot must have shape [frames, 4], got {tuple(root_rot_xyzw.shape)}")
    if dof_pos.shape not in ((frames, len(T1_DOF_NAMES)), (frames, len(T2_DOF_NAMES))):
        raise ValueError(
            f"dof_pos must have shape [{frames}, {len(T1_DOF_NAMES)}] (T1) or "
            f"[{frames}, {len(T2_DOF_NAMES)}] (T2), got {tuple(dof_pos.shape)}"
        )
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not (torch.isfinite(base_pos).all() and torch.isfinite(root_rot_xyzw).all() and torch.isfinite(dof_pos).all()):
        raise ValueError("Reference motion contains non-finite values")

    # Reference files use scipy xyzw; Isaac Lab and the helpers below use wxyz.
    root_quat_wxyz = root_rot_xyzw[:, (3, 0, 1, 2)]
    root_quat_wxyz = _make_quaternion_sequence_continuous(_normalize_quaternion(root_quat_wxyz))
    dt = 1.0 / float(fps)
    root_lin_vel_w = _finite_difference(base_pos, dt)

    return build_amp_state(
        _quaternion_rotate_inverse(root_quat_wxyz, root_lin_vel_w),
        _body_angular_velocity(root_quat_wxyz, dt),
        dof_pos,
        _finite_difference(dof_pos, dt),
    )


def resolve_motion_directory(configured_path: str | Path) -> Path:
    environment_path = os.environ.get("BADMINTONLAB_MOTION_DIR")
    requested = Path(environment_path if environment_path else configured_path).expanduser()
    candidates = [requested] if requested.is_absolute() else [Path.cwd() / requested]
    if not requested.is_absolute():
        candidates.append(Path(__file__).resolve().parent / requested)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    attempted = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"AMP motion directory not found. Tried: {attempted}")


class ReferenceMotionDataset:
    """In-memory sampler of consecutive expert-state sequences."""

    def __init__(self, motion_directory: str | Path, file_pattern: str, device: str | torch.device) -> None:
        self.motion_directory = resolve_motion_directory(motion_directory)
        self.files = sorted(self.motion_directory.glob(file_pattern))
        if not self.files:
            raise FileNotFoundError(f"No AMP motion files match {file_pattern!r} in {self.motion_directory}")

        transitions = []
        self.clip_lengths = []
        for motion_file in self.files:
            with np.load(motion_file, allow_pickle=False) as data:
                missing = {"fps", "dof_pos", "base_pos", "root_rot"}.difference(data.files)
                if missing:
                    raise ValueError(f"{motion_file.name} is missing fields: {sorted(missing)}")
                states = reference_clip_to_amp_states(
                    torch.as_tensor(data["base_pos"], dtype=torch.float32),
                    torch.as_tensor(data["root_rot"], dtype=torch.float32),
                    torch.as_tensor(data["dof_pos"], dtype=torch.float32),
                    float(data["fps"]),
                )
            if not transitions:
                self.state_dim = states.shape[-1]
                self.transition_dim = AMP_TRANSITION_STEPS * self.state_dim
            elif states.shape[-1] != self.state_dim:
                raise ValueError(
                    f"Mixed robot AMP dimensions in {self.motion_directory}: "
                    f"{motion_file.name} has {states.shape[-1]} features, expected {self.state_dim}"
                )
            transitions.append(torch.stack([states[i : i + AMP_TRANSITION_STEPS] for i in range(states.shape[0] - AMP_TRANSITION_STEPS + 1)], dim=0).flatten(1))
            self.clip_lengths.append(states.shape[0])

        self.clip_transitions = [transition.to(device=device) for transition in transitions]
        self.transitions = torch.cat(self.clip_transitions, dim=0)
        self.feature_mean = self.transitions.mean(dim=0)
        self.feature_std = self.transitions.std(dim=0).clamp_min(0.05)
        if self.transitions.shape[-1] != self.transition_dim or not torch.isfinite(self.transitions).all():
            raise ValueError("Preprocessed AMP transitions are invalid")

    def normalize(self, transitions: torch.Tensor) -> torch.Tensor:
        return (transitions - self.feature_mean) / self.feature_std

    def sample(self, num_samples: int) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        indices = torch.randint(self.transitions.shape[0], (num_samples,), device=self.transitions.device)
        return self.transitions[indices]
