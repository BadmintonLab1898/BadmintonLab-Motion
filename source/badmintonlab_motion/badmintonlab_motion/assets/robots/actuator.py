from __future__ import annotations
from dataclasses import MISSING

import torch
from collections.abc import Sequence

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg, ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions

RPM2RADPS = 2 * torch.pi / 60

class DelayedImplicitActuator(ImplicitActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedImplicitActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""


class BadmintonDelayedPDActuator(DelayedPDActuator):
    """Delayed PD actuator with speed-dependent torque clipping.

    This implements a piecewise-linear torque-speed curve (T-N curve) similar to
    :class:`DelayedImplicitActuator`, but for a PD actuator with delayed commands.
    """

    cfg: BadmintonDelayedPDActuatorCfg

    def __init__(self, cfg: "BadmintonDelayedPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # Knee speed for the torque-speed curve. Defaults to velocity_limit (i.e. no reduction).
        self.knee_point_velocity = self._parse_joint_parameter(cfg.knee_point_velocity, self.velocity_limit)
        self.knee_point_velocity = torch.clamp(self.knee_point_velocity, min=0.0)
        self.knee_point_velocity = torch.minimum(self.knee_point_velocity, self.velocity_limit)
        # buffer used for speed-based clipping
        self._joint_vel = torch.zeros_like(self.computed_effort)
        v_knee = self.knee_point_velocity
        v_max = self.velocity_limit
        self._denom = (v_max - v_knee).clamp(min=1e-6)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # Cache measured joint velocity for speed-based torque clipping.
        self._joint_vel[:] = joint_vel
        # Apply delay using the base implementation.
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # Piecewise-linear T-N curve using existing limits:
        # - effort_limit: max torque for |v| <= knee_point_velocity
        # - velocity_limit: torque goes to 0 at |v| == velocity_limit
        joint_vel_abs = self._joint_vel.abs()
        v_max = self.velocity_limit
        tau_max = self.effort_limit

        # If velocity limit is not finite or non-positive, fall back to box limit (or zero).
        non_positive_vmax = v_max <= 0.0
        non_finite_vmax = ~torch.isfinite(v_max)

        # Avoid division-by-zero when v_max == v_knee for some joints.
        tau_linear = tau_max * (v_max - joint_vel_abs) / self._denom
        max_effort = tau_linear.clamp(min=0.0).clamp(max=tau_max)

        max_effort = torch.where(non_finite_vmax, tau_max, max_effort)
        max_effort = torch.where(non_positive_vmax, torch.zeros_like(max_effort), max_effort)

        return torch.clip(effort, min=-max_effort, max=max_effort)


@configclass
class BadmintonJointCfg:
    """Configuration for robot joint models."""

    joint_model_name: str = MISSING

    effort_limit: float = MISSING
    velocity_limit: float = MISSING
    knee_point_velocity: float = MISSING
    armature: float = MISSING

    stiffness: float = None
    damping: float = None

    natural_freq: float = 10.     # 10Hz
    '''
    Natural frequency (Hz) for computing default stiffness and damping.
    '''
    damping_ratio: float = 2.0
    '''
    Damping ratio for computing default damping.
    '''

    def __post_init__(self):
        if self.stiffness is None:
            self.stiffness = self.armature * (2 * 3.1415926535 * self.natural_freq)**2
        if self.damping is None:
            self.damping = 2 * self.damping_ratio * self.armature * (2 * 3.1415926535 * self.natural_freq)


@configclass
class BadmintonDelayedActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`BadmintonDelayedPDActuator`."""

    class_type: type = MISSING

    knee_point_velocity: dict[str, float] | float | None = None
    """Knee speed for the torque-speed curve.

    If None, defaults to :attr:`velocity_limit` (no speed-dependent reduction).
    """

    stiffness: dict[str, float] | float | None = None
    damping: dict[str, float] | float | None = None

    joint_cfgs: dict[str, BadmintonJointCfg] | BadmintonJointCfg | None = None

    def __post_init__(self):
        if self.joint_cfgs is not None:
            if isinstance(self.joint_cfgs, BadmintonJointCfg):
                self.effort_limit_sim = self.joint_cfgs.effort_limit
                self.velocity_limit_sim = self.joint_cfgs.velocity_limit
                self.knee_point_velocity = self.joint_cfgs.knee_point_velocity
                self.armature = self.joint_cfgs.armature
                if self.stiffness is None:
                    self.stiffness = self.joint_cfgs.stiffness
                if self.damping is None:
                    self.damping = self.joint_cfgs.damping

            elif isinstance(self.joint_cfgs, dict):
                self.effort_limit_sim = {
                    joint_name: joint_cfg.effort_limit for joint_name, joint_cfg in self.joint_cfgs.items()
                }
                self.velocity_limit_sim = {
                    joint_name: joint_cfg.velocity_limit for joint_name, joint_cfg in self.joint_cfgs.items()
                }
                self.armature = {
                    joint_name: joint_cfg.armature for joint_name, joint_cfg in self.joint_cfgs.items()
                }
                self.knee_point_velocity = {
                    joint_name: joint_cfg.knee_point_velocity for joint_name, joint_cfg in self.joint_cfgs.items()
                }
                if self.stiffness is None:
                    self.stiffness = {
                        joint_name: joint_cfg.stiffness for joint_name, joint_cfg in self.joint_cfgs.items()
                    }
                if self.damping is None:
                    self.damping = {
                        joint_name: joint_cfg.damping for joint_name, joint_cfg in self.joint_cfgs.items()
                    }


@configclass
class BadmintonDelayedPDActuatorCfg(BadmintonDelayedActuatorCfg):
    """Configuration for :class:`BadmintonDelayedPDActuator`."""

    class_type: type = BadmintonDelayedPDActuator


@configclass
class BadmintonDelayedImplicitActuatorCfg(BadmintonDelayedActuatorCfg):
    """Configuration for :class:`BadmintonDelayedImplicitActuator`."""

    class_type: type = DelayedImplicitActuator


@configclass
class ParallelJointWrapperCfg(BadmintonJointCfg):
    """Configuration for a parallel joint wrapper that transforms a base joint
       configuration to parallel configuration.
    """

    joint_model_name: str = "ParallelJointWrapper"

    effort_ratio: tuple[float, float] = MISSING
    velocity_ratio: tuple[float, float] = MISSING
    armature_ratio: tuple[float, float] = MISSING
    knee_point_velocity_ratio: tuple[float, float] = (1.0, 1.0)

    base_joint_cfg: BadmintonJointCfg = MISSING
    serial_index: int = MISSING     # 0 for pitch, 1 for roll

    def __post_init__(self):
        self.effort_limit = self.effort_ratio[self.serial_index] * self.base_joint_cfg.effort_limit
        self.velocity_limit = self.velocity_ratio[self.serial_index] * self.base_joint_cfg.velocity_limit
        self.knee_point_velocity = self.knee_point_velocity_ratio[self.serial_index] * self.base_joint_cfg.knee_point_velocity
        self.armature = self.armature_ratio[self.serial_index] * self.base_joint_cfg.armature
        self.joint_model_name = f"{self.joint_model_name}({self.base_joint_cfg.joint_model_name})[{self.serial_index}]"
        super().__post_init__()

@configclass
class BadmintonT2WaistParaWrapperCfg(ParallelJointWrapperCfg):
    """Configuration for a parallel joint wrapper that transforms a base joint
       configuration to parallel configuration for T2 waist.
    """

    joint_model_name: str = "BadmintonT2WaistParaWrapper"

    effort_ratio: tuple[float, float] = (2.3, 3.5)
    velocity_ratio: tuple[float, float] = (0.7, 0.5)
    armature_ratio: tuple[float, float] = (3.237335, 6.473473)


@configclass
class BadmintonT2AnkleParaWrapperCfg(ParallelJointWrapperCfg):
    """Configuration for a parallel joint wrapper that transforms a base joint
       configuration to parallel configuration for T2 ankle.
    """

    joint_model_name: str = "BadmintonT2AnkleParaWrapper"

    effort_ratio: tuple[float, float] = (2.3, 1.6)
    velocity_ratio: tuple[float, float] = (0.8, 1.1)
    armature_ratio: tuple[float, float] = (2.974391, 1.491439)


@configclass
class BadmintonK1AnkleParaWrapperCfg(ParallelJointWrapperCfg):
    """Configuration for a parallel joint wrapper that transforms a base joint
       configuration to parallel configuration for K1 ankle.
    """

    joint_model_name: str = "BadmintonK1AnkleParaWrapper"

    effort_ratio: tuple[float, float] = (1.0, 1.0)
    velocity_ratio: tuple[float, float] = (1.0, 1.0)
    armature_ratio: tuple[float, float] = (2.0, 2.0)

@configclass
class BadmintonT1AnkleParaWrapperCfg(ParallelJointWrapperCfg):
    """Configuration for a parallel joint wrapper that transforms a base joint
       configuration to parallel configuration for T1 ankle.
    """

    joint_model_name: str = "BadmintonT1AnkleParaWrapper"

    effort_ratio: tuple[float, float] = (1.0, 1.0)
    velocity_ratio: tuple[float, float] = (1.0, 1.0)
    armature_ratio: tuple[float, float] = (2.0, 2.0)



@configclass
class BadmintonJointB7025_25(BadmintonJointCfg):
    """Configuration for B7025-25 robot joint model."""

    joint_model_name: str = "B7025-25"

    effort_limit: float = 107
    velocity_limit: float = 159 * RPM2RADPS
    knee_point_velocity: float = 35 * RPM2RADPS
    armature: float = 0.02781875

@configclass
class BadmintonJointB5025_25(BadmintonJointCfg):
    """Configuration for B5025-25 robot joint model."""

    joint_model_name: str = "B5025-25"

    effort_limit: float = 35
    velocity_limit: float = 135 * RPM2RADPS
    knee_point_velocity: float = 55 * RPM2RADPS
    armature: float = 0.0071

@configclass
class BadmintonJointH5036_02(BadmintonJointCfg):
    """Configuration for H5036-02 robot joint model."""

    joint_model_name: str = "H5036-02"

    effort_limit: float = 16
    velocity_limit: float = 145 * RPM2RADPS
    knee_point_velocity: float = 70 * RPM2RADPS
    armature: float = 0.0132192

@configclass
class BadmintonJointB7627_14(BadmintonJointCfg):
    """Configuration for B7627-14 robot joint model."""

    joint_model_name: str = "B7627-14"

    effort_limit: float = 65
    velocity_limit: float = 300 * RPM2RADPS
    knee_point_velocity: float = 175 * RPM2RADPS
    armature: float = 0.0136

@configclass
class BadmintonJointB7629_22(BadmintonJointCfg):
    """Configuration for B7629-22 robot joint model."""

    joint_model_name: str = "B7629-22"

    effort_limit: float = 125
    velocity_limit: float = 210 * RPM2RADPS
    knee_point_velocity: float = 55 * RPM2RADPS
    armature: float = 0.0281

@configclass
class BadmintonJointB7528_14(BadmintonJointCfg):
    """Configuration for B7528-14 robot joint model."""

    joint_model_name: str = "B7528-14"

    effort_limit: float = 75
    velocity_limit: float = 28.80
    knee_point_velocity: float = 8.38
    armature: float = 0.0135


@configclass
class BadmintonJointB7528_22(BadmintonJointCfg):
    """Configuration for B7528-22 robot joint model."""

    joint_model_name: str = "B7528-22"

    effort_limit: float = 130
    velocity_limit: float = 18.85
    knee_point_velocity: float = 1.88
    armature: float = 0.028


@configclass
class BadmintonJointR120(BadmintonJointCfg):
    """Configuration for R120 robot joint model."""

    joint_model_name: str = "R120"

    effort_limit: float = 120
    velocity_limit: float = 20.94
    knee_point_velocity: float = 10.47
    armature: float = 0.0399978


@configclass
class UnitreeJoint5020(BadmintonJointCfg):
    """Configuration for Unitree-5020 robot joint model."""

    joint_model_name: str = "Unitree-5020"

    effort_limit: float = 25
    velocity_limit: float = 37.
    knee_point_velocity: float = 37.
    armature: float = 0.003609725


@configclass
class UnitreeJoint7520_22(BadmintonJointCfg):
    """Configuration for Unitree-7520-22 robot joint model."""

    joint_model_name: str = "Unitree-7520-22"

    effort_limit: float = 139.
    velocity_limit: float = 20.
    knee_point_velocity: float = 20.
    armature: float = 0.025101925


@configclass
class UnitreeJoint7520_14(BadmintonJointCfg):
    """Configuration for Unitree-7520-14 robot joint model."""

    joint_model_name: str = "Unitree-7520-14"

    effort_limit: float = 88.
    velocity_limit: float = 32.
    knee_point_velocity: float = 32.
    armature: float = 0.010177520


@configclass
class UnitreeJoint4010(BadmintonJointCfg):
    """Configuration for Unitree-4010 robot joint model."""

    joint_model_name: str = "Unitree-4010"

    effort_limit: float = 5.
    velocity_limit: float = 2.30
    knee_point_velocity: float = 22.
    armature: float = 0.00425


# K1 Hip Pitch / T1 Waist, Hip Roll & Yaw
@configclass
class BadmintonJointE6408(BadmintonJointCfg):
    """Configuration for E6408 robot joint model."""

    joint_model_name: str = "E6408"

    effort_limit: float = 68.0
    velocity_limit: float = 14.66       # rad/s, 140 rpm
    knee_point_velocity: float = 1.88   # rad/s, 18 rpm
    armature: float = 0.0478125

# K1 Hip Roll / T1 Ankle
@configclass
class BadmintonJointE4315(BadmintonJointCfg):
    """Configuration for E4315 robot joint model."""

    joint_model_name: str = "E4315"

    effort_limit: float = 76.0
    velocity_limit: float = 12.57
    knee_point_velocity: float = 2.62
    armature: float = 0.0339552

# K1 Hip Yaw & Ankle / T1 Arm
@configclass
class BadmintonJointE4310(BadmintonJointCfg):
    """Configuration for E4310 robot joint model."""

    joint_model_name: str = "E4310"

    effort_limit: float = 38.3
    velocity_limit: float = 17.59
    knee_point_velocity: float = 7.85
    armature: float = 0.0282528


# K1 Knee
@configclass
class BadmintonJointE6416(BadmintonJointCfg):
    """Configuration for E6416 robot joint model."""

    joint_model_name: str = "E6416"

    effort_limit: float = 112.0
    velocity_limit: float = 12.57
    knee_point_velocity: float = 2.09
    armature: float = 0.095625


# K1 Arm
@configclass
class BadmintonJointR14(BadmintonJointCfg):
    """Configuration for R14 robot joint model."""

    joint_model_name: str = "R14"

    effort_limit: float = 14
    velocity_limit: float = 33.51
    knee_point_velocity: float = 5.24
    armature: float = 0.001


# K1 Neck
@configclass
class BadmintonJointHT4438(BadmintonJointCfg):
    """Configuration for HT4438 robot joint model."""

    joint_model_name: str = "HT4438"

    effort_limit: float = 6.0
    velocity_limit: float = 7.85
    knee_point_velocity: float = 10.47
    armature: float = 0.001


# T1 Hip-Pitch
@configclass
class BadmintonJointE8112(BadmintonJointCfg):
    """Configuration for E8112 robot joint model."""

    joint_model_name: str = "E8112"

    effort_limit: float = 98.8
    velocity_limit: float = 15.29       # rad/s, 146 rpm 
    knee_point_velocity: float = 7.53   # rad/s, 71.91 rpm
    armature: float = 0.0523908


# T1 Knee  
@configclass
class BadmintonJointE8116(BadmintonJointCfg):
    """Configuration for E8116 robot joint model."""

    joint_model_name: str = "E8116"

    effort_limit: float = 130.5
    velocity_limit: float = 14.76       # rad/s, 141 rpm
    knee_point_velocity: float = 6.71   # rad/s, 64.14 rpm
    armature: float = 0.0636012


# T1 Neck
@configclass
class BadmintonJointDM4310(BadmintonJointCfg):
    """Configuration for DM4310 robot joint model."""

    joint_model_name: str = "DM4310"

    effort_limit: float = 7.0
    velocity_limit: float = 41.89           # rad/s, 400 rpm
    knee_point_velocity: float = 26.18      # rad/s, 250 rpm
    armature: float = 0.0018