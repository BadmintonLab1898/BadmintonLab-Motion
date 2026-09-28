# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from badmintonlab_motion.assets.court import BadmintonCourtCfg, create_badminton_court
from badmintonlab_motion.assets.robots.t1 import BADMINTON_T1_CFG, T1_ACTION_SCALE
from badmintonlab_motion.assets.robots.t2 import BADMINTON_T2_CFG, T2_ACTION_SCALE
from badmintonlab_motion.assets.shuttlecock import ShuttlecockSphereCfg

from . import mdp


_COURT_CFG = BadmintonCourtCfg()
_ROBOT_RESET_X_CENTER = -0.5 * _COURT_CFG.half_length
_ROBOT_RESET_X_RANGE = (_ROBOT_RESET_X_CENTER - 0.35, _ROBOT_RESET_X_CENTER + 0.35)
_ROBOT_RESET_Y_RANGE = (-0.15 * _COURT_CFG.singles_half_width, 0.15 * _COURT_CFG.singles_half_width)


@configclass
class BadmintonLabMotionSceneCfg(InteractiveSceneCfg):
    """Configuration for the base hitting scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robot — merge_fixed_joints=False so the racket_sweet_spot dummy body survives
    robot: ArticulationCfg = BADMINTON_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    shuttle: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shuttle",
        spawn=ShuttlecockSphereCfg(
            radius=0.025,
            # The sphere is collision-only; the child visual is the FBX asset.
            visual_material=sim_utils.PreviewSurfaceCfg(opacity=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.05,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.00519),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.4,
                restitution=0.05,
                friction_combine_mode="average",
                restitution_combine_mode="min",
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, 0.0, 1.55)),
    )

    racket_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/racket_link",
        history_length=4,
        track_air_time=False,
        force_threshold=1.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Shuttle"],
    )
    racket_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/racket_link",
        history_length=4,
        track_air_time=False,
        force_threshold=1.0,
        filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn = dataclasses.replace(self.robot.spawn, merge_fixed_joints=False,func=mdp.spawn_urdf_with_racket_material)



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True, scale=T1_ACTION_SCALE)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        robot_heading_court = ObsTerm(
            func=mdp.robot_heading_court,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_pos_court = ObsTerm(
            func=mdp.robot_pos_court,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        shuttle_pos_b = ObsTerm(
            func=mdp.shuttle_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot"), "shuttle_cfg": SceneEntityCfg("shuttle")},
        )
        shuttle_vel_b = ObsTerm(
            func=mdp.shuttle_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot"), "shuttle_cfg": SceneEntityCfg("shuttle")},
        )
        racket_pos_b = ObsTerm(
            func=mdp.racket_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["racket_sweet_spot"])},
        )
        racket_vel_b = ObsTerm(
            func=mdp.racket_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["racket_sweet_spot"])},
        )
        racket_ang_vel_b = ObsTerm(
            func=mdp.racket_ang_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["racket_sweet_spot"])},
        )
        shuttle_pos_racket = ObsTerm(
            func=mdp.shuttle_pos_in_sweet_spot_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["racket_sweet_spot"]),
                "shuttle_cfg": SceneEntityCfg("shuttle"),
            },
        )
        shuttle_vel_racket = ObsTerm(
            func=mdp.shuttle_vel_in_sweet_spot_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["racket_sweet_spot"]),
                "shuttle_cfg": SceneEntityCfg("shuttle"),
            },
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_fall = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.35},
    )
    bad_orientation = DoneTerm(
        func=mdp.root_vertical_deviation_exceed_limit,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_angle_deg": 45.0},
    )


@configclass
class EventCfg:
    """Configuration for events."""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["^(?!racket_link$).*"]),
            "static_friction_range": (0.6, 1.3),
            "dynamic_friction_range": (0.5, 1.1),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(Trunk|trunk)$"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.025, 0.025)},
        },
    )
    robot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )
    reset_robot_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": _ROBOT_RESET_X_RANGE,
                "y": _ROBOT_RESET_Y_RANGE,
                "z": (-0.01, 0.01),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.12, 0.12),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_shuttle_pose = EventTerm(
        func=mdp.reset_shuttle_root_state,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("shuttle"),
            "position_range": {
                "x": (1.8, 2.2),
                "y": (-1.3, 1.3),
                "z": (1.3, 1.6),
            },
            "flight_time_range": (1.4, 1.8),
            "target_x_range": (-_COURT_CFG.half_length, -3.96),
            "target_y_range": (-1.3, 1.3),
            "target_height_range": (0.05, 0.10),
            "gravity_z": -9.81,
            "shuttle_mass": 0.00519,
            "shuttle_drag_k": 0.00110,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the sweet-spot hit-and-fly task."""
    # body regularization
    # alive = RewTerm(func=mdp.is_alive, weight=0.1)
    anti_fall = RewTerm(
        func=mdp.root_height_above_minimum_reward,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": 0.55,
        },
    )
    # root_orientation = RewTerm(
    #     func=mdp.root_rpy_limit_reward,
    #     weight=0.1,
    # )
    racket_ground_support = RewTerm(
        func=mdp.racket_ground_support_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("racket_ground_contact", body_names=["racket_link"]),
            "force_threshold": 3.0,
            "force_scale": 10.0,
            "max_penalty": 1.0,
        },
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["^(left_foot_link|left_ankle_roll_link)$", "^(right_foot_link|right_ankle_roll_link)$"]
            ),
        },
    )
    # Best racket proximity over the incoming receive-height interval.
    interval_approach = RewTerm(
        func=mdp.interval_approach_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": 1.5,
            "max_height": 2.0,
            "incoming_speed_epsilon": 0.05,
            "full_reward_radius": 0.1,
            "distance_sigma": 0.45,
        },
    )
    hit_bonus = RewTerm(
        func=mdp.hit_bonus,
        weight=1.0,
        params={
            "required_hit_side": "forehand",
            # racket_sweet_spot local X (red) axis is the racket-face normal;
            # positive is forehand and negative is backhand for this model.
            "local_axis": 0,
            "side_margin": 0.0,
        },
    )

    # post hit
    dynamic_outgoing_velocity = RewTerm(
        func=mdp.dynamic_outgoing_velocity_reward,
        weight=1.0,
        params={
            "shuttle_cfg": SceneEntityCfg("shuttle"),
            "target_speed": 10.0,
            "minimum_net_clearance_height": 1.55,
            "direction_sigma": 0.15,
        },
    )
    net_cross_bonus = RewTerm(
        func=mdp.net_cross_bonus,
        weight=1.0,
        params={
            "net_full_reward_height": 1.55,
        },
    )
    backcourt_landing_reward = RewTerm(
        func=mdp.backcourt_landing_reward,
        weight=1.0,
        params={
            "landing_height": 0.03,
            "backcourt_depth": 1.50,
            "target_half_width": 0.75,
            "distance_sigma": 0.50,
        },
    )
    post_hit_robot_stable = RewTerm(
        func=mdp.post_hit_robot_stable_reward,
        weight=0.75,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "follow_time_s": 0.35,
            "position_sigma": 0.35,
            "heading_zero_angle_deg": 90.0,
        },
    )


@configclass
class BadmintonLabMotionEnvCfg(ManagerBasedRLEnvCfg):
    scene: BadmintonLabMotionSceneCfg = BadmintonLabMotionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Kept off for the ordinary PPO task so AMP feature collection adds no
    # overhead or behavioral changes to existing training and playback.
    enable_amp: bool = False
    shuttle_mass: float = 0.00519
    shuttle_drag_k: float = 0.00110
    gravity_z: float = -9.81
    shuttle_ground_height: float = 0.025
    shuttle_landing_detection_height: float = 0.03
    shuttle_orientation_gain: float = 2.0e-6
    shuttle_orientation_damping: float = 2.0e-5
    shuttle_orientation_speed_threshold: float = 0.10
    shuttle_orientation_max_torque: float = 2.0e-4

    hit_contact_threshold: float = 1.0
    hit_speed_threshold: float = 1.5
    hit_front_threshold: float = -0.01
    hit_region_y_radius: float = 0.045
    hit_region_z_radius: float = 0.070
    required_hit_side: str = "forehand"
    hit_side_local_axis: int = 0
    hit_side_margin: float = 0.0

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.origin_type = "world"
        self.viewer.eye = (7.0, -9.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)
        self.court: BadmintonCourtCfg = BadmintonCourtCfg()
        if self.required_hit_side not in ("forehand", "backhand"):
            raise ValueError("required_hit_side must be 'forehand' or 'backhand'")
        if self.hit_side_local_axis not in (0, 1, 2):
            raise ValueError("hit_side_local_axis must be 0, 1, or 2")
        if self.hit_side_margin < 0.0:
            raise ValueError("hit_side_margin must be non-negative")
        # attach court / net / line / post asset cfgs to the scene cfg so the
        # InteractiveScene clone-step spawns them under each env namespace
        for asset_name, asset_cfg in create_badminton_court(self.court).items():
            setattr(self.scene, asset_name, asset_cfg)


@configclass
class BadmintonLabMotionAmpEnvCfg(BadmintonLabMotionEnvCfg):
    enable_amp: bool = True


@configclass
class BadmintonLabMotionT2SceneCfg(BadmintonLabMotionSceneCfg):
    robot: ArticulationCfg = BADMINTON_T2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class T2ActionsCfg(ActionsCfg):
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True, scale=T2_ACTION_SCALE
    )

@configclass
class BadmintonLabMotionT2AmpEnvCfg(BadmintonLabMotionAmpEnvCfg):
    scene: BadmintonLabMotionT2SceneCfg = BadmintonLabMotionT2SceneCfg(num_envs=4096, env_spacing=2.5)
    actions: T2ActionsCfg = T2ActionsCfg()
    enable_amp: bool = True
