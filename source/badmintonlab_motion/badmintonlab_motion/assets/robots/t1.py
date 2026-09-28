from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.math import *
from badmintonlab_motion.assets.robots import actuator
from badmintonlab_motion.assets.robots.actuator import (
    BadmintonDelayedImplicitActuatorCfg,
    BadmintonDelayedPDActuatorCfg,
    DelayedImplicitActuatorCfg
)

_T1_URDF = Path(__file__).resolve().parent / "T1" / "T1_23dof.urdf"


BADMINTON_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        usd_dir="./tmp/usd",
        fix_base=False,
        asset_path=str(_T1_URDF),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.3,
            "Right_Shoulder_Roll": 1.3,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            ".*_Hip_Pitch": -0.2,
            ".*_Knee_Pitch": 0.4,
            ".*_Ankle_Pitch": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arms": BadmintonDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            joint_cfgs=actuator.BadmintonJointE4310(natural_freq=4.0, damping_ratio=1.5),
        ),
        "waist": BadmintonDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=["Waist"],
            joint_cfgs=actuator.BadmintonJointE6408(natural_freq=4.0, damping_ratio=1.5),
        ),
        "legs": BadmintonDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            joint_cfgs={
                ".*_Hip_Pitch": actuator.BadmintonJointE8112(natural_freq=4.0, damping_ratio=1.5),
                ".*_Hip_Roll": actuator.BadmintonJointE6408(natural_freq=4.0, damping_ratio=1.5),
                ".*_Hip_Yaw": actuator.BadmintonJointE6408(natural_freq=4.0, damping_ratio=1.5),
                ".*_Knee_Pitch": actuator.BadmintonJointE8116(natural_freq=4.0, damping_ratio=1.5),
            },
        ),
        "feet": BadmintonDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Ankle_Pitch",
                ".*_Ankle_Roll",
            ],
            joint_cfgs={
                ".*_Ankle_Pitch": actuator.BadmintonT1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BadmintonJointE4315(),
                    serial_index=0,
                    natural_freq = 4.0,
                    damping_ratio = 1.5,
                    effort_ratio=(1.7, 0.4),
                    velocity_ratio=(0.6, 1.0),
                    armature_ratio=(2,0.6),
                    knee_point_velocity_ratio=(1.0, 1.0),
                ),
                ".*_Ankle_Roll": actuator.BadmintonT1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BadmintonJointE4315(),
                    serial_index=1,
                    natural_freq = 4.0,
                    damping_ratio = 1.5,
                    effort_ratio=(1.7, 0.4),
                    velocity_ratio=(0.6, 1.0),
                    armature_ratio=(2,0.6),
                    knee_point_velocity_ratio=(1.0, 1.0),
                ),
            },
        ),
        "head": BadmintonDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[".*Head.*"],
            joint_cfgs=actuator.BadmintonJointDM4310(natural_freq=4.0, damping_ratio=1.5),
        ),
    },
)

T1_ACTION_SCALE = {}
for a in BADMINTON_T1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    d = a.damping
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    if not isinstance(d, dict):
        d = {n: d for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            T1_ACTION_SCALE[n] = 0.5 * e[n] / s[n]

print(f'{BADMINTON_T1_CFG.actuators=}')
print(f'{T1_ACTION_SCALE=}')
