"""Task-specific reference-motion badminton configurations.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from badmintonlab_motion.assets.court import BadmintonCourtCfg

from . import mdp
from .badmintonlab_motion_env_cfg import (
    BadmintonLabMotionAmpEnvCfg,
    EventCfg,
    RewardsCfg,
)


_COURT = BadmintonCourtCfg()
_LONG_SERVICE_X = _COURT.half_length - 0.76

# Keep targets inside the painted lines.  The robot is on the negative-x side
# and the opponent is on the positive-x side.
OWN_BACKCOURT_X_RANGE = (-_COURT.half_length + 0.08, -_LONG_SERVICE_X - 0.08)
OPPONENT_BACKCOURT_X_RANGE = (_LONG_SERVICE_X + 0.08, _COURT.half_length - 0.08)
OWN_FRONTCOURT_X_RANGE = (-1.80, -0.35)
OPPONENT_FRONTCOURT_X_RANGE = (0.35, 1.80)
INCOMING_Y_RANGE = (-1.30, 1.30)
TARGET_Y_RANGE = (-_COURT.singles_half_width + 0.15, _COURT.singles_half_width - 0.15)


def make_incoming_reset_event(
    *,
    position_x_range: tuple[float, float],
    position_z_range: tuple[float, float],
    flight_time_range: tuple[float, float],
    target_x_range: tuple[float, float],
    target_y_range: tuple[float, float]=INCOMING_Y_RANGE,
) -> EventTerm:
    """Create a batched shuttle reset for one incoming shot family."""
    return EventTerm(
        func=mdp.reset_shuttle_root_state,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("shuttle"),
            "position_range": {
                "x": position_x_range,
                "y": INCOMING_Y_RANGE,
                "z": position_z_range,
            },
            "flight_time_range": flight_time_range,
            "target_x_range": target_x_range,
            "target_y_range": target_y_range,
            "target_height_range": (0.05, 0.10),
            "gravity_z": -9.81,
            "shuttle_mass": 0.00519,
            "shuttle_drag_k": 0.00110,
        },
    )

def make_approach_reward(min_height:float=1.5,max_height:float=2.0) -> RewTerm:
    """Use a smooth approach signal for all shot-specific tasks."""
    return RewTerm(
        func=mdp.interval_approach_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": min_height,
            "max_height": max_height,
            "incoming_speed_epsilon": 0.05,
            "full_reward_radius": 0.10,
            "distance_sigma": 1.0,
        },
    )

def make_target_landing_reward(target_x_range: tuple[float, float]) -> RewTerm:
    """Create the common post-hit reward for a requested court zone."""
    return RewTerm(
        func=mdp.target_landing_reward,
        weight=1.0,
        params={
            "target_x_range": target_x_range,
            "target_y_range": TARGET_Y_RANGE,
            "landing_height": 0.03,
            "distance_sigma": 0.50,
        },
    )


@configclass
class ClearEventCfg(EventCfg):
    """High rear-court feed used by clear and drop."""

    reset_shuttle_pose = make_incoming_reset_event(
        position_x_range=(1.80, 2.20),
        position_z_range=(1.30, 1.60),
        flight_time_range=(1.40, 1.80),
        target_x_range=OWN_BACKCOURT_X_RANGE,
    )

@configclass
class ClearEnvCfg(BadmintonLabMotionAmpEnvCfg):
    events: ClearEventCfg = ClearEventCfg()
    # rewards: default


@configclass
class DropEventCfg(EventCfg):
    """A slightly shorter high feed for drop practice."""

    reset_shuttle_pose = make_incoming_reset_event(
        position_x_range=(1.55, 2.00),
        position_z_range=(1.45, 1.75),
        flight_time_range=(1.4, 1.8),
        target_x_range=(-_COURT.half_length + 0.08, -_COURT.half_length + 0.88),
    )

@configclass
class DropRewardsCfg(RewardsCfg):
    interval_approach = make_approach_reward(min_height=2.0, max_height=2.5)
    backcourt_landing_reward = make_target_landing_reward(OPPONENT_FRONTCOURT_X_RANGE)

@configclass
class DropEnvCfg(BadmintonLabMotionAmpEnvCfg):
    events: DropEventCfg = DropEventCfg()
    rewards: DropRewardsCfg = DropRewardsCfg()


@configclass
class LiftEventCfg(EventCfg):
    """High frontcourt feed used by lift."""

    reset_shuttle_pose = make_incoming_reset_event(
        position_x_range=(6.00, 6.70),
        position_z_range=(2.00, 3.0),
        flight_time_range=(1.50, 1.90),
        target_x_range=(-2, -1),
        target_y_range=(-2.5, -1.5),
    )

@configclass
class LiftRewardsCfg(RewardsCfg):
    hit_bonus = RewTerm(
        func=mdp.hit_bonus_any_side,
        weight=1.0,
    )
    interval_approach = make_approach_reward(min_height=0.5, max_height=1.0)
    target_landing_reward = make_target_landing_reward(OPPONENT_BACKCOURT_X_RANGE)

@configclass
class LiftEnvCfg(BadmintonLabMotionAmpEnvCfg):
    events: LiftEventCfg = LiftEventCfg()
    rewards: LiftRewardsCfg = LiftRewardsCfg()
