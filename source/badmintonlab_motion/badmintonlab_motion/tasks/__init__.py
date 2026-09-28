# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="BadmintonLab-Motion",
    entry_point=f"{__name__}.badmintonlab_motion_env:BadmintonLabMotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.badmintonlab_motion_env_cfg:BadmintonLabMotionAmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_ppo_cfg:AmpPPORunnerCfg",
    },
)


for task_name, env_cfg_name, runner_cfg_name in (
    ("Clear", "ClearEnvCfg", "ClearAmpPPORunnerCfg"),
    ("Drop", "DropEnvCfg", "DropAmpPPORunnerCfg"),
    ("Lift", "LiftEnvCfg", "LiftAmpPPORunnerCfg"),
):
    gym.register(
        id=f"BadmintonLab-Motion-{task_name}",
        entry_point=f"{__name__}.badmintonlab_motion_env:BadmintonLabMotionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.shot_experiment_cfg:{env_cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_ppo_cfg:{runner_cfg_name}",
        },
    )

