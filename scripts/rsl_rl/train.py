# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the reference-motion badminton task with AMP-PPO on a single device."""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args


def parse_args() -> argparse.Namespace:
    """Parse training arguments and forward unknown arguments to Hydra."""
    parser = argparse.ArgumentParser(description="Train the reference-motion badminton policy with AMP-PPO.")
    parser.add_argument("--num_envs", type=int, help="Number of parallel environments.")
    parser.add_argument("--task", type=str, required=True, help="Registered Gym task name.")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--seed", type=int, help="Environment and policy seed.")
    parser.add_argument("--max_iterations", type=int, help="Number of training iterations.")
    parser.add_argument("--export_io_descriptors", action="store_true")
    parser.add_argument("--discriminator_checkpoint", type=Path,
                        help="Load only discriminator/AMP normalization from this checkpoint; policy resume is unchanged.")
    parser.add_argument("--freeze_discriminator", action=argparse.BooleanOptionalAction, default=None,
                        help="Keep the loaded discriminator fixed while training actor/critic with AMP rewards.")
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    if args.discriminator_checkpoint is not None:
        args.discriminator_checkpoint = args.discriminator_checkpoint.expanduser().resolve(strict=True)
    sys.argv = [sys.argv[0], *hydra_args]
    return args


# Isaac Sim must be launched before importing the remaining simulation modules.
args_cli = parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch
from rsl_rl.runners import on_policy_runner as _on_policy_runner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import badmintonlab_motion.tasks  # noqa: F401
from badmintonlab_motion.tasks.agents.amp_ppo import AMPPPO
from badmintonlab_motion.tasks.agents.amp_runner import AMPOnPolicyRunner

# RSL-RL resolves algorithm classes as attributes of its runner module.
_on_policy_runner.AMPPPO = AMPPPO


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train the reference-motion AMP policy."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.export_io_descriptors = args_cli.export_io_descriptors

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    try:
        runner = AMPOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)

        if agent_cfg.resume:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path, load_discriminator=args_cli.discriminator_checkpoint is None)

        # Independent reward-model source takes precedence over any AMP fields in
        # the policy resume checkpoint. Its actor/critic and PPO iteration are ignored.
        if args_cli.discriminator_checkpoint is not None:
            print(f"[INFO] Loading discriminator only from: {args_cli.discriminator_checkpoint}")
            checkpoint = torch.load(args_cli.discriminator_checkpoint, weights_only=False, map_location=agent_cfg.device)
            runner.alg.load_discriminator_checkpoint(checkpoint)
        if args_cli.freeze_discriminator is not None:
            runner.alg.set_discriminator_frozen(args_cli.freeze_discriminator)
        if runner.alg.amp_freeze_discriminator and not runner.alg.discriminator_loaded:
            raise ValueError("--freeze_discriminator requires a loaded discriminator checkpoint; refusing to freeze random weights")
        agent_cfg.algorithm.amp_freeze_discriminator = runner.alg.amp_freeze_discriminator
        print(f"[INFO] Discriminator frozen: {runner.alg.amp_freeze_discriminator}; actor/critic trained with PPO")

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_yaml(os.path.join(log_dir, "params", "amp_sources.yaml"), {
            "policy_checkpoint": resume_path if agent_cfg.resume else None,
            "discriminator_checkpoint": str(args_cli.discriminator_checkpoint) if args_cli.discriminator_checkpoint else None,
            "freeze_discriminator": runner.alg.amp_freeze_discriminator,
        })
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
