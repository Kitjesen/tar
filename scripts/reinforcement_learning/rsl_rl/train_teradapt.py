#!/usr/bin/env python3
# Copyright (c) 2024-2026 Inovxio (穹沛科技)
# SPDX-License-Identifier: Apache-2.0
"""Train TerAdapt: VQ-VAE terrain codebook + dual-horizon proprio + PPO.

Usage:
  python scripts/reinforcement_learning/rsl_rl/train_teradapt.py \
      --task GaitTerRough --num_envs 4096 --headless
"""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

# ---- CLI ----
parser = argparse.ArgumentParser(description="Train TerAdapt RL agent.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)

# TerAdapt-specific
parser.add_argument("--short_history_steps", type=int, default=5, help="Short encoder history length")
parser.add_argument("--long_history_steps", type=int, default=50, help="Long encoder history length")
parser.add_argument("--codebook_size", type=int, default=256)
parser.add_argument("--codebook_dim", type=int, default=16)
parser.add_argument("--vel_coef", type=float, default=1.0)
parser.add_argument("--tok_coef", type=float, default=1.0)
parser.add_argument("--vq_coef", type=float, default=1.0)

# local imports expected on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Hydra needs clean sys.argv
sys.argv = [sys.argv[0]] + hydra_args

# ---- Launch Isaac Sim ----
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Imports after simulation_app ----
import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Our TerAdapt runner
from runners.teradapt_on_policy_runner import TerAdaptOnPolicyRunner


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    # Update with CLI overrides
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    agent_cfg.num_envs = args_cli.num_envs if args_cli.num_envs is not None else agent_cfg.num_envs

    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.seed = agent_cfg.seed

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if agent_cfg.resume:
        if agent_cfg.load_run and "/" in agent_cfg.load_run:
            parent_log_path = os.path.dirname(log_root_path)
            parts = agent_cfg.load_run.split("/", 1)
            exp_from_load, run_from_load = parts[0], parts[1]
            resume_path = get_checkpoint_path(parent_log_path, exp_from_load, agent_cfg.load_checkpoint, other_dirs=[run_from_load])
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = None

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Resolve obs_groups from agent config
    obs_groups = getattr(agent_cfg, "obs_groups", None)
    if not isinstance(obs_groups, dict):
        obs_groups = {
            "policy_short": ["policy_short"],
            "policy_long": ["policy_long"],
            "critic": ["critic", "height_scan_group"],
            "height_scan": ["height_scan_group"],
            "vel": ["vel_gt"],
        }

    # Build train_cfg dict for TerAdaptOnPolicyRunner
    train_cfg = {
        "runner": {
            "policy_class_name": "TerAdaptActorCritic",
            "algorithm_class_name": "TerAdaptPPO",
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "save_interval": agent_cfg.save_interval,
            "experiment_name": agent_cfg.experiment_name,
            "max_iterations": agent_cfg.max_iterations,
            "short_history_steps": args_cli.short_history_steps,
            "long_history_steps": args_cli.long_history_steps,
            "obs_groups": obs_groups,
        },
        "policy": {
            "num_actions": env.num_actions,
            "init_noise_std": getattr(agent_cfg.policy, "init_noise_std", 1.0),
            "actor_hidden_dims": list(getattr(agent_cfg.policy, "actor_hidden_dims", [512, 256, 128])),
            "critic_hidden_dims": list(getattr(agent_cfg.policy, "critic_hidden_dims", [512, 256, 128])),
            "activation": getattr(agent_cfg.policy, "activation", "elu"),
            "codebook_size": args_cli.codebook_size,
            "codebook_dim": args_cli.codebook_dim,
            "short_history_steps": args_cli.short_history_steps,
            "long_history_steps": args_cli.long_history_steps,
            "vel_coef": args_cli.vel_coef,
            "tok_coef": args_cli.tok_coef,
            "vq_coef": args_cli.vq_coef,
        },
        "algorithm": {
            "num_learning_epochs": agent_cfg.algorithm.num_learning_epochs,
            "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
            "clip_param": agent_cfg.algorithm.clip_param,
            "gamma": agent_cfg.algorithm.gamma,
            "lam": agent_cfg.algorithm.lam,
            "value_loss_coef": agent_cfg.algorithm.value_loss_coef,
            "entropy_coef": agent_cfg.algorithm.entropy_coef,
            "learning_rate": agent_cfg.algorithm.learning_rate,
            "max_grad_norm": agent_cfg.algorithm.max_grad_norm,
            "use_clipped_value_loss": agent_cfg.algorithm.use_clipped_value_loss,
            "schedule": agent_cfg.algorithm.schedule,
            "desired_kl": agent_cfg.algorithm.desired_kl,
        },
        "obs_groups": obs_groups,
    }

    runner = TerAdaptOnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)

    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
