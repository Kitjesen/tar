#!/usr/bin/env python3
"""Train TAR official (paper-accurate TARLoco port) on Thunder.

Usage:
  python scripts/train_tar.py \
      --task GaitTarRough --num_envs 4096 --headless
"""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train TAR official (TARLoco port) on Thunder")
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
# TAR-specific overrides
parser.add_argument("--num_hist", type=int, default=10, help="Actor history length")
parser.add_argument("--num_hist_short", type=int, default=4, help="Short history for vel estimator")
parser.add_argument("--latent_dims", type=int, default=20)
parser.add_argument("--tar_coef", type=float, default=1.0)
parser.add_argument("--vel_coef", type=float, default=1.0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401 -- triggers gym.register for GaitTarRough

from tar.tar_on_policy_runner import TAROnPolicyRunner


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    agent_cfg.num_envs = args_cli.num_envs if args_cli.num_envs is not None else agent_cfg.num_envs

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
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
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if agent_cfg.resume:
        if agent_cfg.load_run and "/" in agent_cfg.load_run:
            parent = os.path.dirname(log_root_path)
            exp, run = agent_cfg.load_run.split("/", 1)
            resume_path = get_checkpoint_path(parent, exp, agent_cfg.load_checkpoint, other_dirs=[run])
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = None

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    obs_groups = getattr(agent_cfg, "obs_groups", None)
    if not isinstance(obs_groups, dict):
        obs_groups = {
            "policy": ["policy"],
            "critic": ["critic", "height_scan_group"],
        }

    train_cfg = {
        "runner": {
            "policy_class_name": "TARActorCritic",
            "algorithm_class_name": "TARPPO",
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "save_interval": agent_cfg.save_interval,
            "experiment_name": agent_cfg.experiment_name,
            "max_iterations": agent_cfg.max_iterations,
            "num_hist": args_cli.num_hist,
            "obs_groups": obs_groups,
        },
        "policy": {
            "num_actions": env.num_actions,
            "init_noise_std": getattr(agent_cfg.policy, "init_noise_std", 1.0),
            "actor_hidden_dims": list(getattr(agent_cfg.policy, "actor_hidden_dims", [256, 128, 128])),
            "critic_hidden_dims": list(getattr(agent_cfg.policy, "critic_hidden_dims", [512, 256, 256])),
            "activation": getattr(agent_cfg.policy, "activation", "elu"),
            "num_hist_short": args_cli.num_hist_short,
            "latent_dims": args_cli.latent_dims,
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
            "tar_coef": args_cli.tar_coef,
            "vel_coef": args_cli.vel_coef,
        },
        "obs_groups": obs_groups,
    }

    runner = TAROnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)

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
