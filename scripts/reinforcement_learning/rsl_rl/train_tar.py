#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Train RL agent with TAR (Teacher-Aligned Representations, arXiv:2503.20839).

Usage:
    python scripts/reinforcement_learning/rsl_rl/train_tar.py \
        --task GaitTarRough --num_envs 4096 --headless
"""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train RL agent with TAR.")

parser.add_argument("--video", action="store_true", default=False,
                    help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200,
                    help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000,
                    help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None,
                    help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None,
                    help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False,
                    help="Run training with multiple GPUs.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False,
                    help="Export IO descriptors.")

# TAR-specific arguments
parser.add_argument("--z_dim", type=int, default=45,
                    help="TAR latent representation dimension (default: 45).")
parser.add_argument("--triplet_margin", type=float, default=0.5,
                    help="Triplet loss margin (default: 0.5).")
parser.add_argument("--triplet_coef", type=float, default=1.0,
                    help="Triplet loss coefficient (default: 1.0).")
parser.add_argument("--estimator_coef", type=float, default=1.0,
                    help="Estimator MSE loss coefficient (default: 1.0).")
parser.add_argument("--num_estimator_targets", type=int, default=6,
                    help="GT estimator target dim: lin_vel(3)+pos_z(1)+com_xy(2)=6.")

import cli_args  # isort: skip

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""
import importlib.metadata as metadata

from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    print(
        f"Please install the correct version of RSL-RL.\n"
        f"Existing version is: '{installed_version}' and required version is: '{RSL_RL_VERSION}'."
    )
    exit(1)

"""Rest everything follows."""
from pathlib import Path

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from tensordict import TensorDict

rsl_rl_path = Path(__file__).resolve().parent
if str(rsl_rl_path) not in sys.path:
    sys.path.insert(0, str(rsl_rl_path))

from runners.tar_on_policy_runner import TAROnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Train with TAR (Teacher-Aligned Representations)."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

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

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors

    env_cfg.log_dir = log_dir

    env = gym.make(
        args_cli.task, cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    resume_path = None
    if agent_cfg.resume:
        if agent_cfg.load_run and "/" in agent_cfg.load_run:
            load_path_parts = agent_cfg.load_run.split("/", 1)
            experiment_name_from_load = load_path_parts[0]
            run_dir_from_load = load_path_parts[1]
            parent_log_path = os.path.abspath(os.path.join("logs", "rsl_rl"))
            resume_path = get_checkpoint_path(
                parent_log_path,
                experiment_name_from_load,
                agent_cfg.load_checkpoint,
                other_dirs=[run_dir_from_load],
            )
        else:
            resume_path = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint,
            )

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Detect history_length from env observation config
    obs_manager = env.unwrapped.observation_manager
    history_len = 15  # TAR default
    if "policy" in obs_manager._group_obs_term_cfgs:
        policy_term_cfgs = obs_manager._group_obs_term_cfgs["policy"]
        if policy_term_cfgs:
            history_len = getattr(policy_term_cfgs[0], "history_length", 15)

    obs = env.get_observations()
    policy_obs_dim = (
        obs["policy"].shape[1] if isinstance(obs, TensorDict) else obs.shape[1]
    )
    one_step_obs_dim = policy_obs_dim // history_len if history_len > 0 else policy_obs_dim
    print(
        f"[INFO] TAR history: length={history_len}, policy_obs={policy_obs_dim}, one_step_obs={one_step_obs_dim}"
    )

    # Resolve obs_groups from agent config (supports dict or plain attr)
    obs_groups = getattr(agent_cfg, "obs_groups", {
        "policy": ["policy"],
        "critic": ["critic", "height_scan_group"],
        "estimator": ["estimator_targets"],
    })
    if not isinstance(obs_groups, dict):
        obs_groups = {
            "policy": ["policy"],
            "critic": ["critic", "height_scan_group"],
            "estimator": ["estimator_targets"],
        }

    train_cfg = {
        "runner": {
            "policy_class_name": "TARActorCritic",
            "algorithm_class_name": "TARPPO",
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "save_interval": agent_cfg.save_interval,
            "obs_groups": obs_groups,
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
        "policy": {
            "actor_hidden_dims": agent_cfg.policy.actor_hidden_dims,
            "critic_hidden_dims": agent_cfg.policy.critic_hidden_dims,
            "activation": agent_cfg.policy.activation,
            "init_noise_std": agent_cfg.policy.init_noise_std,
            "z_dim": args_cli.z_dim,
            "triplet_margin": args_cli.triplet_margin,
            "triplet_coef": args_cli.triplet_coef,
            "estimator_coef": args_cli.estimator_coef,
            "num_estimator_targets": args_cli.num_estimator_targets,
        },
        "history_len": history_len,
    }

    runner = TAROnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    if agent_cfg.resume:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "train.yaml"), train_cfg)

    if agent_cfg.resume:
        print(f"[INFO] Resuming from iteration {runner.current_learning_iteration}")
        print(f"[INFO] Will train for {agent_cfg.max_iterations} more iterations")
    else:
        print(f"[INFO] Starting new TAR training")
        print(f"[INFO] Max iterations: {agent_cfg.max_iterations}")
    print(f"[INFO] History length: {history_len}")
    print(f"[INFO] z_dim: {args_cli.z_dim}")
    print(f"[INFO] Triplet margin: {args_cli.triplet_margin}, coef: {args_cli.triplet_coef}")
    print(f"[INFO] Estimator coef: {args_cli.estimator_coef}")
    print(f"[INFO] Estimator targets: {args_cli.num_estimator_targets}")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
