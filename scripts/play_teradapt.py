#!/usr/bin/env python3
"""Play TerAdapt policy with forced terrain level.

Usage:
  python scripts/reinforcement_learning/rsl_rl/play_teradapt.py \
      --task GaitTerRough --num_envs 64 --headless \
      --video --video_length 1000 --enable_cameras \
      --load_run 2026-04-23_04-46-43 --checkpoint model_2500.pt \
      --force_terrain_level 7 \
      --experiment_name thunder_gait_teradapt_rough
"""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play TerAdapt policy.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="GaitTerRough")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--force_terrain_level", type=int, default=None,
                    help="If set, override all env terrain levels to this value after reset.")
parser.add_argument("--num_steps", type=int, default=1000)

parser.add_argument("--short_history_steps", type=int, default=5)
parser.add_argument("--long_history_steps", type=int, default=50)
parser.add_argument("--codebook_size", type=int, default=256)
parser.add_argument("--codebook_dim", type=int, default=16)
parser.add_argument("--vel_coef", type=float, default=1.0)
parser.add_argument("--tok_coef", type=float, default=1.0)
parser.add_argument("--vq_coef", type=float, default=1.0)

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

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401
from runners.teradapt_on_policy_runner import TerAdaptOnPolicyRunner


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.num_envs = args_cli.num_envs
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # 2026-04-26 patch: viewer — 单 env 跟随特写 / 多 env 俯视全场
    if args_cli.video and hasattr(env_cfg, "viewer"):
        if args_cli.num_envs == 1:
            # 特写跟随机器人 0
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.asset_name = "robot"
            env_cfg.viewer.env_index = 0
            env_cfg.viewer.eye = (2.5, 2.5, 1.5)
            env_cfg.viewer.lookat = (0.0, 0.0, 0.4)
        elif args_cli.num_envs <= 16:
            # 近距离多狗中景：相机抬高 ~3.5m，水平 ~6m 距离
            import math
            grid_size = int(math.ceil(math.sqrt(args_cli.num_envs)))
            half = (grid_size - 1) * 2.5 / 2
            env_cfg.viewer.origin_type = "world"
            env_cfg.viewer.eye = (half + 6.0, half + 6.0, 3.5)
            env_cfg.viewer.lookat = (half, half, 0.5)
        else:
            # 俯视全场 (env_spacing=2.5, 64 envs → 8x8 网格 ~17.5m, 中心约 (8.75, 8.75))
            import math
            grid_size = int(math.ceil(math.sqrt(args_cli.num_envs)))
            half = (grid_size - 1) * 2.5 / 2
            env_cfg.viewer.origin_type = "world"
            env_cfg.viewer.eye = (half + 18.0, half + 18.0, 14.0)
            env_cfg.viewer.lookat = (half, half, 0.5)
        env_cfg.viewer.resolution = (1280, 720)

    # Reduce curriculum randomness for evaluation — disable terrain curriculum,
    # we will force levels manually after reset.
    if args_cli.force_terrain_level is not None:
        if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None
        # Allow init level to be at the forced level (so layout is valid)
        if hasattr(env_cfg.scene.terrain, "max_init_terrain_level"):
            env_cfg.scene.terrain.max_init_terrain_level = args_cli.force_terrain_level

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    play_log_dir = os.path.join(log_root_path, agent_cfg.load_run, "play_" + datetime.now().strftime("%H%M%S"))
    os.makedirs(play_log_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(play_log_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    obs_groups = getattr(agent_cfg, "obs_groups", None)
    if not isinstance(obs_groups, dict):
        obs_groups = {
            "policy_short": ["policy_short"],
            "policy_long": ["policy_long"],
            "critic": ["critic", "height_scan_group"],
            "height_scan": ["height_scan_group"],
            "vel": ["vel_gt"],
        }

    train_cfg = {
        "runner": {
            "policy_class_name": "TerAdaptActorCritic",
            "algorithm_class_name": "TerAdaptPPO",
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "save_interval": agent_cfg.save_interval,
            "experiment_name": agent_cfg.experiment_name,
            "max_iterations": 0,
            "short_history_steps": args_cli.short_history_steps,
            "long_history_steps": args_cli.long_history_steps,
            "obs_groups": obs_groups,
        },
        "policy": {
            "num_actions": env.num_actions,
            "init_noise_std": 1.0,
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
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
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

    runner = TerAdaptOnPolicyRunner(env, train_cfg, log_dir=play_log_dir, device=agent_cfg.device)

    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(resume_path)

    policy = runner.actor_critic
    policy.eval()

    # Force terrain level across all envs
    if args_cli.force_terrain_level is not None:
        terrain = env.unwrapped.scene.terrain
        forced_level = torch.full_like(terrain.terrain_levels, args_cli.force_terrain_level)
        terrain.terrain_levels[:] = forced_level
        terrain.env_origins[:] = terrain.terrain_origins[
            terrain.terrain_levels, terrain.terrain_types
        ]
        print(f"[INFO] Forced all {len(terrain.terrain_levels)} envs to terrain level {args_cli.force_terrain_level}")

    # Reset and roll out
    obs_raw = env.get_observations()
    if isinstance(obs_raw, tuple):
        obs_raw = obs_raw[0]

    extract_obs = runner._extract_observations

    short_obs, long_obs, critic_obs, hscan, vel_gt = extract_obs(obs_raw)
    short_obs = short_obs.to(agent_cfg.device)
    long_obs = long_obs.to(agent_cfg.device)

    step_count = 0
    reward_sum = torch.zeros(env.num_envs, device=agent_cfg.device)
    dist_sum = torch.zeros(env.num_envs, device=agent_cfg.device)
    init_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()

    with torch.inference_mode():
        while step_count < args_cli.num_steps and simulation_app.is_running():
            # Deterministic inference via policy.act_inference
            action = policy.act_inference(short_obs, long_obs)
            next_obs_raw, rew, dones, extras = env.step(action.to(env.device))
            short_obs, long_obs, critic_obs, hscan, vel_gt = extract_obs(next_obs_raw)
            short_obs = short_obs.to(agent_cfg.device)
            long_obs = long_obs.to(agent_cfg.device)
            reward_sum += rew.to(agent_cfg.device)
            # Track distance traveled (from initial position)
            cur_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
            dist_sum = torch.norm(cur_pos - init_pos, dim=1)
            step_count += 1
            if step_count % 100 == 0:
                mean_rew = reward_sum.mean().item()
                mean_dist = dist_sum.mean().item()
                print(f"[step {step_count}]  mean_rew={mean_rew:.2f}  mean_dist={mean_dist:.2f}m  "
                      f"fallen_envs={(env.unwrapped.scene['robot'].data.root_pos_w[:, 2] < 0.2).sum().item()}")

    print(f"\n[SUMMARY] {args_cli.num_steps} steps on terrain level {args_cli.force_terrain_level}:")
    print(f"  mean cumulative reward : {reward_sum.mean().item():.2f}")
    print(f"  mean distance traveled : {dist_sum.mean().item():.2f} m")
    print(f"  min / max distance     : {dist_sum.min().item():.2f} / {dist_sum.max().item():.2f} m")
    robot_z = env.unwrapped.scene["robot"].data.root_pos_w[:, 2]
    print(f"  final robot height z   : {robot_z.mean().item():.3f} m (min {robot_z.min().item():.3f})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
