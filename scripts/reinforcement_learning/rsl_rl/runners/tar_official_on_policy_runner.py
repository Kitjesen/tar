# Copyright (c) 2024-2026 Inovxio (穹沛科技)
"""TAR Official runner — wires env / policy / PPO / storage together."""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from algorithms.tar_official_ppo import TAROfficialPPO
from modules.tar_official_actor_critic import TARActorCriticOfficial


class TAROfficialOnPolicyRunner:
    """On-policy runner for TAR official architecture.

    obs_groups (dict) specifies how to assemble actor vs critic observations:
      {"policy": [...], "critic": [...]}
    Each group is a list of Isaac Lab obs group names to concatenate along feature dim.
    """

    def __init__(self, env, train_cfg: dict, log_dir: Optional[str] = None, device: str = "cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        obs = self.env.get_observations()
        self.obs_groups = train_cfg.get("obs_groups", None) or self.cfg.get("obs_groups", None) or {
            "policy": ["policy"],
            "critic": ["critic", "height_scan_group"],
        }
        self.policy_groups = self.obs_groups["policy"]
        self.critic_groups = self.obs_groups["critic"]

        if isinstance(obs, dict):
            actor_obs_dim = sum(obs[g].shape[1] for g in self.policy_groups if g in obs)
            critic_obs_dim = sum(obs[g].shape[1] for g in self.critic_groups if g in obs)
        else:
            raise ValueError("TAR official runner expects dict obs")

        self.num_actor_obs = actor_obs_dim
        self.num_critic_obs = critic_obs_dim

        # num_hist should be supplied by train_cfg
        num_hist = int(self.cfg.get("num_hist", 10))
        prop_dim = actor_obs_dim // num_hist
        if actor_obs_dim % num_hist != 0:
            raise ValueError(
                f"actor_obs_dim={actor_obs_dim} not divisible by num_hist={num_hist}"
            )

        print("\n=== TAR Official Runner ===")
        print(f"  policy groups: {self.policy_groups} -> {actor_obs_dim} = {num_hist} x {prop_dim}")
        print(f"  critic groups: {self.critic_groups} -> {critic_obs_dim}")

        # Policy
        policy_kwargs = dict(self.policy_cfg)
        policy_kwargs.setdefault("num_actor_obs", actor_obs_dim)
        policy_kwargs.setdefault("num_critic_obs", critic_obs_dim)
        policy_kwargs.setdefault("num_actions", self.env.num_actions)
        policy_kwargs.setdefault("num_hist", num_hist)
        policy_kwargs.setdefault("prop_dim", prop_dim)
        for k in ("policy_class_name", "algorithm_class_name", "class_name",
                  "num_steps_per_env", "save_interval", "max_iterations",
                  "experiment_name", "run_name", "obs_groups", "num_hist"):
            policy_kwargs.pop(k, None)

        self.actor_critic = TARActorCriticOfficial(**policy_kwargs).to(self.device)

        # Algorithm
        alg_kwargs = dict(self.alg_cfg)
        alg_kwargs.pop("class_name", None)
        self.alg: TAROfficialPPO = TAROfficialPPO(
            actor_critic=self.actor_critic,
            num_envs=self.env.num_envs,
            device=self.device,
            **alg_kwargs,
        )

        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.cfg["num_steps_per_env"],
            obs_shape=(actor_obs_dim,),
            critic_obs_shape=(critic_obs_dim,),
            actions_shape=(self.env.num_actions,),
        )

        # Logging
        self.log_dir = log_dir
        self.writer = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    # ----- obs extract -----

    def _extract(self, obs_dict):
        """Return (actor_obs, critic_obs) as flat tensors."""
        actor_obs = torch.cat([obs_dict[g] for g in self.policy_groups], dim=1)
        critic_obs = torch.cat([obs_dict[g] for g in self.critic_groups], dim=1)
        return actor_obs, critic_obs

    # ----- main loop -----

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs_dict = self.env.get_observations()
        obs, critic_obs = self._extract(obs_dict)
        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)

        self.alg.train_mode()
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs, critic_obs)
                    next_obs_dict, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    next_obs, next_critic_obs = self._extract(next_obs_dict)
                    next_obs = next_obs.to(self.device)
                    next_critic_obs = next_critic_obs.to(self.device)

                    self.alg.process_env_step(rewards, dones, extras, next_critic_obs)

                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

                    obs = next_obs
                    critic_obs = next_critic_obs

                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(critic_obs)

            mean_value, mean_surr, mean_triplet, mean_vel, mean_pos, mean_neg = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it

            locs = {
                "it": it, "tot_iter": tot_iter,
                "collect_time": collect_time, "learn_time": learn_time,
                "value_loss": mean_value, "surrogate_loss": mean_surr,
                "triplet_loss": mean_triplet, "vel_loss": mean_vel,
                "pos_loss": mean_pos, "neg_loss": mean_neg,
                "rewbuffer": self.rewbuffer, "lenbuffer": self.lenbuffer,
            }
            self.log(locs)

            if (it + 1) % self.cfg["save_interval"] == 0 or it == tot_iter - 1:
                self.save(os.path.join(self.log_dir, f"model_{it+1}.pt"))

    def log(self, locs: dict, width: int = 80):
        self.tot_timesteps += self.cfg["num_steps_per_env"] * self.env.num_envs
        self.tot_time += locs["collect_time"] + locs["learn_time"]
        fps = int(self.cfg["num_steps_per_env"] * self.env.num_envs /
                  (locs["collect_time"] + locs["learn_time"] + 1e-6))
        mean_rew = statistics.mean(locs["rewbuffer"]) if len(locs["rewbuffer"]) > 0 else 0.0
        mean_len = statistics.mean(locs["lenbuffer"]) if len(locs["lenbuffer"]) > 0 else 0.0

        if self.writer is not None:
            self.writer.add_scalar("Loss/value", locs["value_loss"], locs["it"])
            self.writer.add_scalar("Loss/surrogate", locs["surrogate_loss"], locs["it"])
            self.writer.add_scalar("Loss/triplet", locs["triplet_loss"], locs["it"])
            self.writer.add_scalar("Loss/velocity_MSE", locs["vel_loss"], locs["it"])
            self.writer.add_scalar("Loss/tar_pos", locs["pos_loss"], locs["it"])
            self.writer.add_scalar("Loss/tar_neg", locs["neg_loss"], locs["it"])
            self.writer.add_scalar("Performance/total_fps", fps, locs["it"])
            self.writer.add_scalar("Train/mean_reward", mean_rew, locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", mean_len, locs["it"])

        print("=" * width)
        print(
            f"Iter {locs['it']+1}/{locs['tot_iter']}  fps={fps}  "
            f"val={locs['value_loss']:.3f}  sur={locs['surrogate_loss']:.3f}  "
            f"trip={locs['triplet_loss']:.3f}  vel={locs['vel_loss']:.3f}  "
            f"pos={locs['pos_loss']:.3f}  neg={locs['neg_loss']:.3f}  "
            f"rew={mean_rew:.2f}  len={mean_len:.1f}"
        )

    def save(self, path: str, infos: dict = None):
        torch.save({
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }, path)

    def load(self, path: str, load_optimizer: bool = True):
        loaded = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(loaded["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in loaded:
            self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        self.current_learning_iteration = loaded.get("iter", 0)
        return loaded.get("infos", None)

    def get_inference_policy(self, device: Optional[str] = None):
        self.alg.test_mode()
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_inference
