"""TerAdaptOnPolicyRunner - runs TerAdapt training loop.

Extracts 5 groups from obs_dict:
  policy_short, policy_long, critic (+ height_scan), height_scan, vel_gt
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from teradapt.teradapt_ppo import TerAdaptPPO
from teradapt.teradapt_actor_critic import TerAdaptActorCritic


class TerAdaptOnPolicyRunner:
    """On-policy runner for TerAdapt (dual-horizon + VQ-VAE terrain codebook)."""

    def __init__(self, env, train_cfg: dict, log_dir: Optional[str] = None, device: str = "cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        obs = self.env.get_observations()
        # Isaac Lab RslRlVecEnvWrapper returns (obs, extras) tuple in newer versions
        if isinstance(obs, tuple):
            obs = obs[0]

        # obs_groups
        self.obs_groups = train_cfg.get("obs_groups", None) or self.cfg.get("obs_groups", None) or {
            "policy_short": ["policy_short"],
            "policy_long": ["policy_long"],
            "critic": ["critic", "height_scan_group"],
            "height_scan": ["height_scan_group"],
            "vel": ["vel_gt"],
        }
        self.policy_short_groups = self.obs_groups["policy_short"]
        self.policy_long_groups = self.obs_groups["policy_long"]
        self.critic_groups = self.obs_groups["critic"]
        self.height_scan_groups = self.obs_groups["height_scan"]
        self.vel_groups = self.obs_groups["vel"]

        # Extract initial obs to infer dims (dict or TensorDict supported)
        if hasattr(obs, "keys"):
            short_obs_dim = sum(obs[g].shape[1] for g in self.policy_short_groups if g in obs)
            long_obs_dim = sum(obs[g].shape[1] for g in self.policy_long_groups if g in obs)
            critic_obs_dim = sum(obs[g].shape[1] for g in self.critic_groups if g in obs)
            height_scan_dim = sum(obs[g].shape[1] for g in self.height_scan_groups if g in obs)
            vel_dim = sum(obs[g].shape[1] for g in self.vel_groups if g in obs)
        else:
            raise ValueError(
                f"TerAdapt runner expects dict/TensorDict obs; got {type(obs).__name__}"
            )

        self.num_short_obs = short_obs_dim
        self.num_long_obs = long_obs_dim
        self.num_critic_obs = critic_obs_dim
        self.num_height_scan = height_scan_dim
        self.num_vel_targets = vel_dim

        # Derive per-step obs dim from short history length (provided by train cfg)
        short_hist = self.cfg.get("short_history_steps", 5)
        long_hist = self.cfg.get("long_history_steps", 50)
        if short_obs_dim % short_hist != 0:
            raise ValueError(
                f"short_obs_dim={short_obs_dim} not divisible by short_history_steps={short_hist}"
            )
        obs_dim_per_step = short_obs_dim // short_hist
        expected_long = long_hist * obs_dim_per_step
        if long_obs_dim != expected_long:
            raise ValueError(
                f"long_obs_dim={long_obs_dim} != {long_hist}*{obs_dim_per_step}={expected_long}"
            )
        self.obs_dim_per_step = obs_dim_per_step
        self.short_history_steps = short_hist
        self.long_history_steps = long_hist

        print("\n=== TerAdapt Runner ===")
        print(f"  short groups : {self.policy_short_groups} -> {short_obs_dim} = {short_hist} x {obs_dim_per_step}")
        print(f"  long groups  : {self.policy_long_groups}  -> {long_obs_dim} = {long_hist} x {obs_dim_per_step}")
        print(f"  critic groups: {self.critic_groups}     -> {critic_obs_dim}")
        print(f"  height_scan  : {self.height_scan_groups} -> {height_scan_dim}")
        print(f"  vel_gt       : {self.vel_groups}         -> {vel_dim}")

        # Build policy
        actor_critic_kwargs = dict(self.policy_cfg)
        actor_critic_kwargs.setdefault("num_short_obs", short_obs_dim)
        actor_critic_kwargs.setdefault("num_long_obs", long_obs_dim)
        actor_critic_kwargs.setdefault("num_actor_obs", obs_dim_per_step)
        actor_critic_kwargs.setdefault("num_critic_obs", critic_obs_dim)
        actor_critic_kwargs.setdefault("num_height_scan", height_scan_dim)
        actor_critic_kwargs.setdefault("num_vel_targets", vel_dim)
        actor_critic_kwargs.setdefault("obs_dim_per_step", obs_dim_per_step)
        actor_critic_kwargs.setdefault("short_history_steps", short_hist)
        actor_critic_kwargs.setdefault("long_history_steps", long_hist)
        actor_critic_kwargs.setdefault("num_actions", self.env.num_actions)
        for key in (
            "policy_class_name",
            "algorithm_class_name",
            "class_name",
            "num_steps_per_env",
            "save_interval",
            "max_iterations",
            "experiment_name",
            "run_name",
            "obs_groups",
        ):
            actor_critic_kwargs.pop(key, None)

        self.actor_critic = TerAdaptActorCritic(**actor_critic_kwargs).to(self.device)

        # Build algorithm
        alg_kwargs = dict(self.alg_cfg)
        for key in ("class_name",):
            alg_kwargs.pop(key, None)
        self.alg: TerAdaptPPO = TerAdaptPPO(
            actor_critic=self.actor_critic,
            device=self.device,
            **alg_kwargs,
        )

        # Storage
        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.cfg["num_steps_per_env"],
            short_obs_shape=(short_obs_dim,),
            long_obs_shape=(long_obs_dim,),
            critic_obs_shape=(critic_obs_dim,),
            actions_shape=(self.env.num_actions,),
            num_height_scan=height_scan_dim,
            num_vel_targets=vel_dim,
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

    # ---- observation extraction ----

    def _extract_observations(self, obs_dict):
        """Return (short_obs, long_obs, critic_obs, height_scan, vel_gt)."""
        # Unwrap (obs, extras) tuple if wrapper returned one
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        # Accept dict or TensorDict
        if not hasattr(obs_dict, "keys"):
            raise ValueError(
                f"TerAdapt runner expects dict/TensorDict obs; got {type(obs_dict).__name__}"
            )
        short_obs = torch.cat([obs_dict[g] for g in self.policy_short_groups], dim=1)
        long_obs = torch.cat([obs_dict[g] for g in self.policy_long_groups], dim=1)
        critic_obs = torch.cat([obs_dict[g] for g in self.critic_groups], dim=1)
        height_scan = torch.cat([obs_dict[g] for g in self.height_scan_groups], dim=1)
        vel_gt = torch.cat([obs_dict[g] for g in self.vel_groups], dim=1)
        return short_obs, long_obs, critic_obs, height_scan, vel_gt

    # ---- main training loop ----

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs_dict = self.env.get_observations()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        short_obs, long_obs, critic_obs, height_scan, vel_gt = self._extract_observations(obs_dict)
        short_obs, long_obs, critic_obs = short_obs.to(self.device), long_obs.to(self.device), critic_obs.to(self.device)
        height_scan, vel_gt = height_scan.to(self.device), vel_gt.to(self.device)
        self.alg.train_mode()
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        ep_log_buf: dict[str, list[float]] = {}

        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(short_obs, long_obs, critic_obs)
                    next_obs_dict, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    next_short, next_long, next_critic, next_hscan, next_vel = self._extract_observations(next_obs_dict)
                    next_short = next_short.to(self.device)
                    next_long = next_long.to(self.device)
                    next_critic = next_critic.to(self.device)
                    next_hscan = next_hscan.to(self.device)
                    next_vel = next_vel.to(self.device)
                    # Step bookkeeping
                    self.alg.process_env_step(rewards, dones, extras, next_hscan, next_vel)
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0
                    short_obs, long_obs, critic_obs = next_short, next_long, next_critic
                    height_scan, vel_gt = next_hscan, next_vel
                    # Collect Isaac Lab extras["log"] — Curriculum/terrain_levels,
                    # Episode_Reward/*, Episode_Termination/*, etc.
                    log = extras.get("log") if isinstance(extras, dict) else None
                    if log:
                        for k, v in log.items():
                            val = v.item() if hasattr(v, "item") else float(v)
                            ep_log_buf.setdefault(k, []).append(val)
                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_tok_loss,
                mean_vel_loss,
                mean_vq_recon,
                mean_vq_commit,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            locs = {
                "it": it,
                "tot_iter": tot_iter,
                "collect_time": collect_time,
                "learn_time": learn_time,
                "value_loss": mean_value_loss,
                "surrogate_loss": mean_surrogate_loss,
                "tok_loss": mean_tok_loss,
                "vel_loss": mean_vel_loss,
                "vq_recon": mean_vq_recon,
                "vq_commit": mean_vq_commit,
                "rewbuffer": self.rewbuffer,
                "lenbuffer": self.lenbuffer,
                "ep_log": ep_log_buf,
            }
            self.log(locs)
            ep_log_buf = {}

            if (it + 1) % self.cfg["save_interval"] == 0 or it == tot_iter - 1:
                self.save(os.path.join(self.log_dir, f"model_{it+1}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.cfg["num_steps_per_env"] * self.env.num_envs
        self.tot_time += locs["collect_time"] + locs["learn_time"]
        fps = int(self.cfg["num_steps_per_env"] * self.env.num_envs / (locs["collect_time"] + locs["learn_time"] + 1e-6))
        mean_rew = statistics.mean(locs["rewbuffer"]) if len(locs["rewbuffer"]) > 0 else 0.0
        mean_len = statistics.mean(locs["lenbuffer"]) if len(locs["lenbuffer"]) > 0 else 0.0

        if self.writer is not None:
            self.writer.add_scalar("Loss/value", locs["value_loss"], locs["it"])
            self.writer.add_scalar("Loss/surrogate", locs["surrogate_loss"], locs["it"])
            self.writer.add_scalar("Loss/token_CE", locs["tok_loss"], locs["it"])
            self.writer.add_scalar("Loss/velocity_MSE", locs["vel_loss"], locs["it"])
            self.writer.add_scalar("Loss/vq_recon", locs["vq_recon"], locs["it"])
            self.writer.add_scalar("Loss/vq_commit", locs["vq_commit"], locs["it"])
            self.writer.add_scalar("Performance/total_fps", fps, locs["it"])
            self.writer.add_scalar("Train/mean_reward", mean_rew, locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", mean_len, locs["it"])
            # Isaac Lab curriculum / episode-reward / termination breakdown
            ep_log = locs.get("ep_log") or {}
            for k, values in ep_log.items():
                if not values:
                    continue
                self.writer.add_scalar(k, sum(values) / len(values), locs["it"])

        # Pull a couple of curriculum scalars for the stdout tail (main one: terrain_levels)
        ep_log = locs.get("ep_log") or {}
        terr = ep_log.get("Curriculum/terrain_levels") or ep_log.get("Metrics/terrain_levels")
        terr_str = f"  terr={sum(terr)/len(terr):.2f}" if terr else ""

        print("=" * width)
        print(
            f"Iter {locs['it']+1}/{locs['tot_iter']}  "
            f"fps={fps}  "
            f"val={locs['value_loss']:.3f}  sur={locs['surrogate_loss']:.3f}  "
            f"tok={locs['tok_loss']:.3f}  vel={locs['vel_loss']:.3f}  "
            f"vq_rec={locs['vq_recon']:.3f}  vq_com={locs['vq_commit']:.3f}  "
            f"rew={mean_rew:.2f}  len={mean_len:.1f}{terr_str}"
        )

    def save(self, path: str, infos: Optional[dict] = None):
        state = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(state, path)

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
        policy = self.actor_critic.act_inference
        return policy
