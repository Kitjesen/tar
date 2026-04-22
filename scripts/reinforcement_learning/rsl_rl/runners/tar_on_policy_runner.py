# Copyright (c) 2024-2026 Inovxio (穹沛科技)
# SPDX-License-Identifier: Apache-2.0
"""TAROnPolicyRunner -- on-policy runner for TAR training.

Mirrors HIMOnPolicyRunner but:
- Uses TARActorCritic / TARPPO / TARRolloutStorage
- Extracts an extra 'estimator_targets' obs group and passes it to process_env_step
- Logs triplet_loss and estimator_loss instead of swap_loss
"""

import os
import time
import statistics
from collections import deque

import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from algorithms.tar_ppo import TARPPO
from modules.tar_actor_critic import TARActorCritic
from utils.observation_reshaper import reshape_isaac_to_him, THUNDER_HIST_POLICY_DIMS


class TAROnPolicyRunner:
    """On-policy runner for TAR (Teacher-Aligned Representations) training."""

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str = None,
        device: str = "cpu",
    ):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Resolve obs groups from config, with sensible defaults
        obs_groups_cfg = self.cfg.get("obs_groups", {
            "policy": ["policy"],
            "critic": ["critic", "height_scan_group"],
            "estimator": ["estimator_targets"],
        })
        self.policy_groups = obs_groups_cfg.get("policy", ["policy"])
        self.critic_groups = obs_groups_cfg.get("critic", ["critic"])
        self.estimator_groups = obs_groups_cfg.get("estimator", ["estimator_targets"])

        # Get observation dimensions
        obs = self.env.get_observations()

        if isinstance(obs, (TensorDict, dict)):
            actor_obs_dim = sum(obs[g].shape[1] for g in self.policy_groups if g in obs)
            critic_obs_dim = sum(obs[g].shape[1] for g in self.critic_groups if g in obs)
            estimator_dim = sum(obs[g].shape[1] for g in self.estimator_groups if g in obs)
        else:
            actor_obs_dim = obs.shape[1]
            critic_obs_dim = obs.shape[1]
            estimator_dim = self.policy_cfg.get("num_estimator_targets", 6)

        self.num_actor_obs = actor_obs_dim
        self.num_critic_obs = critic_obs_dim
        self.num_estimator_targets = estimator_dim if estimator_dim > 0 else self.policy_cfg.get("num_estimator_targets", 6)

        # History configuration
        history_len = train_cfg.get("history_len", 15)
        self.history_len = history_len
        self.num_one_step_obs = actor_obs_dim // history_len

        # Observation reshaping
        self.need_reshape = history_len > 1
        self.policy_dims = train_cfg.get("policy_dims", THUNDER_HIST_POLICY_DIMS)
        if sum(self.policy_dims) != self.num_one_step_obs:
            print(f"   [TAR] Warning: policy_dims sum ({sum(self.policy_dims)}) != one_step_obs ({self.num_one_step_obs})")
            print(f"   [TAR] Disabling observation reshaping")
            self.need_reshape = False

        print(f"\n[TAR] TAROnPolicyRunner Configuration:")
        print(f"   - Actor obs dim:         {self.num_actor_obs} ({history_len} frames)")
        print(f"   - Critic obs dim:        {self.num_critic_obs}")
        print(f"   - One-step obs dim:      {self.num_one_step_obs}")
        print(f"   - History length:        {history_len}")
        print(f"   - Estimator targets dim: {self.num_estimator_targets}")
        print(f"   - Policy groups:         {self.policy_groups}")
        print(f"   - Critic groups:         {self.critic_groups}")
        print(f"   - Estimator groups:      {self.estimator_groups}")

        # Build policy_cfg for TARActorCritic
        policy_kwargs = dict(self.policy_cfg)
        policy_kwargs.setdefault("num_estimator_targets", self.num_estimator_targets)
        policy_kwargs.setdefault("z_dim", 45)

        actor_critic = TARActorCritic(
            num_actor_obs=self.num_actor_obs,
            num_critic_obs=self.num_critic_obs,
            num_one_step_obs=self.num_one_step_obs,
            num_actions=self.env.num_actions,
            **policy_kwargs,
        ).to(self.device)

        self.alg: TARPPO = TARPPO(
            actor_critic=actor_critic,
            device=self.device,
            **self.alg_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=(self.num_actor_obs,),
            critic_obs_shape=(self.num_critic_obs,),
            action_shape=(self.env.num_actions,),
            num_estimator_targets=self.num_estimator_targets,
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.env.reset()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs_dict = self.env.get_observations()
        obs, critic_obs, estimator_targets = self._extract_observations(obs_dict)
        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)
        estimator_targets = estimator_targets.to(self.device)

        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        print(f"\n[TAR] Training loop: {self.current_learning_iteration} -> {tot_iter}")

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)

                    obs_dict, rewards, dones, extras = self.env.step(actions)

                    next_obs, next_critic_obs, next_estimator_targets = self._extract_observations(obs_dict)
                    next_obs = next_obs.to(self.device)
                    next_critic_obs = next_critic_obs.to(self.device)
                    next_estimator_targets = next_estimator_targets.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    self.alg.process_env_step(
                        rewards, dones, extras, next_critic_obs,
                        estimator_targets=estimator_targets,
                    )

                    if self.log_dir is not None:
                        if "log" in extras and len(extras["log"]) > 0:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if len(new_ids) > 0:
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                    obs = next_obs
                    critic_obs = next_critic_obs
                    estimator_targets = next_estimator_targets

                stop = time.time()
                collection_time = stop - start

                start = stop
                self.alg.compute_returns(critic_obs)

            loss_dict = self.alg.update()
            mean_value_loss = loss_dict["mean_value_loss"]
            mean_surrogate_loss = loss_dict["mean_surrogate_loss"]
            mean_triplet_loss = loss_dict["mean_triplet_loss"]
            mean_estimator_loss = loss_dict["mean_estimator_loss"]

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                log_dict = locals()
                log_dict["tot_iter"] = tot_iter
                self.log(log_dict)

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _extract_observations(self, obs_dict):
        """Extract actor obs, critic obs, and estimator_targets from obs_dict.

        Returns:
            actor_obs: [B, num_actor_obs]
            critic_obs: [B, num_critic_obs]
            estimator_targets: [B, num_estimator_targets]
        """
        if isinstance(obs_dict, (TensorDict, dict)):
            actor_obs = torch.cat(
                [obs_dict[g] for g in self.policy_groups if g in obs_dict], dim=1
            )
            critic_obs = torch.cat(
                [obs_dict[g] for g in self.critic_groups if g in obs_dict], dim=1
            )
            # estimator_targets: clean GT group (no history stacking)
            est_parts = [obs_dict[g] for g in self.estimator_groups if g in obs_dict]
            if est_parts:
                estimator_targets = torch.cat(est_parts, dim=1)
            else:
                estimator_targets = torch.zeros(
                    actor_obs.shape[0], self.num_estimator_targets,
                    device=actor_obs.device,
                )
        else:
            actor_obs = obs_dict
            critic_obs = obs_dict
            estimator_targets = torch.zeros(
                actor_obs.shape[0], self.num_estimator_targets,
                device=actor_obs.device,
            )

        if self.need_reshape:
            actor_obs = reshape_isaac_to_him(
                actor_obs,
                history_len=self.history_len,
                obs_dims=self.policy_dims,
            )

        return actor_obs, critic_obs, estimator_targets

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"{f' {key}:':>{pad}} {value:.4f}\n"

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env * self.env.num_envs /
            (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/triplet_loss", locs["mean_triplet_loss"], locs["it"])
        self.writer.add_scalar("Loss/estimator_loss", locs["mean_estimator_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        if "tot_iter" in locs:
            total_iter = locs["tot_iter"]
        else:
            total_iter = self.current_learning_iteration + locs.get("num_learning_iterations", 0)

        title_str = f" \033[1m Learning iteration {locs['it']}/{total_iter} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"{'#' * width}\n"
                f"{title_str.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
                f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
                f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
                f"{'Triplet loss:':>{pad}} {locs['mean_triplet_loss']:.4f}\n"
                f"{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"
                f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        else:
            log_string = (
                f"{'#' * width}\n"
                f"{title_str.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
                f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
                f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
                f"{'Triplet loss:':>{pad}} {locs['mean_triplet_loss']:.4f}\n"
                f"{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"
                f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
            )

        log_string += ep_string

        if "tot_iter" in locs:
            remaining_iters = locs["tot_iter"] - locs["it"] - 1
        else:
            remaining_iters = locs.get("num_learning_iterations", 0) - (locs["it"] - self.current_learning_iteration) - 1

        completed_iters = locs["it"] - self.current_learning_iteration + 1

        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
        )

        if completed_iters > 0 and remaining_iters > 0:
            avg_time_per_iter = self.tot_time / completed_iters
            eta = avg_time_per_iter * remaining_iters
            log_string += f"{'ETA:':>{pad}} {eta:.1f}s\n"
        else:
            log_string += f"{'ETA:':>{pad}} N/A\n"

        print(log_string)

    def save(self, path: str, infos: dict = None):
        torch.save({
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }, path)
        print(f"[TAR] Saved checkpoint to: {path}")

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, map_location=self.device)

        if "model_state_dict" not in loaded_dict:
            raise KeyError(f"Checkpoint missing model_state_dict: {path}")
        if "iter" not in loaded_dict:
            raise KeyError(f"Checkpoint missing iter: {path}")

        try:
            self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        except RuntimeError:
            print("[TAR] Strict load failed, doing shape-filtered partial transfer...")
            current_state = self.alg.actor_critic.state_dict()
            loaded_state = loaded_dict["model_state_dict"]
            filtered = {k: v for k, v in loaded_state.items()
                        if k in current_state and v.shape == current_state[k].shape}
            current_state.update(filtered)
            self.alg.actor_critic.load_state_dict(current_state)
            print(f"[TAR] Loaded {len(filtered)}/{len(loaded_state)} params")
            load_optimizer = False

        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

        loaded_iter = loaded_dict.get("iter", 0) or 0
        self.current_learning_iteration = loaded_iter
        print(f"[TAR] Loaded checkpoint from: {path} (iter={loaded_iter})")
        return loaded_dict.get("infos", None)

    def get_inference_policy(self, device: str = None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
