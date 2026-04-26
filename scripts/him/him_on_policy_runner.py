# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os
import statistics
import time
import torch
from collections import deque
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from him.him_ppo import HIMPPO
from him.him_actor_critic import HIMActorCritic
from him.utils.observation_reshaper import THUNDER_HIST_POLICY_DIMS, reshape_isaac_to_him


class HIMOnPolicyRunner:
    """
    On-policy runner for HIM (History-based Implicit Model) training.

    Features:
    - SwAV contrastive learning for latent encoding
    - Asymmetric actor-critic with privileged observations
    - Observation history management
    - TensorBoard logging
    - Checkpoint save/load
    """

    def __init__(self, env, train_cfg: dict, log_dir: str = None, device: str = "cpu"):
        """
        Args:
            env: Isaac Lab environment (wrapped with RslRlVecEnvWrapper)
            train_cfg: Training configuration dictionary
            log_dir: Directory for logging
            device: Device to run on
        """
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Get observation dimensions from environment
        obs = self.env.get_observations()

        # Auto-detect observation groups from environment
        if isinstance(obs, TensorDict):
            # Policy: always use "policy" group (for estimator training)
            self.policy_groups = ["policy"]
            actor_obs_dim = obs["policy"].shape[1]

            # Critic: use all groups EXCEPT policy (privileged information only)
            # policy is for actor/estimator, critic uses privileged observations
            self.critic_groups = [k for k in obs.keys() if k != "policy"]
            critic_obs_dim = sum(obs[g].shape[1] for g in self.critic_groups)
        else:
            self.policy_groups = None
            self.critic_groups = None
            actor_obs_dim = obs.shape[1]
            critic_obs_dim = obs.shape[1]

        self.num_actor_obs = actor_obs_dim
        self.num_critic_obs = critic_obs_dim

        # History configuration
        history_len = train_cfg.get("history_len", 6)
        self.history_len = history_len
        self.num_one_step_obs = actor_obs_dim // history_len

        # Debug: Print observation groups
        if isinstance(obs, TensorDict):
            print(f"\n📊 Observation Groups Configuration:")
            print(f"   - Policy groups: {self.policy_groups} -> {actor_obs_dim} dims")
            print(f"   - Critic groups: {self.critic_groups} -> {critic_obs_dim} dims")
            for group in obs.keys():
                print(f"     • {group}: {obs[group].shape[1]} dims")

        # Observation reshaping: Isaac Lab outputs per-variable order, HIM needs per-timestep order
        # Only needed when history_len > 1
        self.need_reshape = history_len > 1
        self.policy_dims = train_cfg.get("policy_dims", THUNDER_HIST_POLICY_DIMS)

        print(f"\n📊 HIMOnPolicyRunner Configuration:")
        print(f"   - Actor obs dim: {self.num_actor_obs} ({history_len} frames)")
        print(f"   - Critic obs dim: {self.num_critic_obs}")
        print(f"   - One-step obs dim: {self.num_one_step_obs}")
        print(f"   - History length: {history_len}")
        if self.need_reshape:
            print(f"   - Observation reshaping: Enabled (Isaac Lab → HIM order)")

        # Create actor-critic (expects observations in HIM order after reshaping)
        actor_critic = HIMActorCritic(
            num_actor_obs=self.num_actor_obs,
            num_critic_obs=self.num_critic_obs,
            num_one_step_obs=self.num_one_step_obs,
            num_actions=self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # Create PPO algorithm
        self.alg: HIMPPO = HIMPPO(actor_critic=actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Initialize storage
        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=(self.num_actor_obs,),
            critic_obs_shape=(self.num_critic_obs,),
            action_shape=(self.env.num_actions,),
        )

        # Set environment reference for direct velocity extraction (most accurate)
        self.alg.actor_critic.estimator.set_env_reference(self.env)
        print("✅ Direct velocity extraction from environment enabled")

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # Reset environment
        self.env.reset()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """
        Main training loop.

        Args:
            num_learning_iterations: Number of training iterations
            init_at_random_ep_len: Whether to randomize initial episode lengths
        """
        # Initialize TensorBoard writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # Randomize episode lengths if requested
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Get initial observations
        obs_dict = self.env.get_observations()
        obs, critic_obs = self._extract_observations(obs_dict)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        # Switch to train mode
        self.alg.actor_critic.train()

        # Episode tracking
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        print(
            f"\n🔄 Training loop: {self.current_learning_iteration} → {tot_iter} (total:"
            f" {num_learning_iterations} iterations)"
        )

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # ================================================================
            # Rollout collection
            # ================================================================
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # Select actions
                    actions = self.alg.act(obs, critic_obs)

                    # Environment step
                    obs_dict, rewards, dones, extras = self.env.step(actions)

                    # Extract observations
                    next_obs, next_critic_obs = self._extract_observations(obs_dict)
                    next_obs = next_obs.to(self.device)
                    next_critic_obs = next_critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Process environment step
                    self.alg.process_env_step(rewards, dones, extras, next_critic_obs)

                    # Logging
                    if self.log_dir is not None:
                        # Episode information from Isaac Lab
                        # Isaac Lab stores episode info in extras["log"] when environments reset
                        if "log" in extras and len(extras["log"]) > 0:
                            # Extract reward breakdown and other episode metrics
                            ep_infos.append(extras["log"])

                        # Accumulate rewards and episode lengths
                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        # Record completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if len(new_ids) > 0:
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                    # Update observations for next step
                    obs = next_obs
                    critic_obs = next_critic_obs

                stop = time.time()
                collection_time = stop - start

                # ================================================================
                # Compute returns
                # ================================================================
                start = stop
                self.alg.compute_returns(critic_obs)

            # ================================================================
            # Update policy and estimator
            # ================================================================
            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            # Logging
            if self.log_dir is not None:
                log_dict = locals()
                log_dict["tot_iter"] = tot_iter  # Add total iteration for logging
                self.log(log_dict)

            # Save checkpoint
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        # Save final model
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _extract_observations(self, obs_dict):
        """
        Extract and reshape observations from Isaac Lab format to HIM format.

        Isaac Lab outputs history in per-variable order: [var1(all_time), var2(all_time), ...]
        HIM expects per-timestep order: [all_vars(t-n), ..., all_vars(t-0)]

        Args:
            obs_dict: Observation dictionary (TensorDict or dict)

        Returns:
            actor_obs: Actor observations in HIM format [batch, history_len * obs_dim]
            critic_obs: Critic observations (with privileged info)
        """
        # Extract and concatenate observations
        if isinstance(obs_dict, (TensorDict, dict)):
            # Actor: use policy group(s)
            actor_obs = torch.cat([obs_dict[g] for g in self.policy_groups], dim=1)

            # Critic: use all groups (policy + height_scan_group + any others)
            critic_obs = torch.cat([obs_dict[g] for g in self.critic_groups], dim=1)
        else:
            actor_obs = obs_dict
            critic_obs = obs_dict

        if self.need_reshape:
            actor_obs = reshape_isaac_to_him(actor_obs, history_len=self.history_len, obs_dims=self.policy_dims)

        return actor_obs, critic_obs

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Episode information
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # Handle scalar and zero-dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"{f' {key}:':>{pad}} {value:.4f}\n"

        # Policy statistics
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # TensorBoard logging
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/estimation_loss", locs["mean_estimation_loss"], locs["it"])
        self.writer.add_scalar("Loss/swap_loss", locs["mean_swap_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar(
            "Loss/estimator_learning_rate", self.alg.actor_critic.estimator.learning_rate, locs["it"]
        )

        # Log loss ratio for debugging
        if locs["mean_swap_loss"] > 0:
            loss_ratio = locs["mean_estimation_loss"] / locs["mean_swap_loss"]
            self.writer.add_scalar("Loss/estimation_swap_ratio", loss_ratio, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        # Console logging
        # Use tot_iter if available (for correct display when resuming), otherwise calculate
        if "tot_iter" in locs:
            total_iter = locs["tot_iter"]
        else:
            total_iter = self.current_learning_iteration + locs.get("num_learning_iterations", 0)
        title_str = f" \033[1m Learning iteration {locs['it']}/{total_iter} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"{'#' * width}\n"
                f"{title_str.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning"
                f" {locs['learn_time']:.3f}s)\n"
                f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
                f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
                f"{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"
                f"{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"
                f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        else:
            log_string = (
                f"{'#' * width}\n"
                f"{title_str.center(width, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning"
                f" {locs['learn_time']:.3f}s)\n"
                f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
                f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
                f"{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"
                f"{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"
                f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
            )

        log_string += ep_string
        # Calculate remaining iterations for ETA
        if "tot_iter" in locs:
            remaining_iters = locs["tot_iter"] - locs["it"] - 1
        else:
            remaining_iters = (
                locs.get("num_learning_iterations", 0) - (locs["it"] - self.current_learning_iteration) - 1
            )

        # Calculate completed iterations in this session
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
        """
        Save checkpoint.

        Args:
            path: Path to save checkpoint
            infos: Additional information to save
        """
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "estimator_optimizer_state_dict": self.alg.actor_critic.estimator.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )
        print(f"💾 Saved checkpoint to: {path}")

    def load(self, path: str, load_optimizer: bool = True):
        """
        Load checkpoint.

        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state

        Returns:
            infos: Additional information from checkpoint
        """
        loaded_dict = torch.load(path, map_location=self.device)

        # Verify required keys exist
        if "model_state_dict" not in loaded_dict:
            raise KeyError(f"Checkpoint missing 'model_state_dict' key: {path}")
        if "iter" not in loaded_dict:
            raise KeyError(f"Checkpoint missing 'iter' key: {path}")

        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])

        if load_optimizer:
            if "optimizer_state_dict" in loaded_dict:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            else:
                print("⚠️  Warning: Checkpoint missing 'optimizer_state_dict', skipping optimizer load")

            if "estimator_optimizer_state_dict" in loaded_dict:
                self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
            else:
                print(
                    "⚠️  Warning: Checkpoint missing 'estimator_optimizer_state_dict', skipping estimator optimizer load"
                )

        # Load iteration number
        loaded_iter = loaded_dict["iter"]
        if loaded_iter is None or loaded_iter < 0:
            print(f"⚠️  Warning: Invalid iteration number in checkpoint: {loaded_iter}, using 0")
            loaded_iter = 0

        self.current_learning_iteration = loaded_iter
        print(f"📥 Loaded checkpoint from: {path}")
        print(f"   - Iteration: {self.current_learning_iteration}")
        print(f"   - Model state: ✅")
        if load_optimizer:
            print(f"   - Optimizer state: ✅")

        return loaded_dict.get("infos", None)

    def get_inference_policy(self, device: str = None):
        self.alg.actor_critic.eval()  # Switch to evaluation mode
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
