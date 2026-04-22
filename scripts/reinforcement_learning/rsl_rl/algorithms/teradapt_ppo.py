# Copyright (c) 2024-2026 Inovxio (穹沛科技)
# SPDX-License-Identifier: Apache-2.0
"""TerAdapt PPO algorithm with VQ + token + vel auxiliary losses."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from modules.teradapt_actor_critic import TerAdaptActorCritic
from storage.teradapt_rollout_storage import TerAdaptRolloutStorage


class TerAdaptPPO:
    """PPO + TerAdapt aux losses (VQ-VAE reconstruction/commit + token CE + velocity MSE)."""

    actor_critic: TerAdaptActorCritic

    def __init__(
        self,
        actor_critic: TerAdaptActorCritic,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = "cpu",
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = TerAdaptRolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        print("\nTerAdaptPPO initialized:")
        print(f"   - Learning epochs: {num_learning_epochs}")
        print(f"   - Mini-batches: {num_mini_batches}")
        print(f"   - Clip param: {clip_param}")
        print(f"   - Learning rate: {learning_rate}")

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        short_obs_shape,
        long_obs_shape,
        critic_obs_shape,
        actions_shape,
        num_height_scan: int,
        num_vel_targets: int,
    ):
        self.storage = TerAdaptRolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            short_obs_shape=short_obs_shape,
            long_obs_shape=long_obs_shape,
            critic_obs_shape=critic_obs_shape,
            actions_shape=actions_shape,
            num_height_scan=num_height_scan,
            num_vel_targets=num_vel_targets,
            device=self.device,
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, short_obs: torch.Tensor, long_obs: torch.Tensor, critic_obs: torch.Tensor):
        self.transition.actions = self.actor_critic.act(short_obs, long_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.short_obs = short_obs
        self.transition.long_obs = long_obs
        self.transition.critic_obs = critic_obs
        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        height_scan: torch.Tensor,
        vel_gt: torch.Tensor,
    ):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.height_scan = height_scan
        self.transition.vel_gt = vel_gt

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs: torch.Tensor):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self) -> Tuple[float, float, float, float, float, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_tok_loss = 0.0
        mean_vel_loss = 0.0
        mean_vq_recon = 0.0
        mean_vq_commit = 0.0
        num_updates = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            short_b,
            long_b,
            critic_b,
            hscan_b,
            vgt_b,
            actions_b,
            values_b,
            advantages_b,
            returns_b,
            old_log_prob_b,
            old_mu_b,
            old_sigma_b,
        ) in generator:
            # Forward student path (updates self.distribution)
            self.actor_critic.act(short_b, long_b)
            actions_log_prob_b = self.actor_critic.get_actions_log_prob(actions_b)
            value_b = self.actor_critic.evaluate(critic_b)
            mu_b = self.actor_critic.action_mean
            sigma_b = self.actor_critic.action_std
            entropy_b = self.actor_critic.entropy

            # KL-based LR schedule
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_b / old_sigma_b + 1e-5)
                        + (torch.square(old_sigma_b) + torch.square(old_mu_b - mu_b))
                        / (2.0 * torch.square(sigma_b))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            # PPO surrogate loss
            ratio = torch.exp(actions_log_prob_b - torch.squeeze(old_log_prob_b))
            surrogate = -torch.squeeze(advantages_b) * ratio
            surrogate_clipped = -torch.squeeze(advantages_b) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = values_b + (value_b - values_b).clamp(-self.clip_param, self.clip_param)
                v_losses = (value_b - returns_b).pow(2)
                v_losses_clipped = (value_clipped - returns_b).pow(2)
                value_loss = torch.max(v_losses, v_losses_clipped).mean()
            else:
                value_loss = (returns_b - value_b).pow(2).mean()

            # TerAdapt aux losses
            total_aux, aux_info = self.actor_critic.compute_aux_loss(
                short_b, long_b, hscan_b, vgt_b
            )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_b.mean()
                + total_aux
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_tok_loss += aux_info["tok_loss"].item()
            mean_vel_loss += aux_info["vel_loss"].item()
            mean_vq_recon += aux_info["vq_recon"].item()
            mean_vq_commit += aux_info["vq_commit"].item()
            num_updates += 1

        if num_updates == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_tok_loss /= num_updates
        mean_vel_loss /= num_updates
        mean_vq_recon /= num_updates
        mean_vq_commit /= num_updates

        self.storage.clear()
        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_tok_loss,
            mean_vel_loss,
            mean_vq_recon,
            mean_vq_commit,
        )
