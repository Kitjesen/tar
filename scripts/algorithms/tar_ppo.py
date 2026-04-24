"""PPO for TAR official — adds TAR contrastive-hinge aux loss."""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from modules.tar_actor_critic import TARActorCritic
from storage.tar_rollout_storage import TARRolloutStorage


class TARPPO:
    """PPO + TAR aux loss (pos² + max(0, 1-d_neg) contrastive hinge + vel MSE)."""

    actor_critic: TARActorCritic

    def __init__(
        self,
        actor_critic: TARActorCritic,
        num_envs: int,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        # TAR aux coefficients
        tar_coef: float = 1.0,
        vel_coef: float = 1.0,
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.num_envs = num_envs

        self.actor_critic = actor_critic.to(self.device)
        self.storage: TARRolloutStorage | None = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = TARRolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.tar_coef = tar_coef
        self.vel_coef = vel_coef

        print(f"\n[TAR] PPO initialized: tar_coef={tar_coef}, vel_coef={vel_coef}")

    def init_storage(self, num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, actions_shape):
        self.storage = TARRolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=obs_shape,
            critic_obs_shape=critic_obs_shape,
            actions_shape=actions_shape,
            device=self.device,
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor):
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.obs = obs
        self.transition.critic_obs = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, extras, next_critic_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_critic_obs = next_critic_obs

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

    def update(self) -> Tuple[float, ...]:
        mean_value = 0.0
        mean_surr = 0.0
        mean_triplet = 0.0
        mean_vel = 0.0
        mean_pos = 0.0
        mean_neg = 0.0
        n = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (obs_b, crit_b, next_crit_b, actions_b, values_b, adv_b, returns_b, old_lp_b, old_mu_b, old_sig_b) in generator:
            # Forward for log_prob + value
            self.actor_critic.act(obs_b)
            lp_b = self.actor_critic.get_actions_log_prob(actions_b)
            v_b = self.actor_critic.evaluate(crit_b)
            mu_b = self.actor_critic.action_mean
            sigma_b = self.actor_critic.action_std
            entropy_b = self.actor_critic.entropy

            # Adaptive KL schedule
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_b / old_sig_b + 1e-5)
                        + (old_sig_b.pow(2) + (old_mu_b - mu_b).pow(2)) / (2.0 * sigma_b.pow(2))
                        - 0.5,
                        dim=-1,
                    ).mean()
                    if kl > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            # PPO surrogate
            ratio = torch.exp(lp_b - torch.squeeze(old_lp_b))
            surr1 = -torch.squeeze(adv_b) * ratio
            surr2 = -torch.squeeze(adv_b) * torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
            surr_loss = torch.max(surr1, surr2).mean()

            # Value loss (clipped)
            if self.use_clipped_value_loss:
                v_clipped = values_b + (v_b - values_b).clamp(-self.clip_param, self.clip_param)
                v_losses = (v_b - returns_b).pow(2)
                v_losses_c = (v_clipped - returns_b).pow(2)
                value_loss = torch.max(v_losses, v_losses_c).mean()
            else:
                value_loss = (returns_b - v_b).pow(2).mean()

            # TAR aux loss
            tar_info = self.actor_critic.compute_tar_loss(
                obs=obs_b,
                critic_obs=crit_b,
                next_critic_obs=next_crit_b,
                actions=actions_b.detach(),
                num_envs=self.num_envs,
            )
            triplet = tar_info["triplet"]
            vel_mse = tar_info["vel_mse"]

            loss = (
                surr_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_b.mean()
                + self.tar_coef * triplet
                + self.vel_coef * vel_mse
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value += value_loss.item()
            mean_surr += surr_loss.item()
            mean_triplet += triplet.item()
            mean_vel += vel_mse.item()
            mean_pos += tar_info["pos"].item()
            mean_neg += tar_info["neg"].item()
            n += 1

        if n == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        self.storage.clear()
        return (
            mean_value / n, mean_surr / n,
            mean_triplet / n, mean_vel / n,
            mean_pos / n, mean_neg / n,
        )
