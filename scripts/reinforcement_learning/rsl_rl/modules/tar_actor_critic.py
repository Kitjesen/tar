# Copyright (c) 2024-2026 Inovxio (穹沛科技)
# SPDX-License-Identifier: Apache-2.0
"""TARActorCritic -- independent TAR (arXiv:2503.20839) implementation.

Independent from existing HIM SwAV codebase (him_actor_critic.py, him_estimator.py).
All TAR-specific logic lives in this single module.

Architecture:
  Student encoder: history[obs_dim x H] -> MLP -> Z_S
  Teacher encoder: privileged_obs       -> MLP -> Z_T
  Dynamics model:  [Z_S, action]        -> MLP -> Z_S_next
  Vel estimator:   Z_S                  -> Linear -> est (for MSE loss)
  Actor:           [current_obs, Z_S, est] -> MLP -> action
  Critic:          privileged_obs       -> MLP -> value

Losses (returned by compute_aux_loss):
  L_triplet = clamp(||Z_T - Z_S_next||^2 - ||Z_T - Z_neg||^2 + margin, 0).mean()
  L_estimator = MSE(est, gt_targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def get_activation(act_name: str) -> nn.Module:
    m = {
        "elu": nn.ELU(), "selu": nn.SELU(), "relu": nn.ReLU(),
        "silu": nn.SiLU(), "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
    }
    return m.get(act_name, nn.ELU())


def _build_mlp(in_dim, hidden_dims, out_dim, activation):
    """Simple MLP builder: in -> hidden... -> out."""
    act = get_activation(activation)
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), act]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class TARActorCritic(nn.Module):
    """TAR Actor-Critic with contrastive triplet teacher alignment."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_one_step_obs: int,
        num_actions: int,
        num_estimator_targets: int = 6,
        z_dim: int = 45,
        actor_hidden_dims: list = [512, 256, 128],
        critic_hidden_dims: list = [512, 256, 128],
        student_hidden_dims: list = [512, 256, 128],
        teacher_hidden_dims: list = [256, 128],
        dynamics_hidden_dims: list = [64],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        triplet_margin: float = 0.5,
        triplet_coef: float = 1.0,
        estimator_coef: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            print(f"TARActorCritic unexpected kwargs (ignored): {list(kwargs)}")
        super().__init__()

        self.history_size = num_actor_obs // num_one_step_obs
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs
        self.num_actions = num_actions
        self.num_estimator_targets = num_estimator_targets
        self.z_dim = z_dim
        self.triplet_margin = triplet_margin
        self.triplet_coef = triplet_coef
        self.estimator_coef = estimator_coef

        self.student_encoder = _build_mlp(num_actor_obs, student_hidden_dims, z_dim, activation)
        self.teacher_encoder = _build_mlp(num_critic_obs, teacher_hidden_dims, z_dim, activation)
        self.dynamics_model = _build_mlp(z_dim + num_actions, dynamics_hidden_dims, z_dim, activation)
        self.vel_estimator = nn.Linear(z_dim, num_estimator_targets)

        actor_in_dim = num_one_step_obs + z_dim + num_estimator_targets
        self.actor = _build_mlp(actor_in_dim, actor_hidden_dims, num_actions, activation)
        self.critic = _build_mlp(num_critic_obs, critic_hidden_dims, 1, activation)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        print("\n[TAR] TARActorCritic initialized:")
        print(f"   - history_size:     {self.history_size} (num_actor_obs={num_actor_obs}, obs_dim={num_one_step_obs})")
        print(f"   - z_dim:            {z_dim}")
        print(f"   - actor input:      {actor_in_dim} = {num_one_step_obs}(obs) + {z_dim}(Z_S) + {num_estimator_targets}(est)")
        print(f"   - critic input:     {num_critic_obs}")
        print(f"   - num_estimator_targets: {num_estimator_targets}")
        print(f"   - triplet margin:   {triplet_margin}, coef: {triplet_coef}")
        print(f"   - estimator coef:   {estimator_coef}")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _student_path(self, obs_history: torch.Tensor):
        """history -> Z_S -> (vel_est, actor_input -> mean)."""
        Z_S = self.student_encoder(obs_history)
        vel_est = self.vel_estimator(Z_S)
        current_obs = obs_history[:, -self.num_one_step_obs:]
        actor_in = torch.cat([current_obs, Z_S.detach(), vel_est.detach()], dim=-1)
        mean = self.actor(actor_in)
        return mean, Z_S, vel_est

    def update_distribution(self, obs_history: torch.Tensor):
        mean, _, _ = self._student_path(obs_history)
        std = self.std.clamp(min=0.01, max=1.0).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs_history: torch.Tensor, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def act_inference(self, obs_history: torch.Tensor):
        mean, _, _ = self._student_path(obs_history)
        return mean

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_obs: torch.Tensor, **kwargs):
        return self.critic(critic_obs)

    def compute_aux_loss(
        self,
        obs_history: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        gt_targets: torch.Tensor,
    ):
        """Compute TAR triplet + velocity estimator MSE.

        Args:
            obs_history: [B, num_actor_obs]
            critic_obs:  [B, num_critic_obs]
            actions:     [B, num_actions]
            gt_targets:  [B, num_estimator_targets]

        Returns:
            triplet_loss (scalar tensor)
            estimator_loss (scalar tensor)
            total_aux (scalar tensor)
        """
        Z_S = self.student_encoder(obs_history)
        vel_est = self.vel_estimator(Z_S)

        Z_T = self.teacher_encoder(critic_obs)

        Z_S_next = self.dynamics_model(torch.cat([Z_S, actions], dim=-1))

        perm = torch.randperm(Z_T.shape[0], device=Z_T.device)
        Z_neg = Z_T[perm]
        d_pos = ((Z_T - Z_S_next) ** 2).sum(dim=-1)
        d_neg = ((Z_T - Z_neg) ** 2).sum(dim=-1)
        triplet_loss = torch.clamp(d_pos - d_neg + self.triplet_margin, min=0.0).mean()

        estimator_loss = F.mse_loss(vel_est, gt_targets)

        total_aux = self.triplet_coef * triplet_loss + self.estimator_coef * estimator_loss
        return triplet_loss, estimator_loss, total_aux
