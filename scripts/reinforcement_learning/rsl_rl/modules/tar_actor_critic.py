"""TARActorCritic for the current Thunder TAR MLP variant.

This module implements the fixed-window MLP actor/critic path currently used in
this repository. It borrows the TAR auxiliary-loss structure from TARLoco and
adapts the observation slicing to Thunder's 16-DOF layout.

This is not the recurrent TAR student from the paper's main LSTM-based setup.

Architecture (current repo defaults):
  encoder:        MLP(actor_obs = hist_len*prop_dim -> hidden [256,128,64] -> latent 45)
  encoder_critic: MLP(critic_obs -> hidden [256,128,64] -> latent 45)
  trans:          MLP(latent + action -> hidden [64] -> latent 45)
  vel_estimator:  MLP(latent + hist_short*prop_dim -> hidden [64,32] -> 3)
  actor:          MLP(prop_current + latent + 3 -> hidden [512,256,128] -> action)
  critic:         MLP(prop_current + latent + 3 -> hidden [512,256,128] -> 1)

Inputs:
  observations (actor view): [B, hist_len, prop_dim=57] flattened to [B, hist_len*57]
  critic_observations:       [B, hist_len, critic_dim] where
                             [0:3]       = base_lin_vel (privileged)
                             [3:60]      = proprio (57 dims same as actor)
                             [60:]       = other privileged (height_scan, contacts, friction, mass)

Thunder vs Go2 mapping:
  Go2: prop_dim=45, critic prop slice [3:48], action=12
  Thunder: prop_dim=57, critic prop slice [3:60], action=16
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(act_name: str) -> nn.Module:
    m = {
        "elu": nn.ELU(), "selu": nn.SELU(), "relu": nn.ReLU(),
        "silu": nn.SiLU(), "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
    }
    return m.get(act_name, nn.ELU())


def _mlp(in_dim: int, hidden: list, out_dim: int, activation: str = "elu") -> nn.Sequential:
    act = get_activation(activation)
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), act]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class TARActorCritic(nn.Module):
    """TAR MLP actor-critic used by the current Thunder training stack."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,          # hist_len * prop_dim (flattened actor obs)
        num_critic_obs: int,         # full critic dim per timestep (e.g., 60 + height_scan etc.)
        num_actions: int,            # 16 for Thunder
        num_hist: int,               # actor history length (e.g., 10)
        num_hist_short: int = 4,     # short history for vel estimator (repo default)
        latent_dims: int = 45,       # current repo default
        prop_dim: int = 57,          # proprio per-frame dim (Thunder 16 DOF)
        critic_vel_slice: tuple = (0, 3),    # slice of base_lin_vel in critic obs
        critic_prop_slice: tuple = (3, 60),  # slice of proprio in critic obs (Thunder 57 dims)
        actor_hidden_dims: list = [512, 256, 128],
        critic_hidden_dims: list = [512, 256, 128],
        mlp_encoder_dims: list = [256, 128, 64],
        vel_encoder_dims: list = [64, 32],
        trans_hidden_dims: list = [64],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        clip_action: float = 100.0,
        squash_mode: str = "clip",
        **kwargs,
    ):
        if kwargs:
            print(f"[TAR] unexpected kwargs (ignored): {list(kwargs)}")
        super().__init__()

        self.num_hist = int(num_hist)
        self.num_hist_short = int(num_hist_short)
        self.prop_dim = int(prop_dim)
        self.num_obs_h1 = int(prop_dim)  # single-frame proprio
        self.num_latents = int(latent_dims)
        self.num_actions = int(num_actions)
        self.critic_vel_slice = critic_vel_slice
        self.critic_prop_slice = critic_prop_slice
        self.clip_action = float(clip_action)
        self.squash_mode = squash_mode

        # Assertions
        assert num_actor_obs == num_hist * prop_dim, \
            f"num_actor_obs={num_actor_obs} != num_hist={num_hist} * prop_dim={prop_dim}"
        assert critic_prop_slice[1] - critic_prop_slice[0] == prop_dim, \
            f"critic_prop_slice {critic_prop_slice} width != prop_dim={prop_dim}"
        assert critic_vel_slice[1] - critic_vel_slice[0] == 3, \
            f"critic_vel_slice {critic_vel_slice} must be width 3 (base_lin_vel)"

        # --- Encoders ---
        self.encoder = _mlp(num_actor_obs, mlp_encoder_dims, latent_dims, activation)
        # encoder_critic takes the full critic obs per-timestep (flattened over hist)
        # For simplicity we assume critic_obs is flat [B, num_critic_obs_flat] where
        # num_critic_obs_flat already accounts for history if used.
        self.encoder_critic = _mlp(num_critic_obs, mlp_encoder_dims, latent_dims, activation)

        # --- Dynamics (trans) ---
        self.trans = _mlp(latent_dims + num_actions, trans_hidden_dims, latent_dims, activation)

        # --- Velocity estimator: cat(z, hist_short_flat) -> 3 ---
        self.vel_estimator = _mlp(
            latent_dims + prop_dim * num_hist_short,
            vel_encoder_dims,
            3,
            activation,
        )

        # --- Actor: cat(prop_current, z, vel) -> action ---
        actor_in_dim = prop_dim + latent_dims + 3
        self.actor = _mlp(actor_in_dim, actor_hidden_dims, num_actions, activation)
        self.critic = _mlp(actor_in_dim, critic_hidden_dims, 1, activation)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        print("\n[TAR] TARActorCritic initialized:")
        print(f"   hist_len / hist_short:  {self.num_hist} / {self.num_hist_short}")
        print(f"   prop_dim (single frame): {prop_dim}")
        print(f"   latent_dims:             {latent_dims}")
        print(f"   actor_in:                {actor_in_dim} = {prop_dim}(prop) + {latent_dims}(z) + 3(vel)")
        print(f"   critic_in (encoder input): {num_critic_obs}")
        print(f"   critic vel/prop slices:  vel{critic_vel_slice}, prop{critic_prop_slice}")
        print(f"   action_dim:              {num_actions}")

    # ------- distribution helpers -------

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

    def update_distribution(self, mean: torch.Tensor):
        std = self.std.clamp(min=0.01, max=1.0).expand_as(mean)
        self.distribution = Normal(mean, std)

    # ------- observation extraction -------

    def _as_2d_hist(self, observations: torch.Tensor) -> torch.Tensor:
        """If observations is flat [B, hist*prop], reshape to [B, hist, prop].
        If already 3D [B, hist, prop], return as-is.
        """
        if observations.dim() == 2:
            return observations.view(observations.shape[0], self.num_hist, self.prop_dim)
        return observations

    def extract(self, observations: torch.Tensor):
        """Return (hist[B,H,P], prop_current[B,P], hist_short[B,Hs,P])."""
        hist = self._as_2d_hist(observations)                 # [B, H, P]
        prop = hist[:, -1, :]                                 # [B, P] (most recent)
        hist_short = hist[:, -self.num_hist_short:, :]        # [B, Hs, P]
        return hist, prop, hist_short

    def extract_critic(self, critic_observations: torch.Tensor):
        """Return (full_critic_obs_flat[B,D], prop[B,P], vel[B,3]).

        For Thunder, critic_observations can be either:
          - [B, hist, critic_dim] with history stacked, or
          - [B, critic_dim] single-frame.
        We treat the last frame as the current observation.
        """
        if critic_observations.dim() == 3:
            # Single-frame critic obs per step; use last frame
            current = critic_observations[:, -1, :]
        else:
            current = critic_observations                       # [B, D]
        vs, ve = self.critic_vel_slice
        ps, pe = self.critic_prop_slice
        vel = current[:, vs:ve]                                 # [B, 3]
        prop = current[:, ps:pe]                                # [B, P]
        # Encoder_critic sees the full flat critic obs (may be hist-concatenated upstream)
        full_flat = critic_observations.view(critic_observations.shape[0], -1)
        return full_flat, prop, vel

    # ------- encoders -------

    def encode(self, observations: torch.Tensor):
        """Actor path: (z, vel_hat) from actor observations."""
        hist, _, hist_short = self.extract(observations)
        z = self.encoder(hist.reshape(hist.shape[0], -1))
        hs_flat = hist_short.reshape(hist_short.shape[0], -1)
        vel_hat = self.vel_estimator(torch.cat([z, hs_flat], dim=-1))
        return z, vel_hat

    def encode_critic(self, critic_observations: torch.Tensor):
        """Critic path: (z_c, vel_from_priv) where vel is taken from privileged obs."""
        full_flat, _, vel = self.extract_critic(critic_observations)
        z_c = self.encoder_critic(full_flat)
        return z_c, vel

    # ------- inference / actions -------

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        _, prop, _ = self.extract(observations)
        z, vel_hat = self.encode(observations)
        actor_in = torch.cat([prop, z.detach(), vel_hat.detach()], dim=-1)
        mean = self.actor(actor_in)
        return mean

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        mean = self.act_inference(observations, **kwargs)
        self.update_distribution(mean)
        if self.squash_mode == "tanh":
            return torch.tanh(self.distribution.sample()) * self.clip_action
        return torch.clamp(self.distribution.sample(), -self.clip_action, self.clip_action)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        _, prop, vel = self.extract_critic(critic_observations)
        z_c, _ = self.encode_critic(critic_observations)
        critic_in = torch.cat([prop, z_c, vel], dim=-1)
        return self.critic(critic_in)

    # ------- contrastive TAR aux loss (hinge + dynamics + MSE vel) -------

    def compute_tar_loss(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        actions: torch.Tensor,
        num_envs: int,
    ):
        """Official TARLoco aux loss:
          pred_next_z = trans(z_a, actions)
          next_z_c = encoder_critic(next_critic_obs)
          neg_z = next_z_c shuffled (excluding same-env slots)
          pos_loss = ||next_z_c - pred_next_z||^2 (mean over batch, sum over z dim)
          neg_loss = max(0, 1 - ||next_z_c - neg_z||^2)
          triplet = pos_loss + neg_loss
          vel_loss = MSE(vel_hat_student, vel_from_critic)

        Args:
            obs:            actor observations (flat or 3D)
            critic_obs:     current critic obs
            next_critic_obs: next-step critic obs (from rollout)
            actions:        actions taken (detached)
            num_envs:       total parallel envs (so we can avoid selecting same-env neg)

        Returns: dict with {'triplet', 'vel_mse', 'pos', 'neg'}
        """
        # Student path
        z_a, vel_hat = self.encode(obs)
        pred_next_z = self.trans(torch.cat([z_a, actions], dim=-1))

        # Critic path (current + next)
        _, _, vel_c = self.extract_critic(critic_obs)
        next_z_c, _ = self.encode_critic(next_critic_obs)

        # Generate negative indices avoiding multiples of num_envs (same env wrap-around)
        B = next_z_c.shape[0]
        with torch.no_grad():
            device = next_z_c.device
            idx = torch.arange(B, device=device)
            offsets = torch.randint(1, B, (B,), device=device)
            neg_idx = (idx + offsets) % B
            # Avoid same-env negatives (idx where (idx - neg_idx) % num_envs == 0)
            invalid = ((neg_idx - idx) % num_envs == 0)
            while invalid.any():
                offsets[invalid] = torch.randint(1, B, (int(invalid.sum()),), device=device)
                neg_idx = (idx + offsets) % B
                invalid = ((neg_idx - idx) % num_envs == 0)
        next_neg_z = next_z_c[neg_idx].detach()

        pos_diff = next_z_c - pred_next_z
        neg_diff = next_z_c - next_neg_z
        pos_loss = (pos_diff.pow(2)).sum(dim=-1).mean()
        neg_loss = (neg_diff.pow(2)).sum(dim=-1)
        neg_loss = torch.clamp(1.0 - neg_loss, min=0.0).mean()
        triplet = pos_loss + neg_loss

        # Velocity MSE (student vel_hat vs critic vel which is GT from privileged obs)
        if vel_c.shape == vel_hat.shape:
            vel_mse = torch.nn.functional.mse_loss(vel_hat, vel_c.detach())
        else:
            vel_mse = torch.tensor(0.0, device=vel_hat.device)

        return {
            "triplet": triplet,
            "pos": pos_loss.detach(),
            "neg": neg_loss.detach(),
            "vel_mse": vel_mse,
        }
