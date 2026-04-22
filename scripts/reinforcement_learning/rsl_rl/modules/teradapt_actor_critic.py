# Copyright (c) 2024-2026 Inovxio (穹沛科技)
# SPDX-License-Identifier: Apache-2.0
"""TerAdaptActorCritic - dual-horizon proprio + TCA + policy/critic.

Independent from him_/tar_ codebases. TerAdapt (IEEE RA-L 2026, arXiv:2510) core:
  - Short Encoder MLP[128,64]: recent 5-frame history -> h^short[16]
  - Long 1D CNN: 50-frame history -> h^long[16]
  - Latent Encoder MLP[64,32]: cat(h^s,h^l)[32] -> l_tilde[16]
  - Velocity Head MLP[64,32]:  cat(h^s,h^l)[32] -> v_hat[3]
  - Token Classifier MLP[64,128]: l_tilde[16] -> logits[N=256]
  - Actor MLP[512,256,128]: cat(o_t[57], h^s, h^l, l_tilde, v_hat)=109 -> a[16]
  - Critic MLP[512,256,128]: privileged_obs -> V[1]
  - TCA (teacher): height_scan[187] -> VQ indices (256 codes x 16 dims)

Losses (via compute_aux_loss):
  vel_loss = MSE(v_hat, vel_gt[3])
  tok_loss = CE(logits, indices.detach())
  vq_recon + 0.25 * vq_commit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from modules.teradapt_tca import TCAModule, get_activation, _build_mlp


class Long1DCNN(nn.Module):
    """1D CNN over long proprio history, input [B, C=obs_dim, L=seq_len]."""

    def __init__(
        self,
        in_channels: int,
        channels=(32, 32, 32),
        kernels=(8, 5, 5),
        strides=(4, 1, 1),
        seq_len: int = 50,
        out_dim: int = 16,
        activation: str = "elu",
    ):
        super().__init__()
        act = get_activation(activation)
        layers = []
        prev_ch = in_channels
        L = seq_len
        for c, k, s in zip(channels, kernels, strides):
            layers += [nn.Conv1d(prev_ch, c, kernel_size=k, stride=s), act]
            prev_ch = c
            L = (L - k) // s + 1
        if L <= 0:
            raise ValueError(
                f"Long1DCNN output length <= 0 after conv layers (seq_len={seq_len}, channels={channels}, kernels={kernels}, strides={strides})"
            )
        self.conv = nn.Sequential(*layers)
        self.flat_dim = prev_ch * L
        self.head = nn.Linear(self.flat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        x = self.conv(x)
        x = x.flatten(1)
        return self.head(x)


class TerAdaptActorCritic(nn.Module):
    """TerAdapt actor-critic: TCA teacher + dual-horizon student + PPO actor/critic."""

    is_recurrent: bool = False

    def __init__(
        self,
        num_actor_obs: int,        # per-frame obs dim (e.g., 57)
        num_critic_obs: int,       # privileged obs dim
        num_short_obs: int,        # flattened 5*57=285
        num_long_obs: int,         # flattened 50*57=2850
        num_actions: int = 16,
        num_height_scan: int = 187,
        num_vel_targets: int = 3,
        codebook_size: int = 256,
        codebook_dim: int = 16,
        short_history_steps: int = 5,
        long_history_steps: int = 50,
        obs_dim_per_step: int = 57,
        # Architecture hidden dims (TerAdapt Table IV)
        terrain_enc_hidden=(64, 32),
        terrain_dec_hidden=(32, 64),
        short_enc_hidden=(128, 64),
        long_cnn_channels=(32, 32, 32),
        long_cnn_kernels=(8, 5, 5),
        long_cnn_strides=(4, 1, 1),
        latent_enc_hidden=(64, 32),
        vel_head_hidden=(64, 32),
        tok_cls_hidden=(64, 128),
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        # Aux loss coefficients
        vel_coef: float = 1.0,
        tok_coef: float = 1.0,
        vq_coef: float = 1.0,
        vq_commit_beta: float = 0.25,
        **kwargs,
    ) -> None:
        if kwargs:
            print(f"TerAdaptActorCritic unexpected kwargs (ignored): {list(kwargs)}")
        super().__init__()

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_short_obs = num_short_obs
        self.num_long_obs = num_long_obs
        self.num_actions = num_actions
        self.num_height_scan = num_height_scan
        self.num_vel_targets = num_vel_targets
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.short_history_steps = short_history_steps
        self.long_history_steps = long_history_steps
        self.obs_dim_per_step = obs_dim_per_step
        self.vel_coef = vel_coef
        self.tok_coef = tok_coef
        self.vq_coef = vq_coef
        self.vq_commit_beta = vq_commit_beta

        # Basic shape sanity
        assert num_short_obs == short_history_steps * obs_dim_per_step, (
            f"num_short_obs={num_short_obs} != {short_history_steps}*{obs_dim_per_step}"
        )
        assert num_long_obs == long_history_steps * obs_dim_per_step, (
            f"num_long_obs={num_long_obs} != {long_history_steps}*{obs_dim_per_step}"
        )

        # --- TCA (teacher) ---
        self.tca = TCAModule(
            input_dim=num_height_scan,
            latent_dim=codebook_dim,
            num_codes=codebook_size,
            enc_hidden=list(terrain_enc_hidden),
            dec_hidden=list(terrain_dec_hidden),
            activation=activation,
        )

        # --- Short Encoder ---
        self.short_enc = _build_mlp(
            num_short_obs, list(short_enc_hidden), codebook_dim, activation
        )

        # --- Long 1D CNN ---
        self.long_enc = Long1DCNN(
            in_channels=obs_dim_per_step,
            channels=list(long_cnn_channels),
            kernels=list(long_cnn_kernels),
            strides=list(long_cnn_strides),
            seq_len=long_history_steps,
            out_dim=codebook_dim,
            activation=activation,
        )

        # --- Latent Encoder (cat(h^s, h^l)[32] -> 16) ---
        feat_dim = 2 * codebook_dim
        self.latent_enc = _build_mlp(
            feat_dim, list(latent_enc_hidden), codebook_dim, activation
        )

        # --- Velocity Head ---
        self.vel_head = _build_mlp(
            feat_dim, list(vel_head_hidden), num_vel_targets, activation
        )

        # --- Token Classifier (l_tilde -> logits over N codes) ---
        self.tok_cls = _build_mlp(
            codebook_dim, list(tok_cls_hidden), codebook_size, activation
        )

        # --- Actor ---
        actor_in_dim = (
            obs_dim_per_step + codebook_dim + codebook_dim + codebook_dim + num_vel_targets
        )
        self.actor = _build_mlp(
            actor_in_dim, list(actor_hidden_dims), num_actions, activation
        )

        # --- Critic ---
        self.critic = _build_mlp(
            num_critic_obs, list(critic_hidden_dims), 1, activation
        )

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        print("\nTerAdaptActorCritic initialized:")
        print(f"  - obs_dim_per_step:  {obs_dim_per_step}")
        print(f"  - short/long steps:  {short_history_steps} / {long_history_steps}")
        print(f"  - codebook:          {codebook_size} codes x {codebook_dim} dims")
        print(f"  - actor input:       {actor_in_dim} = {obs_dim_per_step}(obs) + {codebook_dim}(h^s) + {codebook_dim}(h^l) + {codebook_dim}(l_tilde) + {num_vel_targets}(v_hat)")
        print(f"  - critic input:      {num_critic_obs}")
        print(f"  - num_actions:       {num_actions}")
        print(f"  - loss coefs:        vel={vel_coef} tok={tok_coef} vq={vq_coef} beta={vq_commit_beta}")

    # --- Required ActorCritic interface ---

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

    # --- Student path ---

    def _reshape_long(self, long_obs: torch.Tensor) -> torch.Tensor:
        # Incoming long_obs: [B, L*C] where C=obs_dim, L=long_history_steps
        # Layout assumed: [t-L+1, ..., t] concatenated per timestep (Isaac Lab default)
        # Reshape to [B, L, C] then transpose to [B, C, L] for Conv1d
        B = long_obs.shape[0]
        x = long_obs.view(B, self.long_history_steps, self.obs_dim_per_step)
        x = x.transpose(1, 2).contiguous()  # [B, C, L]
        return x

    def _student_forward(self, short_obs: torch.Tensor, long_obs: torch.Tensor):
        """short_obs [B, 5*57], long_obs [B, 50*57] -> (mean, h_s, h_l, l_tilde, v_hat)."""
        h_s = self.short_enc(short_obs)          # [B, 16]
        long_reshape = self._reshape_long(long_obs)
        h_l = self.long_enc(long_reshape)        # [B, 16]
        feat = torch.cat([h_s, h_l], dim=-1)     # [B, 32]
        l_tilde = self.latent_enc(feat)          # [B, 16]
        v_hat = self.vel_head(feat)              # [B, 3]
        current_obs = short_obs[:, -self.obs_dim_per_step:]  # [B, 57] - last frame of short
        actor_in = torch.cat([current_obs, h_s, h_l, l_tilde, v_hat], dim=-1)  # [B, 109]
        mean = self.actor(actor_in)
        return mean, h_s, h_l, l_tilde, v_hat

    def update_distribution(self, short_obs: torch.Tensor, long_obs: torch.Tensor):
        mean, _, _, _, _ = self._student_forward(short_obs, long_obs)
        std = self.std.clamp(min=0.01, max=1.0).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, short_obs: torch.Tensor, long_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(short_obs, long_obs)
        return self.distribution.sample()

    def act_inference(self, short_obs: torch.Tensor, long_obs: torch.Tensor) -> torch.Tensor:
        mean, _, _, _, _ = self._student_forward(short_obs, long_obs)
        return mean

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_obs)

    # --- TerAdapt aux loss ---

    def compute_aux_loss(
        self,
        short_obs: torch.Tensor,
        long_obs: torch.Tensor,
        height_scan: torch.Tensor,
        vel_gt: torch.Tensor,
    ):
        """Returns (total_aux_loss, info_dict).

        info_dict keys: tok_loss, vel_loss, vq_recon, vq_commit (all scalar tensors)
        """
        # Teacher path (VQ-VAE on height_scan)
        z_q, indices, recon_loss, commit_loss = self.tca(height_scan)

        # Student path
        _, h_s, h_l, l_tilde, v_hat = self._student_forward(short_obs, long_obs)

        # Token classifier - train student to match teacher's discrete terrain token
        logits = self.tok_cls(l_tilde)                            # [B, num_codes]
        tok_loss = F.cross_entropy(logits, indices.detach())

        # Velocity estimator
        vel_loss = F.mse_loss(v_hat, vel_gt)

        # VQ-VAE total
        vq_total = recon_loss + self.vq_commit_beta * commit_loss

        total_aux = (
            self.vel_coef * vel_loss
            + self.tok_coef * tok_loss
            + self.vq_coef * vq_total
        )

        info = {
            "tok_loss": tok_loss.detach(),
            "vel_loss": vel_loss.detach(),
            "vq_recon": recon_loss.detach(),
            "vq_commit": commit_loss.detach(),
        }
        return total_aux, info
