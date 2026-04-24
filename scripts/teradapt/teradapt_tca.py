"""TCA: Terrain Codebook Alignment module (VQ-VAE on height_scan)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    return {"elu": nn.ELU(), "relu": nn.ReLU(), "silu": nn.SiLU(), "tanh": nn.Tanh()}.get(name, nn.ELU())


def _build_mlp(in_dim, hidden_dims, out_dim, activation="elu"):
    act = get_activation(activation)
    layers, prev = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), act]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class VectorQuantizer(nn.Module):
    """VQ codebook with EMA updates and straight-through estimator.

    - codebook: [num_codes, code_dim]
    - EMA stats: cluster_size [num_codes], ema_weights [num_codes, code_dim]
    - decay=0.99, eps=1e-5 for Laplace smoothing
    """

    def __init__(self, num_codes=256, code_dim=16, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps

        embed = torch.randn(num_codes, code_dim)
        self.register_buffer("codebook", embed)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_weights", embed.clone())

    @torch.no_grad()
    def _ema_update(self, z_flat: torch.Tensor, onehot: torch.Tensor):
        # z_flat: [B, code_dim]; onehot: [B, num_codes]
        cluster_counts = onehot.sum(dim=0)            # [num_codes]
        embed_sum = onehot.T @ z_flat                 # [num_codes, code_dim]
        self.cluster_size.mul_(self.decay).add_(cluster_counts, alpha=1 - self.decay)
        self.ema_weights.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
        self.codebook.copy_(self.ema_weights / smoothed.unsqueeze(-1))

    def forward(self, z: torch.Tensor):
        # z: [B, code_dim]
        # distances = ||z - e_k||^2 = ||z||^2 + ||e_k||^2 - 2 z·e_k
        z2 = (z ** 2).sum(dim=-1, keepdim=True)           # [B,1]
        e2 = (self.codebook ** 2).sum(dim=-1)             # [num_codes]
        ze = z @ self.codebook.T                          # [B, num_codes]
        d = z2 + e2.unsqueeze(0) - 2 * ze                 # [B, num_codes]
        indices = d.argmin(dim=-1)                        # [B]
        z_q = self.codebook[indices]                      # [B, code_dim]

        if self.training:
            onehot = F.one_hot(indices, num_classes=self.num_codes).float()
            self._ema_update(z.detach(), onehot)

        commit_loss = F.mse_loss(z, z_q.detach())
        # straight-through
        z_q_st = z + (z_q - z).detach()
        return z_q_st, indices, commit_loss


class TCAModule(nn.Module):
    """Terrain Codebook Alignment = TerrainEncoder + VQ + TerrainDecoder."""

    def __init__(
        self,
        input_dim=187,
        latent_dim=16,
        num_codes=256,
        enc_hidden=[64, 32],
        dec_hidden=[32, 64],
        activation="elu",
    ):
        super().__init__()
        self.encoder = _build_mlp(input_dim, enc_hidden, latent_dim, activation)
        self.quantizer = VectorQuantizer(num_codes, latent_dim)
        self.decoder = _build_mlp(latent_dim, dec_hidden, input_dim, activation)

    def encode(self, h: torch.Tensor):
        return self.encoder(h)

    def forward(self, h: torch.Tensor):
        z = self.encoder(h)
        z_q, indices, commit = self.quantizer(z)
        h_hat = self.decoder(z_q)
        recon = F.mse_loss(h_hat, h)
        return z_q, indices, recon, commit
