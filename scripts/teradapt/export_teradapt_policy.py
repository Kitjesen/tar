"""Export TerAdapt policy with a single deployment observation entry.

The training module uses two inference tensors:
  - policy_short: [B, 5 * obs_dim_per_step]
  - policy_long:  [B, 50 * obs_dim_per_step]

Deployment only needs to maintain one long history buffer.  This exporter exposes
that buffer as the only model input and slices the short/long policy inputs
inside the exported graph.
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import torch
import torch.nn as nn


class PolicyExporterTerAdapt(nn.Module):
    """Single-input TerAdapt inference wrapper.

    Input:
        obs_history: [B, long_history_steps, obs_dim_per_step] or
            [B, long_history_steps * obs_dim_per_step]

    Output:
        actions: [B, num_actions]
    """

    def __init__(self, actor_critic: nn.Module):
        super().__init__()
        self.short_enc = copy.deepcopy(actor_critic.short_enc)
        self.long_enc = copy.deepcopy(actor_critic.long_enc)
        self.latent_enc = copy.deepcopy(actor_critic.latent_enc)
        self.vel_head = copy.deepcopy(actor_critic.vel_head)
        self.actor = copy.deepcopy(actor_critic.actor)

        self.obs_dim_per_step = int(actor_critic.obs_dim_per_step)
        self.short_history_steps = int(actor_critic.short_history_steps)
        self.long_history_steps = int(actor_critic.long_history_steps)
        self.num_long_obs = int(actor_critic.num_long_obs)
        self.num_short_obs = int(actor_critic.num_short_obs)

    def _flatten_history(self, obs_history: torch.Tensor) -> torch.Tensor:
        if obs_history.dim() == 3:
            return obs_history.reshape(obs_history.shape[0], self.num_long_obs)
        if obs_history.dim() == 2:
            return obs_history
        raise ValueError(
            "obs_history must be [B, long_history_steps, obs_dim_per_step] "
            "or [B, long_history_steps * obs_dim_per_step]"
        )

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        long_obs = self._flatten_history(obs_history)
        short_obs = long_obs[:, -self.num_short_obs :]

        batch_size = long_obs.shape[0]
        long_reshape = long_obs.view(batch_size, self.long_history_steps, self.obs_dim_per_step)
        long_reshape = long_reshape.transpose(1, 2).contiguous()

        h_s = self.short_enc(short_obs)
        h_l = self.long_enc(long_reshape)
        feat = torch.cat([h_s, h_l], dim=-1)
        l_tilde = self.latent_enc(feat)
        v_hat = self.vel_head(feat)
        current_obs = short_obs[:, -self.obs_dim_per_step :]
        actor_input = torch.cat([current_obs, h_s, h_l, l_tilde, v_hat], dim=-1)
        return self.actor(actor_input)

    def export(self, path: str, filename: str = "policy_teradapt_history.pt") -> None:
        os.makedirs(path, exist_ok=True)
        export_path = os.path.join(path, filename)

        self.to("cpu")
        self.eval()

        traced_script_module = torch.jit.script(self)
        traced_script_module.save(export_path)

        print(f"Exported TerAdapt policy to: {export_path}")
        print(f"  input : obs_history [B, {self.long_history_steps}, {self.obs_dim_per_step}]")
        print(f"  output: actions [B, num_actions]")
        print(f"  size  : {os.path.getsize(export_path) / 1024 / 1024:.2f} MB")


def export_teradapt_policy_as_jit(
    actor_critic: nn.Module,
    path: str,
    filename: str = "policy_teradapt_history.pt",
) -> None:
    exporter = PolicyExporterTerAdapt(actor_critic)
    exporter.export(path, filename)


def export_teradapt_policy_as_onnx(
    actor_critic: nn.Module,
    path: str,
    filename: str = "policy_teradapt_history.onnx",
    input_shape: Optional[tuple[int, int, int]] = None,
) -> None:
    exporter = PolicyExporterTerAdapt(actor_critic)
    exporter.to("cpu")
    exporter.eval()

    if input_shape is None:
        input_shape = (1, exporter.long_history_steps, exporter.obs_dim_per_step)

    dummy_input = torch.zeros(input_shape, dtype=torch.float32)

    os.makedirs(path, exist_ok=True)
    export_path = os.path.join(path, filename)

    torch.onnx.export(
        exporter,
        dummy_input,
        export_path,
        input_names=["obs_history"],
        output_names=["actions"],
        dynamic_axes={"obs_history": {0: "batch_size"}, "actions": {0: "batch_size"}},
        opset_version=11,
        dynamo=False,
    )

    print(f"Exported TerAdapt policy to ONNX: {export_path}")
    print(f"  input : obs_history [B, {exporter.long_history_steps}, {exporter.obs_dim_per_step}]")
    print(f"  output: actions [B, num_actions]")
    print(f"  size  : {os.path.getsize(export_path) / 1024 / 1024:.2f} MB")
