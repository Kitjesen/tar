"""Offline CPU unit test for TerAdaptActorCritic — no Isaac Sim, no GPU.

Verifies:
  1. Module instantiates with correct dims
  2. act/act_inference/evaluate forward succeeds
  3. compute_aux_loss forward succeeds
  4. Backward propagates, all gradients non-NaN
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from teradapt.teradapt_actor_critic import TerAdaptActorCritic
from teradapt.export_teradapt_policy import PolicyExporterTerAdapt

# Thunder-sized params
obs_dim = 57
short_steps = 5
long_steps = 50
num_actions = 16
num_critic_obs = 264  # typical thunder critic (77) + height_scan (187)
num_height_scan = 187
num_vel_targets = 3

B = 8

print("=== Building TerAdaptActorCritic ===")
model = TerAdaptActorCritic(
    num_actor_obs=obs_dim,
    num_critic_obs=num_critic_obs,
    num_short_obs=obs_dim * short_steps,
    num_long_obs=obs_dim * long_steps,
    num_actions=num_actions,
    num_height_scan=num_height_scan,
    num_vel_targets=num_vel_targets,
    codebook_size=256,
    codebook_dim=16,
    short_history_steps=short_steps,
    long_history_steps=long_steps,
    obs_dim_per_step=obs_dim,
)

print("\n=== Inputs ===")
short_obs = torch.randn(B, obs_dim * short_steps)
long_obs = torch.randn(B, obs_dim * long_steps)
critic_obs = torch.randn(B, num_critic_obs)
height_scan = torch.randn(B, num_height_scan)
vel_gt = torch.randn(B, num_vel_targets)
print(f"short_obs:   {tuple(short_obs.shape)}")
print(f"long_obs:    {tuple(long_obs.shape)}")
print(f"critic_obs:  {tuple(critic_obs.shape)}")
print(f"height_scan: {tuple(height_scan.shape)}")
print(f"vel_gt:      {tuple(vel_gt.shape)}")

print("\n=== act() ===")
actions = model.act(short_obs, long_obs)
log_probs = model.get_actions_log_prob(actions)
print(f"actions:    {tuple(actions.shape)}   expected ({B},{num_actions})")
print(f"log_probs:  {tuple(log_probs.shape)}   expected ({B},)")
print(f"mean[0]:    {model.action_mean[0].detach().numpy()[:4]} ...")
print(f"std[0]:     {model.action_std[0].detach().numpy()[:4]} ...")

print("\n=== act_inference() ===")
with torch.no_grad():
    inf = model.act_inference(short_obs, long_obs)
print(f"act_inference: {tuple(inf.shape)}")

print("\n=== single-input exporter ===")
history = long_obs.view(B, long_steps, obs_dim)
short_from_long = long_obs[:, -(obs_dim * short_steps) :]
with torch.no_grad():
    inf_from_history = model.act_inference(short_from_long, long_obs)
exporter = PolicyExporterTerAdapt(model)
exporter.eval()
with torch.no_grad():
    exported_from_history = exporter(history)
    exported_from_flat = exporter(long_obs)
print(f"exporter(history): {tuple(exported_from_history.shape)}")
print(f"exporter(flat):    {tuple(exported_from_flat.shape)}")
assert torch.allclose(exported_from_history, inf_from_history, atol=1e-6), (
    "history exporter output differs from act_inference"
)
assert torch.allclose(exported_from_flat, inf_from_history, atol=1e-6), (
    "flat exporter output differs from act_inference"
)
scripted_exporter = torch.jit.script(exporter)
with torch.no_grad():
    scripted_out = scripted_exporter(history)
assert torch.allclose(scripted_out, inf_from_history, atol=1e-6), (
    "scripted exporter output differs from act_inference"
)

print("\n=== evaluate() ===")
value = model.evaluate(critic_obs)
print(f"value: {tuple(value.shape)}   expected ({B},1)")

print("\n=== compute_aux_loss() ===")
total_aux, info = model.compute_aux_loss(short_obs, long_obs, height_scan, vel_gt)
print(f"total_aux:  {total_aux.item():.4f}")
for k, v in info.items():
    print(f"  {k}: {v.item():.4f}")

print("\n=== Backward ===")
model.zero_grad()
# combine with fake surrogate + value loss so we simulate the full PPO update
fake_surrogate = -(model.action_mean ** 2).mean()
fake_value_loss = ((model.evaluate(critic_obs) - 0.5) ** 2).mean()
total = fake_surrogate + fake_value_loss + total_aux
total.backward()
nans = [n for n, p in model.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
nan_ct = len(nans)
no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
print(f"params with NaN grad: {nan_ct}")
print(f"params missing grad:  {len(no_grad)}  (first 5: {no_grad[:5]})")

assert nan_ct == 0, f"NaN gradients found in: {nans[:5]}"

print("\nTerAdapt OFFLINE TEST PASSED")
