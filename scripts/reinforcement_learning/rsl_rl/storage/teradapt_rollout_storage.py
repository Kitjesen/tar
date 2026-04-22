"""TerAdapt rollout storage - stores short/long obs + height_scan + vel_gt.

Mirrors HIMRolloutStorage / TARRolloutStorage structure; adds TerAdapt-specific fields.
"""

from __future__ import annotations

import torch


class TerAdaptRolloutStorage:
    """PPO rollout storage for TerAdapt architecture.

    Extra tensors per transition:
      - short_obs      [num_transitions_per_env, num_envs, num_short_obs]
      - long_obs       [num_transitions_per_env, num_envs, num_long_obs]
      - critic_obs     [num_transitions_per_env, num_envs, num_critic_obs]
      - height_scan    [num_transitions_per_env, num_envs, num_height_scan]
      - vel_gt         [num_transitions_per_env, num_envs, num_vel_targets]
    Standard PPO tensors: actions, log_probs, rewards, dones, values, returns, advantages, mu, sigma.
    """

    class Transition:
        def __init__(self):
            self.short_obs = None
            self.long_obs = None
            self.critic_obs = None
            self.height_scan = None
            self.vel_gt = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        short_obs_shape,
        long_obs_shape,
        critic_obs_shape,
        actions_shape,
        num_height_scan: int,
        num_vel_targets: int,
        device: str = "cpu",
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.short_obs_shape = short_obs_shape
        self.long_obs_shape = long_obs_shape
        self.critic_obs_shape = critic_obs_shape
        self.actions_shape = actions_shape
        self.num_height_scan = num_height_scan
        self.num_vel_targets = num_vel_targets

        # Core buffers
        self.short_obs = torch.zeros(num_transitions_per_env, num_envs, *short_obs_shape, device=device)
        self.long_obs = torch.zeros(num_transitions_per_env, num_envs, *long_obs_shape, device=device)
        self.critic_obs = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=device)
        self.height_scan = torch.zeros(num_transitions_per_env, num_envs, num_height_scan, device=device)
        self.vel_gt = torch.zeros(num_transitions_per_env, num_envs, num_vel_targets, device=device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.step = 0

    def add_transitions(self, transition: "TerAdaptRolloutStorage.Transition"):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.short_obs[self.step].copy_(transition.short_obs)
        self.long_obs[self.step].copy_(transition.long_obs)
        self.critic_obs[self.step].copy_(transition.critic_obs)
        self.height_scan[self.step].copy_(transition.height_scan)
        self.vel_gt[self.step].copy_(transition.vel_gt)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        short = self.short_obs.flatten(0, 1)
        long = self.long_obs.flatten(0, 1)
        crit = self.critic_obs.flatten(0, 1)
        hscan = self.height_scan.flatten(0, 1)
        vgt = self.vel_gt.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        log_probs = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        mu = self.mu.flatten(0, 1)
        sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    short[batch_idx],
                    long[batch_idx],
                    crit[batch_idx],
                    hscan[batch_idx],
                    vgt[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    log_probs[batch_idx],
                    mu[batch_idx],
                    sigma[batch_idx],
                )
