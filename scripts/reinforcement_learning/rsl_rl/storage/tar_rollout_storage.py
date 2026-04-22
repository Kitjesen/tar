"""Rollout storage for TAR official — tracks next_critic_obs for contrastive target."""

from __future__ import annotations
import torch


class TARRolloutStorage:
    """PPO rollout storage with an extra `next_critic_obs` buffer.

    The TAR contrastive loss needs next-step critic observations to compute
    next_z_c = encoder_critic(next_critic_obs). So storage tracks both
    critic_obs[t] and next_critic_obs[t] = critic_obs[t+1].
    """

    class Transition:
        def __init__(self):
            self.obs = None
            self.critic_obs = None
            self.next_critic_obs = None
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
        obs_shape,
        critic_obs_shape,
        actions_shape,
        device: str = "cpu",
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env

        self.obs = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=device)
        self.critic_obs = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=device)
        self.next_critic_obs = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=device)
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

    def add_transitions(self, transition: "TARRolloutStorage.Transition"):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.obs[self.step].copy_(transition.obs)
        self.critic_obs[self.step].copy_(transition.critic_obs)
        self.next_critic_obs[self.step].copy_(transition.next_critic_obs)
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
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + not_terminal * gamma * next_values - self.values[step]
            advantage = delta + not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)

        obs = self.obs.flatten(0, 1)
        crit = self.critic_obs.flatten(0, 1)
        ncrit = self.next_critic_obs.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        lp = self.actions_log_prob.flatten(0, 1)
        adv = self.advantages.flatten(0, 1)
        mu = self.mu.flatten(0, 1)
        sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                s = i * mini_batch_size
                e = (i + 1) * mini_batch_size
                bi = indices[s:e]
                yield (
                    obs[bi], crit[bi], ncrit[bi],
                    actions[bi], values[bi], adv[bi], returns[bi], lp[bi], mu[bi], sigma[bi],
                )
