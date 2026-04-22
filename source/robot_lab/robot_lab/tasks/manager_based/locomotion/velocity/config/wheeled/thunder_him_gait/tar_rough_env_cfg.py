"""Thunder Gait TAR rough - paper-accurate TARLoco adaptation for Thunder.

Follows ammousa/TARLoco observation design for TAR MLP variant:
  - policy observations: 10-frame history, EXCLUDES privileged terms
      [base_ang_vel(3), projected_gravity(3), velocity_commands(3),
       joint_pos(16), joint_vel(16), last_action(16)] = 57 dims per frame
  - critic observations: 1-frame (no history), INCLUDES all privileged
      [base_lin_vel(3), proprio(57), height_scan(187), contacts, friction, mass]
      with base_lin_vel FIRST (matches extract_critic[0:3] convention)

Training aligned to TAR paper:
  - single-stage end-to-end PPO (no teacher pretraining stage)
  - max_iterations: see rsl_rl_tar_cfg (paper uses ~7500 iter for TAR)
  - terrain: Thunder native rough (inherited from thunder_gait)
"""

import isaaclab.envs.mdp as stock_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.thunder_gait.rough_env_cfg import (
    ThunderGaitRoughEnvCfg,
)


@configclass
class ThunderGaitTarRoughEnvCfg(ThunderGaitRoughEnvCfg):
    """TAR (paper-accurate TARLoco port) environment cfg for Thunder 16 DOF."""

    def __post_init__(self):
        super().__post_init__()

        # Policy: 10-frame history (TARLoco num_hist for MLP variant)
        self.observations.policy.history_length = 10
        # Exclude privileged-style terms from policy if inherited
        for name in ("base_lin_vel", "height_scan"):
            if hasattr(self.observations.policy, name):
                delattr(self.observations.policy, name)

        # Critic: single-frame with all privileged info
        self.observations.critic.history_length = 1
        if hasattr(self.observations, "height_scan_group"):
            self.observations.height_scan_group.history_length = 1

        # Rebuild critic with base_lin_vel first (extract_critic[0:3] = vel)
        existing = {}
        for attr_name in list(vars(self.observations.critic).keys()):
            if attr_name.startswith("_"):
                continue
            val = getattr(self.observations.critic, attr_name)
            if isinstance(val, ObsTerm):
                existing[attr_name] = val
        for name in list(existing.keys()):
            delattr(self.observations.critic, name)
        self.observations.critic.base_lin_vel = ObsTerm(func=stock_mdp.base_lin_vel)
        for name, term in existing.items():
            if name == "base_lin_vel":
                continue
            setattr(self.observations.critic, name, term)

        # Parent disable_zero_weight_rewards() only fires when class name matches exactly
        self.disable_zero_weight_rewards()
