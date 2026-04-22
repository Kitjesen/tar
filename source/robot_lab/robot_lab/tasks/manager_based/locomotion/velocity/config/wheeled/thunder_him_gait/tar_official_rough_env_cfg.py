# Copyright (c) 2024-2026 Inovxio (穹沛科技)
"""Thunder Gait TAR Official rough -- 10-frame history policy + structured critic obs.

Critic obs layout (Thunder 16 DOF):
  [0:3]   base_lin_vel                       (privileged GT velocity)
  [3:60]  proprio (57 dims, same as actor)   (for TAR prop slice)
  [60:]   height_scan (187) + other privileged (contacts, friction, mass)

This layout matches TARLoco's extract_critic convention where [0:3] = vel and
[3:prop_end] = proprio. Exact indices depend on obs term ordering.
"""

import copy

import isaaclab.envs.mdp as stock_mdp
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as robot_lab_mdp
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.thunder_gait.rough_env_cfg import (
    ThunderGaitRoughEnvCfg,
)


@configclass
class ThunderGaitTarOfficialRoughEnvCfg(ThunderGaitRoughEnvCfg):
    """Rough gait for TAR official.

    Policy history_length = 10 (matches official num_hist default for Thunder).
    Critic obs rebuilt with base_lin_vel FIRST so extract_critic[0:3] is velocity.
    """

    def __post_init__(self):
        super().__post_init__()

        # Policy: 10-frame history (TAR official num_hist)
        self.observations.policy.history_length = 10

        # Critic: single-frame (TAR does not use critic history by default)
        self.observations.critic.history_length = 1
        if hasattr(self.observations, "height_scan_group"):
            self.observations.height_scan_group.history_length = 1

        # Rebuild critic obs group with base_lin_vel FIRST (needed for extract_critic)
        # Save original terms (excluding inherited internals)
        existing_terms = {}
        for attr_name in list(vars(self.observations.critic).keys()):
            if attr_name.startswith("_"):
                continue
            val = getattr(self.observations.critic, attr_name)
            if isinstance(val, ObsTerm):
                existing_terms[attr_name] = val

        # Remove existing
        for name in list(existing_terms.keys()):
            delattr(self.observations.critic, name)

        # Prepend base_lin_vel, then re-add rest (deduplicated)
        self.observations.critic.base_lin_vel = ObsTerm(func=stock_mdp.base_lin_vel)
        for name, term in existing_terms.items():
            if name == "base_lin_vel":
                continue
            setattr(self.observations.critic, name, term)

        # Parent only runs disable_zero_weight_rewards when class name matches exactly
        self.disable_zero_weight_rewards()
