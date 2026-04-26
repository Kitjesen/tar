# Copyright (c) 2024-2026 Inovxio (穹沛科技)
# SPDX-License-Identifier: Apache-2.0
"""Thunder Gait HIM rough -- 15-frame history + SwAV HIM estimator.

Compatible with existing train_him.py / HIMActorCritic / HIMEstimator:
  - policy obs: 15-frame history (stacked by Isaac Lab ObsGroup.history_length)
  - critic obs: base_lin_vel MUST be first (HIMEstimator.update extracts GT vel from here)
  - estimator: SwAV contrastive + MSE(pred_vel, gt_vel from critic_obs first 3*H dims)
"""

import isaaclab.envs.mdp as stock_mdp
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
)
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.thunder_gait.rough_env_cfg import (
    ThunderGaitRoughEnvCfg,
)


@configclass
class ThunderGaitHimRoughEnvCfg(ThunderGaitRoughEnvCfg):
    """Rough gait with 15-frame history (for train_him.py)."""

    def __post_init__(self):
        super().__post_init__()

        # 15-frame history on policy + critic (both stacked)
        self.observations.policy.history_length = 15
        self.observations.critic.history_length = 15
        # Height scan stays single-frame (privileged terrain map, no stacking)
        if hasattr(self.observations, "height_scan_group"):
            self.observations.height_scan_group.history_length = 1

        # HIMEstimator extracts ground-truth velocity from the FIRST 3*H dims of
        # critic_obs (see extract_current_velocity_isaac). Prepend base_lin_vel
        # to the critic group. `configclass` is a dataclass-like container, so we
        # rebuild the critic group with base_lin_vel first, then original terms.
        base_lin_vel_term = ObsTerm(func=stock_mdp.base_lin_vel)

        # Copy existing critic terms (excluding dataclass internals) to a temp
        existing_terms = {}
        for attr_name in list(vars(self.observations.critic).keys()):
            if attr_name.startswith("_"):
                continue
            val = getattr(self.observations.critic, attr_name)
            if isinstance(val, ObsTerm):
                existing_terms[attr_name] = val

        # Remove then re-add in order: base_lin_vel first, then the rest
        for name in list(existing_terms.keys()):
            delattr(self.observations.critic, name)
        self.observations.critic.base_lin_vel = base_lin_vel_term
        for name, term in existing_terms.items():
            if name == "base_lin_vel":
                continue  # skip duplicate
            setattr(self.observations.critic, name, term)

        # Parent `disable_zero_weight_rewards()` only runs when class name matches
        # `ThunderGaitRoughEnvCfg`. Since we are a subclass, call it explicitly.
        self.disable_zero_weight_rewards()
