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

        self._apply_current_hist_reward_settings()

        # Parent `disable_zero_weight_rewards()` only runs when class name matches
        # `ThunderGaitRoughEnvCfg`. Since we are a subclass, call it explicitly.
        self.disable_zero_weight_rewards()

    def _apply_current_hist_reward_settings(self):
        """Keep the HIM baseline aligned with the current Thunder Hist rough setup."""

        weights = {
            "track_lin_vel_xy_exp": 8.0,
            "track_ang_vel_z_exp": 3.0,
            "upward": 2.0,
            "lin_vel_z_l2": -2.0,
            "ang_vel_xy_l2": -0.05,
            "flat_orientation_l2": -0.1,
            "base_height_l2": 0.0,
            "body_lin_acc_l2": 0.0,
            "joint_torques_l2": -1e-5,
            "joint_torques_wheel_l2": 0.0,
            "joint_vel_l2": 0.0,
            "joint_vel_wheel_l2": 0.0,
            "joint_acc_l2": -2.5e-7,
            "joint_acc_wheel_l2": 0.0,
            "joint_pos_limits": -3.0,
            "joint_power": -2e-5,
            "stand_still": -2.0,
            "joint_pos_penalty": -1.0,
            "wheel_vel_penalty": 0.0,
            "wheel_vel_zero_cmd": 0.0,
            "joint_mirror": -0.01,
            "action_rate_l2": -0.01,
            "action_smoothness_l2": 0.0,
            "undesired_contacts": -1.0,
            "contact_forces": -0.0003,
            "feet_air_time": 0.0,
            "feet_contact": 0.0,
            "feet_contact_without_cmd": 0.1,
            "feet_stumble": -5.0,
            "feet_slide": 0.0,
            "feet_impact_vel": 0.0,
            "excess_contact_force": 0.0,
            "adaptive_energy": 0.0,
            "feet_height": 0.0,
            "feet_height_body": 0.0,
            "feet_gait": 0.0,
        }

        for name, weight in weights.items():
            term = getattr(self.rewards, name, None)
            if term is not None:
                term.weight = weight

        joint_mirror = getattr(self.rewards, "joint_mirror", None)
        if joint_mirror is not None:
            joint_mirror.params["mirror_joints"] = [
                ["FR_(hip|thigh|calf)_joint", "FL_(hip|thigh|calf)_joint"],
                ["RR_(hip|thigh|calf)_joint", "RL_(hip|thigh|calf)_joint"],
            ]
