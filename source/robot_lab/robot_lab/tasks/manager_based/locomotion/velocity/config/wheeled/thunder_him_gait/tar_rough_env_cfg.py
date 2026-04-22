# Copyright (c) 2024-2026 Inovxio (穹沛科技)
# SPDX-License-Identifier: Apache-2.0
"""Thunder Gait TAR rough -- 15-frame history + teacher/privileged + estimator_targets."""

import torch
import isaaclab.envs.mdp as stock_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as robot_lab_mdp
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.thunder_gait.rough_env_cfg import (
    ThunderGaitRoughEnvCfg,
)


# ---------------------------------------------------------------------------
# Fallback: define base_pos_z locally if not available in stock_mdp
# ---------------------------------------------------------------------------
def _base_pos_z(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height (z) in world frame, shape [num_envs, 1]."""
    from isaaclab.assets import Articulation
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]


# Use stock_mdp version if available, else our fallback
_base_pos_z_func = getattr(stock_mdp, "base_pos_z", _base_pos_z)


@configclass
class TAREstimatorTargetsCfg(ObsGroup):
    """GT targets for TAR velocity/height/CoM estimator.

    Layout: [base_lin_vel(3), base_pos_z(1), com_pos_xy(2)] = 6 dims
    """

    base_lin_vel = ObsTerm(func=stock_mdp.base_lin_vel)
    base_pos_z = ObsTerm(func=_base_pos_z_func)
    com_pos_xy = ObsTerm(func=robot_lab_mdp.center_of_mass_position)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class ThunderGaitTarRoughEnvCfg(ThunderGaitRoughEnvCfg):
    """Rough gait with 15-frame policy history + single-frame critic + GT estimator targets."""

    def __post_init__(self):
        super().__post_init__()

        # 15-frame history on policy; critic single-frame to keep teacher encoder small
        self.observations.policy.history_length = 15
        self.observations.critic.history_length = 1
        if hasattr(self.observations, "height_scan_group"):
            self.observations.height_scan_group.history_length = 1

        # GT estimator targets (clean, no DR noise, no history)
        self.observations.estimator_targets = TAREstimatorTargetsCfg()

        # Explicit call (parent only runs when class name matches exactly)
        self.disable_zero_weight_rewards()
