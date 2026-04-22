"""Thunder Gait TerAdapt rough: short(5) + long(50) history + VQ-VAE TCA."""

import copy

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
class VelGtCfg(ObsGroup):
    """Clean ground-truth base_lin_vel for TerAdapt velocity head MSE supervision."""

    base_lin_vel = ObsTerm(func=stock_mdp.base_lin_vel)

    def __post_init__(self):
        self.enable_corruption = False  # clean GT
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class ThunderGaitTerAdaptRoughEnvCfg(ThunderGaitRoughEnvCfg):
    """Rough gait for TerAdapt: splits policy into short(5) + long(50) groups,
    keeps critic/height_scan single-frame, adds vel_gt group.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Split policy into short and long history variants ---
        # Parent defines self.observations.policy with some history_length. We replace
        # it with policy_short (5 frames) and duplicate into policy_long (50 frames).
        # All proprio obs terms are identical; only history_length differs.
        orig_policy = self.observations.policy
        short = copy.deepcopy(orig_policy)
        short.history_length = 5
        long = copy.deepcopy(orig_policy)
        long.history_length = 50
        self.observations.policy_short = short
        self.observations.policy_long = long
        # Remove the original policy group
        del self.observations.policy

        # Critic single frame (teacher encoder expects small dim)
        self.observations.critic.history_length = 1
        if hasattr(self.observations, "height_scan_group"):
            self.observations.height_scan_group.history_length = 1

        # GT velocity target for vel_head MSE supervision
        self.observations.vel_gt = VelGtCfg()

        # Parent only disables zero-weight rewards when class name matches exactly
        self.disable_zero_weight_rewards()
