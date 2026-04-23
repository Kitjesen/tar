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

        # --- Reward weight fixes (applied BEFORE disable_zero_weight_rewards) ---
        # DEBUG: dump all reward attribute names + weights so we see what's actually there
        print("\n[TerAdapt env __post_init__] Reward audit BEFORE fixes:")
        for _name in sorted(dir(self.rewards)):
            if _name.startswith("__"):
                continue
            _r = getattr(self.rewards, _name)
            if _r is None:
                print(f"  {_name:35s} = None")
                continue
            if callable(_r):
                continue
            _w = getattr(_r, "weight", "<no weight attr>")
            print(f"  {_name:35s} weight={_w}")

        # joint_mirror was severely under-weighted (-0.01 vs raw error ~60).
        if hasattr(self.rewards, "joint_mirror") and self.rewards.joint_mirror is not None:
            self.rewards.joint_mirror.weight = -0.05
            print(f"[TerAdapt] SET joint_mirror.weight = {self.rewards.joint_mirror.weight}")
        else:
            print("[TerAdapt] joint_mirror NOT FOUND or None")

        # hip_pos_penalty
        if hasattr(self.rewards, "hip_pos_penalty") and self.rewards.hip_pos_penalty is not None:
            self.rewards.hip_pos_penalty.weight = -3.0
            print(f"[TerAdapt] SET hip_pos_penalty.weight = {self.rewards.hip_pos_penalty.weight}")
        else:
            print("[TerAdapt] hip_pos_penalty NOT FOUND or None — needs full reconstruction")
            # Try to reconstruct the term directly
            try:
                from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
                import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
                self.rewards.hip_pos_penalty = RewTerm(
                    func=mdp.joint_pos_penalty,
                    weight=-3.0,
                    params={
                        "command_name": "base_velocity",
                        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
                        "stand_still_scale": 3.0,
                        "velocity_threshold": 0.05,
                        "command_threshold": 0.1,
                    },
                )
                print(f"[TerAdapt] RECONSTRUCTED hip_pos_penalty with weight={self.rewards.hip_pos_penalty.weight}")
            except Exception as e:
                print(f"[TerAdapt] hip_pos_penalty reconstruction FAILED: {e}")

        # Parent only disables zero-weight rewards when class name matches exactly
        self.disable_zero_weight_rewards()

        # Verify after disable
        jm = getattr(self.rewards, "joint_mirror", None)
        hp = getattr(self.rewards, "hip_pos_penalty", None)
        print(f"[TerAdapt] AFTER disable: joint_mirror={jm.weight if jm else None}  hip_pos_penalty={hp.weight if hp else None}\n")
