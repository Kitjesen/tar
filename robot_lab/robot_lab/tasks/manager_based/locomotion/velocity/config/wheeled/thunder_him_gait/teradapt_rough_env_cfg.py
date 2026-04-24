"""Thunder Gait TerAdapt rough: short(5) + long(50) history + VQ-VAE TCA.

v1_nogait: gait-shaping rewards removed, let TCA codebook drive gait emergence.
"""

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
    keeps critic/height_scan single-frame, adds vel_gt group, and removes
    gait-shaping rewards so TCA codebook drives gait selection.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Split policy into short and long history variants ---
        orig_policy = self.observations.policy
        short = copy.deepcopy(orig_policy)
        short.history_length = 5
        long = copy.deepcopy(orig_policy)
        long.history_length = 50
        self.observations.policy_short = short
        self.observations.policy_long = long
        del self.observations.policy

        # Critic + height_scan single frame
        self.observations.critic.history_length = 1
        if hasattr(self.observations, "height_scan_group"):
            self.observations.height_scan_group.history_length = 1

        # GT velocity for vel_head MSE supervision
        self.observations.vel_gt = VelGtCfg()

        # ================================================================
        # TerAdapt reward overhaul (v1_nogait):
        # Remove gait-shaping so TCA codebook drives gait emergence naturally.
        # Paper TABLE II reference, Thunder wheeled-legged adapted.
        # ================================================================
        r = self.rewards

        # ---- 1. Velocity tracking — strong signal preserved (no gait gate) ----
        r.track_lin_vel_xy_exp.weight = 8.0
        r.track_ang_vel_z_exp.weight = 3.0

        # ---- 2. Delete all gait-shaping rewards ----
        for _name in (
            "gait_gated_lin_vel",
            "gait_gated_ang_vel",
            "feet_gait",
            "lateral_gated_air_time",
            "gait_contact_symmetry",
            "foot_height_symmetry",
            "morphological_symmetry",
            "foot_height_in_swing",
            "gait_phase_clock",
        ):
            if hasattr(r, _name) and getattr(r, _name) is not None:
                setattr(r, _name, None)

        # ---- 3. Stability CORE — force explicit values (bypass upstream drift) ----
        # Previous audit showed rough/flat's __post_init__ weight assignments
        # did NOT stick (base_height ended at -0.5 not -2.0, upward at +0.5 not +2.0).
        # Root cause unclear; workaround is to re-assign every critical weight here.
        r.upward.weight = 2.0                     # user requested (was +0.5)
        r.base_height_l2.weight = -2.0            # rough baseline (was -0.5)
        r.feet_impact_vel.weight = -5.0           # rough baseline (was -0.5)
        r.joint_pos_penalty.weight = -1.0         # flat dataclass (was -0.3)
        r.flat_orientation_l2.weight = -1.0       # already OK, re-assert
        r.lin_vel_z_l2.weight = -2.0              # flat baseline (was 0 = disabled!)
        r.ang_vel_xy_l2.weight = -0.5             # anti body-twist (was 0 = disabled!)
        r.feet_clearance.weight = -1.5            # anti ground-drag (was 0 = disabled!)

        # ---- 4. Energy consolidation: drop joint_power, soften adaptive_energy ----
        if hasattr(r, "joint_power") and r.joint_power is not None:
            r.joint_power = None
        if hasattr(r, "adaptive_energy") and r.adaptive_energy is not None:
            r.adaptive_energy.weight = 0.5

        # ---- 5. DOF velocity to paper value ----
        if hasattr(r, "joint_vel_l2") and r.joint_vel_l2 is not None:
            r.joint_vel_l2.weight = -1e-5

        # ---- 6. Action smoothness — user requested -0.01 (flat baseline) ----
        if hasattr(r, "action_rate_l2") and r.action_rate_l2 is not None:
            r.action_rate_l2.weight = -0.01

        # ---- 6. Keep joint_mirror + hip_pos_penalty (stability anchors) ----
        if hasattr(r, "joint_mirror") and r.joint_mirror is not None:
            r.joint_mirror.weight = -0.05

        if not hasattr(r, "hip_pos_penalty") or r.hip_pos_penalty is None:
            from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
            import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
            r.hip_pos_penalty = RewTerm(
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
        else:
            r.hip_pos_penalty.weight = -3.0

        # ---- 7. Audit print (final state for verification) ----
        print("\n[TerAdapt v1_nogait] ===== Final reward weights =====")
        for _name in sorted(dir(r)):
            if _name.startswith("__"):
                continue
            _r = getattr(r, _name)
            if _r is None or callable(_r):
                continue
            _w = getattr(_r, "weight", None)
            if _w is not None:
                print(f"  {_name:35s} weight={_w}")
        print("[TerAdapt v1_nogait] ================================\n")

        # ---- 8. Strip any remaining zero-weight rewards ----
        self.disable_zero_weight_rewards()
