"""Thunder Gait TerAdapt rough: short(5) + long(50) history + VQ-VAE TCA.

teradapt: gait-shaping rewards removed, let TCA codebook drive gait emergence.
"""

import copy

import isaaclab.envs.mdp as stock_mdp
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
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
        # TerAdapt reward overhaul (teradapt):
        # Remove gait-shaping so TCA codebook drives gait emergence naturally.
        # Paper TABLE II reference, Thunder wheeled-legged adapted.
        # ================================================================
        r = self.rewards

        # ---- 1. Velocity tracking — strong signal preserved (no gait gate) ----
        r.track_lin_vel_xy_exp.weight = 8.0
        r.track_ang_vel_z_exp.weight = 3.0
        # 2026-04-25 patch: std 0.5 → sqrt(2.0)=1.414，对齐 orix_dog 标杆
        # 旧 std=0.5 在 cmd=1.5 时 reward=0.001（exp 饱和），策略无梯度信号
        # 新 std=1.414 时 reward=2.6（强 2600 倍），from-scratch 能学
        # 参考：orix_dog/rough_env_cfg.py:209 已被验证 from-scratch 训出
        import math as _math
        r.track_lin_vel_xy_exp.params["std"] = _math.sqrt(2.0)
        r.track_ang_vel_z_exp.params["std"] = _math.sqrt(2.0)

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

        # ---- 3. Stability CORE — explicit values (bypass upstream drift) ----
        # Aligned with Unitree B2W reference config for wheeled-legged quadruped
        # 2026-04-26 patch: upward 1.0 → 2.0 (强化直立; b2w 标杆 3.0)
        r.upward.weight = 2.0
        r.base_height_l2.weight = 0.0            # B2W: disabled
        # feet_impact_vel: start lenient (-0.5) so policy can explore; curriculum ramps up
        r.feet_impact_vel.weight = -0.5
        r.joint_pos_penalty.weight = -1.0
        r.flat_orientation_l2.weight = 0.0       # B2W: disabled
        r.lin_vel_z_l2.weight = -2.0
        r.ang_vel_xy_l2.weight = -0.05           # B2W: -0.05 (was -0.5, too strict for yaw turning)
        # 2026-04-26 patch: feet_clearance 关闭 (b2w 标杆完全不用此项)
        r.feet_clearance.weight = 0.0
        r.joint_acc_l2.weight = -1e-7            # B2W: -1e-7 (weakened from -5e-7)
        r.joint_pos_limits.weight = -5.0         # B2W: -5.0 (stricter; was -3.0)

        # ---- 4. Energy: add back joint_power (B2W has it), keep adaptive_energy ----
        if hasattr(r, "joint_power") and r.joint_power is not None:
            r.joint_power.weight = -1e-5         # B2W: -1e-5 (was removed, now restored)
        # 2026-04-26 patch: adaptive_energy 关闭 (悬空腿激励主因 — 悬空腿能耗低 → 拿正分)
        if hasattr(r, "adaptive_energy") and r.adaptive_energy is not None:
            r.adaptive_energy.weight = 0.0

        # ---- 5. DOF velocity ----
        if hasattr(r, "joint_vel_l2") and r.joint_vel_l2 is not None:
            r.joint_vel_l2.weight = -5e-6

        # ---- 6. Action smoothness ----
        if hasattr(r, "action_rate_l2") and r.action_rate_l2 is not None:
            r.action_rate_l2.weight = -0.01

        # ---- 7. Joint mirror ----
        if hasattr(r, "joint_mirror") and r.joint_mirror is not None:
            r.joint_mirror.weight = -0.01

        # hip_pos_penalty removed: wheeled-legged terrain adaptation needs hip freedom
        if hasattr(r, "hip_pos_penalty") and r.hip_pos_penalty is not None:
            r.hip_pos_penalty = None

        # ---- 8. Curriculum: ramp feet_impact_vel -0.5 -> -2.0 -> -5.0 as policy stabilises ----
        from isaaclab.managers import CurriculumTermCfg as CurrTerm
        from isaaclab.envs.mdp.curriculums import modify_reward_weight
        self.curriculum.feet_impact_vel_ramp1 = CurrTerm(
            func=modify_reward_weight,
            params={"term_name": "feet_impact_vel", "weight": -2.0, "num_steps": 48_000},
        )
        self.curriculum.feet_impact_vel_ramp2 = CurrTerm(
            func=modify_reward_weight,
            params={"term_name": "feet_impact_vel", "weight": -5.0, "num_steps": 144_000},
        )

        # ---- 10. v5 patch (2026-04-26): 加陡斜坡 + 加大撞地惩罚 ----
        # pyramid_slope (-pyramid_slope_inv): 0.4 rad (23°) → 0.6 rad (34°)
        # 让 row 6+ 真正具备挑战性，迫使策略适应陡坡
        for _slope_name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
            _sub = self.scene.terrain.terrain_generator.sub_terrains.get(_slope_name)
            if _sub is not None:
                _sub.slope_range = (0.0, 0.6)

        # undesired_contacts: -1.0 → -3.0
        # 罚非脚部位（hip, thigh, calf, base）撞地。因 v4 砍掉 feet_clearance，
        # 用"撞地后果"反推策略学习抬脚（轮足踢台阶 → calf 撞 → 痛 → 抬高）
        if hasattr(r, "undesired_contacts") and r.undesired_contacts is not None:
            r.undesired_contacts.weight = -3.0

        # ---- 9. Bootstrap loosening (2026-04-25 patch): 让策略先会走 ----
        # 旧 run iter 100+ terr 塌底 → 暴露 PPO 在 teradapt 高维 VQ 空间从随机初始化
        # 找不到"会走"的局部最优。临时放松 stability 约束，2000 iter 后再开。
        r.joint_pos_limits.weight = -3.0   # 原 -5.0，放松关节边界，鼓励探索

        # 临时关闭这两项（占位 -1e-8 避开 disable_zero_weight_rewards 的 None 化）
        if hasattr(r, "body_orientation_stability") and r.body_orientation_stability is not None:
            r.body_orientation_stability.weight = -1e-8
        if hasattr(r, "wheel_lateral_slip") and r.wheel_lateral_slip is not None:
            r.wheel_lateral_slip.weight = -1e-8

        # Curriculum: 2000 iter (= 96_000 sim step @ num_steps_per_env=48) 后开回原权重
        self.curriculum.body_orient_open = CurrTerm(
            func=modify_reward_weight,
            params={"term_name": "body_orientation_stability", "weight": -0.5, "num_steps": 96_000},
        )
        self.curriculum.wheel_slip_open = CurrTerm(
            func=modify_reward_weight,
            params={"term_name": "wheel_lateral_slip", "weight": -0.5, "num_steps": 96_000},
        )

        # ---- 7. Audit print (final state for verification) ----
        print("\n[TerAdapt teradapt] ===== Final reward weights =====")
        for _name in sorted(dir(r)):
            if _name.startswith("__"):
                continue
            _r = getattr(r, _name)
            if _r is None or callable(_r):
                continue
            _w = getattr(_r, "weight", None)
            if _w is not None:
                print(f"  {_name:35s} weight={_w}")
        print("[TerAdapt teradapt] ================================\n")

        # ---- 11. Fall termination (2026-04-26 patch) ----
        # 仅在机器人稳定站立过之后摔倒才 terminate（避免初始随机姿态掉落误终止）
        # 站立判定：base_z > 0.30 m 且 gravity_b[z] < -0.7（接近完全直立）持续 20 步 (0.4s)
        # 摔倒判定：base_z < 0.20 m 或 gravity_b[z] > 0.5（翻过 60 度）
        self.terminations.fall = DoneTerm(
            func=mdp.fall_after_stood_up,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_height_threshold": 0.30,
                "gravity_z_threshold": -0.7,
                "stable_steps": 20,
                "fall_height_threshold": 0.20,
                "fall_gravity_z_threshold": 0.5,
            },
        )

        # Reset event 清除 has_stood_up latch（每个 episode 重新开始判断）
        self.events.reset_fall_state = EventTerm(
            func=mdp.reset_fall_after_stood_up_state,
            mode="reset",
        )

        # ---- 12. v6 patch (2026-04-26): 关节惩罚分组（hip/thigh/calf/foot）+ 摔倒时也罚 ----
        # 替换原 joint_pos_penalty (全 16 joints L2，且摔倒不罚) 为 4 个分组：
        #   - hip 最重 (-1.0): 抑制扭身体 + 防侧倒
        #   - thigh 中等 (-0.3): 步态需要前后摆动，留空间
        #   - calf 轻 (-0.1): 爬台阶需要弯曲，几乎不罚
        #   - foot (轮): 罚 wheel velocity² 偏离 0 (-1e-3)，让轮子默认静止
        # 摔倒时也罚（用 joint_pos_penalty_no_fall_filter 不含 upright 调制）

        # 关掉原 joint_pos_penalty（全 16 joints 一刀切）
        if hasattr(r, "joint_pos_penalty") and r.joint_pos_penalty is not None:
            r.joint_pos_penalty.weight = 0.0  # 走 disable_zero_weight_rewards 删除

        # 新加：分组 joint position deviation
        from isaaclab.managers import RewardTermCfg as RewTerm
        from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.thunder_gait.rewards import (
            joint_pos_penalty_no_fall_filter,
        )

        # v9: 每组独立 stand_still_scale，让静止时 effective weight 统一到 ~-5
        _common = {
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        }
        self.rewards.joint_pos_penalty_hip = RewTerm(
            func=joint_pos_penalty_no_fall_filter,
            weight=-1.0,
            params={**_common, "stand_still_scale": 5.0,  # 运动 -1.0, 静止 -5.0
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
        )
        self.rewards.joint_pos_penalty_thigh = RewTerm(
            func=joint_pos_penalty_no_fall_filter,
            weight=-0.3,
            params={**_common, "stand_still_scale": 17.0,  # 运动 -0.3, 静止 -5.1
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint"])},
        )
        self.rewards.joint_pos_penalty_calf = RewTerm(
            func=joint_pos_penalty_no_fall_filter,
            weight=-0.1,
            params={**_common, "stand_still_scale": 50.0,  # 运动 -0.1, 静止 -5.0
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint"])},
        )

        # foot/wheel: 罚速度²，让轮子默认静止（命令需要时再转）
        from isaaclab.envs.mdp.rewards import joint_vel_l2 as _stock_joint_vel_l2
        self.rewards.foot_vel_penalty = RewTerm(
            func=_stock_joint_vel_l2,
            weight=-1e-3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_joint"])},
        )

        # ---- 13/14. v8 patch: 合并 foot_vel_penalty + wheel_vel_zero_cmd 为统一 motion-aware 项 ----
        # 旧设计: foot_vel_penalty (-0.001 全时段 v²) + wheel_vel_zero_cmd (-0.5 仅 cmd=0 |v|)
        # 新设计: wheel_vel_motion_aware (-0.05 基础, 静止时 ×20 = -1.0)
        # 静止时罚力度比 v7 加倍，且消除两项重叠的逻辑

        # 关掉旧的两项
        if hasattr(r, "wheel_vel_zero_cmd") and r.wheel_vel_zero_cmd is not None:
            r.wheel_vel_zero_cmd.weight = 0.0
        if hasattr(r, "foot_vel_penalty") and r.foot_vel_penalty is not None:
            r.foot_vel_penalty.weight = 0.0

        # 加新的 motion-aware 罚
        from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.thunder_gait.rewards import (
            wheel_vel_motion_aware,
        )
        self.rewards.foot_vel_motion_aware = RewTerm(
            func=wheel_vel_motion_aware,
            weight=-0.05,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_joint"]),
                "stand_still_scale": 100.0,  # v9: 20 -> 100 (运动 -0.05, 静止 -5.0)
                "command_threshold": 0.1,
                "velocity_threshold": 0.5,
            },
        )

        # ---- 8. Strip any remaining zero-weight rewards ----
        self.disable_zero_weight_rewards()
