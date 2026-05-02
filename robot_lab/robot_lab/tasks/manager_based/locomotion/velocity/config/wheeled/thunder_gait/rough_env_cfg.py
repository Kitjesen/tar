# Copyright (c) 2024-2026 Inovxio (穹沛科技)
# SPDX-License-Identifier: Apache-2.0
"""
Thunder Gait Rough Terrain — Stage II (Asymmetric Actor-Critic)

Actor: 盲行（policy obs 与 flat 一致，可部署 S100P）
Critic: 带特权观测（height_scan + 接触力 + 摩擦 + 质量 + base_lin_vel）
  - 通过 obs_groups 路由实现（见 agents/rsl_rl_ppo_cfg.py）
  - 训练时 critic 看地形结构，actor 不看

注意: critic 维度变了，不能 resume flat checkpoint 整体
  - 用 agents cfg 里的 obs_groups={"policy":["policy"],"critic":["critic","height_scan_group"]}
"""

import isaaclab.envs.mdp as stock_mdp
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as rl_mdp

from .flat_env_cfg import ThunderGaitFlatEnvCfg


# ============================================================================
# 特权观测组：只给 critic 看的 height_scan
# ============================================================================
@configclass
class HeightScanCfg(ObsGroup):
    """地形高度扫描（privileged，仅 critic 可见）。"""

    height_scan = ObsTerm(
        func=stock_mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.05},
        clip=(-1.0, 1.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False  # privileged，不加噪声
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class ThunderGaitRoughEnvCfg(ThunderGaitFlatEnvCfg):
    """Thunder Gait on rough terrain — asymmetric actor-critic with privileged critic."""

    def __post_init__(self):
        super().__post_init__()

        # ======================== Scene: 启用 rough 地形 + height_scanner ========================
        self.scene.terrain.terrain_type = "generator"
        from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
            ROUGH_TERRAINS_CFG,
        )
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG

        # 启用 height_scanner（提供给 critic 当特权）
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner_base = None  # 暂不用

        # ======================== 观测：policy 保持盲行，critic 加特权 ========================
        # Policy 不动（flat 里已经 height_scan=None）
        if hasattr(self.observations.policy, 'gait_phase') and self.observations.policy.gait_phase is not None:
            self.observations.policy.gait_phase = None
        if hasattr(self.observations.critic, 'gait_phase') and self.observations.critic.gait_phase is not None:
            self.observations.critic.gait_phase = None

        # 给 critic 组加特权观测（通过 setattr 动态添加）
        # base_lin_vel 已在 flat 的 CriticCfg 中
        self.observations.critic.foot_contact_forces = ObsTerm(
            func=rl_mdp.foot_contact_force_magnitudes,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "asset_cfg": SceneEntityCfg("robot"),
                "foot_names_regex": ".*_foot",
            },
            clip=(-500.0, 500.0),
            scale=0.01,  # 归一到 O(1) 量级
        )
        self.observations.critic.foot_friction = ObsTerm(
            func=rl_mdp.average_foot_friction,
            clip=(-10.0, 10.0),
            scale=1.0,
        )
        self.observations.critic.mass_dist = ObsTerm(
            func=rl_mdp.mass_distribution_components,
            params={"normalize": True},
            clip=(-10.0, 10.0),
            scale=1.0,
        )

        # 增加 height_scan_group（仅供 critic 通过 obs_groups 路由吃到）
        self.observations.height_scan_group = HeightScanCfg()

        # ======================== 奖励 (2026-04-17 简化：让机器人动起来) ========================
        # 解决"原地 trot 踏步"问题 — 删冗余约束 + 松绑大冲击惩罚
        self.rewards.base_height_l2.weight = -0.5          # -2.0 → -1.0（rough 上身高必然波动）
        self.rewards.base_height_l2.params["target_height"] = 0.55
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.joint_power.weight = -5e-5
        self.rewards.feet_impact_vel.weight = -0.5         # -5.0 → -0.5（rough 踩台阶必然硬冲击，不该重罚）
        self.rewards.feet_slide.weight = -1.0

        # 删冗余 trot 约束（对称性三项 + joint_mirror 已覆盖 trot 节奏）
        if hasattr(self.rewards, "gait_phase_clock") and self.rewards.gait_phase_clock is not None:
            self.rewards.gait_phase_clock = None           # 对角相位约束，跟 gait_contact_symmetry 重叠
        if hasattr(self.rewards, "feet_gait") and self.rewards.feet_gait is not None:
            self.rewards.feet_gait = None                  # trot 接触时长奖励，对称性已强制 trot
        if hasattr(self.rewards, "hip_pos_penalty") and self.rewards.hip_pos_penalty is not None:
            self.rewards.hip_pos_penalty = None            # 与 joint_pos_penalty 重叠（都压全腿姿态）

        # ======================== Curriculum: 启用地形 curriculum ========================
        import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

        # ======================== 清理零权重奖励 ========================
        if self.__class__.__name__ == "ThunderGaitRoughEnvCfg":
            self.disable_zero_weight_rewards()
