# Copyright (c) 2024-2026 Inovxio (穹沛科技)
# SPDX-License-Identifier: Apache-2.0
"""
Thunder Gait -- standalone flat terrain config with direction-aware gait-gated rewards.

基于 thunder_hist/rough_env_cfg.py 的完整内联版本，改动：
  1. 单帧观测 (history_length=1)，训练速度最快
  2. 平地地形，排除地形变量
  3. track_lin_vel_xy_exp / track_ang_vel_z_exp -> gait_gated_lin_vel / gait_gated_ang_vel
  4. 新增 feet_gait=0.5（驱动 trot 步态）
  5. 新增 wheel_lateral_slip=-0.5（抑制侧滑）

验证目标：步态门控是否消除 spinning-in-place，侧移是否出现 trot 步态。

训练命令:
  conda activate thunder
  cd /home/bsrl/hongsenpang/RLbased/robot_lab
  python scripts/reinforcement_learning/rsl_rl/train.py \
      --task RobotLab-Isaac-Velocity-Flat-Thunder-Gait-v0 \
      --num_envs 4096 --headless --device cuda:0 \
      --max_iterations 10000 --experiment_name thunder_gait_flat_v1
"""

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.thunder_gait import rewards as gait_rewards
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.events import (
    reset_joints_with_type_specific_ranges,
)
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)

from robot_lab.assets.thunder import THUNDER_CFG_V1  # isort: skip


# =============================================================================
# Tunable parameter dataclasses
# =============================================================================

@configclass
class ThunderGaitFlatRewardWeights:
    """Reward weights -- 门控版本替换了 track_lin/ang_vel。"""

    is_terminated: float = 0.0

    # Velocity tracking -- 禁用原始版本，用门控替代
    track_lin_vel_xy_exp: float = 0.0
    track_ang_vel_z_exp: float = 0.0
    # Gait-gated tracking -- 核心实验变量
    gait_gated_lin_vel: float = 8.0         # 原 track_lin_vel_xy_exp=12.0 -> 降至 8.0
    gait_gated_ang_vel: float = 3.0         # 原 track_ang_vel_z_exp=5.0 -> 降至 3.0
    upward: float = 0.5

    # Root penalties
    lin_vel_z_l2: float = 0.0
    ang_vel_xy_l2: float = 0.0
    flat_orientation_l2: float = -1.0
    base_height_l2: float = -8.0             # 惩罚趴低：硬约束站立高度 0.55m
    body_lin_acc_l2: float = 0.0

    # Joint penalties
    joint_torques_l2: float = 0.0
    joint_torques_wheel_l2: float = 0.0
    joint_vel_l2: float = 0.0
    joint_vel_wheel_l2: float = 0.0
    joint_acc_l2: float = -5e-7
    joint_acc_wheel_l2: float = 0.0
    joint_pos_limits: float = -3.0
    joint_vel_limits: float = 0.0
    joint_power: float = -5e-4              # 加重腿部能量惩罚 10×，抑制直行时无意义摆腿
    stand_still: float = -2.0
    joint_pos_penalty: float = -0.3
    wheel_vel_penalty: float = 0.0
    wheel_vel_zero_cmd: float = -0.05
    joint_mirror: float = -0.05

    # Action penalties
    action_rate_l2: float = -0.005
    action_smoothness_l2: float = 0.0

    # Contact penalties
    undesired_contacts: float = -1.0
    contact_forces: float = 0.0

    # Gait rewards -- 核心实验变量
    feet_gait: float = 1.5                  # 驱动 trot 步态学习（隐式课程）
    feet_clearance: float = 0.0            # 摆腿期强制离地高度 ≥ 6cm，解决拖地
    body_orientation_stability: float = -0.5  # 减少 trot 期身体左右摇摆
    wheel_lateral_slip: float = -0.5        # 抑制侧向拖地

    # Phase clock — 速度自适应步频，驱动 trot 节奏
    gait_phase_clock: float = 2.0           # 相位时钟：脚在正确时刻着地/抬起

    # Phase 1 专用：悬空期高度正奖励，打破腿静止陷阱（Phase2 设为 0）
    foot_height_in_swing: float = 0.0       # Phase1 里覆盖为 2.0

    # Hip 专用位置约束（独立于 joint_pos_penalty，权重更强）
    hip_pos_penalty: float = -1.0           # 防止 hip 分叉：比通用 joint_pos_penalty 强 3×

    # Symmetry rewards（Phase2 启用，Phase1 先关）
    # A. 对角接触时序对称 — 管 trot 节奏
    gait_contact_symmetry: float = -0.5      # 推荐 -0.5，Phase2 开启
    # B. 左右摆腿高度对称 — 管视觉歪不歪
    foot_height_symmetry: float = -0.3       # 推荐 -0.3，Phase2 开启
    # C. 关节镜像对称 — 管腿型自然度
    morphological_symmetry: float = -0.1    # 推荐 -0.1，谨慎使用

    # Other
    feet_air_time: float = 0.0              # 关闭无差别抬脚激励（直行不应抬腿）
    feet_contact: float = 0.0
    feet_contact_without_cmd: float = 0.1
    feet_stumble: float = -5.0
    feet_slide: float = -1.0           # 用 wheeled_feet_slide 修正后可以回调（不再误判 stance）
    feet_impact_vel: float = -5.0      # 原 -2.0 → -5.0：加强落地冲击惩罚
    excess_contact_force: float = 0.0
    adaptive_energy: float = 1.0
    feet_height: float = 0.0
    feet_height_body: float = 0.0


@configclass
class ThunderGaitFlatEventParams:
    reset_base_pose_range_x: tuple = (-0.5, 0.5)
    reset_base_pose_range_y: tuple = (-0.5, 0.5)
    reset_base_pose_range_z: tuple = (0.0, 0.2)
    reset_base_pose_range_roll: tuple = (-0.3, 0.3)
    reset_base_pose_range_pitch: tuple = (-0.3, 0.3)
    reset_base_pose_range_yaw: tuple = (-3.14, 3.14)
    reset_base_velocity_range_x: tuple = (-0.5, 0.5)
    reset_base_velocity_range_y: tuple = (-0.5, 0.5)
    reset_base_velocity_range_z: tuple = (-0.5, 0.5)
    reset_base_velocity_range_roll: tuple = (-0.5, 0.5)
    reset_base_velocity_range_pitch: tuple = (-0.5, 0.5)
    reset_base_velocity_range_yaw: tuple = (-0.5, 0.5)
    reset_joint_position_range_hip: tuple = (-0.4, 0.4)
    reset_joint_position_range_thigh: tuple = (-1.5, 1.5)
    reset_joint_position_range_calf: tuple = (-1.5, 1.5)
    reset_joint_position_range_wheel: tuple = (-3.14, 3.14)
    reset_joint_velocity_range: tuple = (-2.0, 2.0)
    external_force_range: tuple = (-20.0, 20.0)
    external_torque_range: tuple = (-10.0, 10.0)


@configclass
class ThunderGaitFlatCommandParams:
    lin_vel_x: tuple = (-1.5, 1.5)
    lin_vel_y: tuple = (-1.0, 1.0)
    ang_vel_z: tuple = (-1.0, 1.0)


@configclass
class ThunderGaitFlatActuatorGains:
    hip_stiffness: float = 70.0
    hip_damping: float = 15.0
    thigh_stiffness: float = 100.0
    thigh_damping: float = 15.0
    calf_stiffness: float = 120.0
    calf_damping: float = 20.0
    wheel_stiffness: float = 0.0
    wheel_damping: float = 1.0


# =============================================================================
# Observations -- 单帧（history_length=1）
# =============================================================================

@configclass
class ThunderGaitFlatObservationsCfg(ObservationsCfg):

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1  # 单帧

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1  # 单帧

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# =============================================================================
# Actions
# =============================================================================

@configclass
class ThunderGaitFlatActionsCfg(ActionsCfg):
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"],
        scale=5.0,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


# =============================================================================
# Rewards -- ThunderHistRewardsCfg 所有项 + 步态门控新增项
# =============================================================================

@configclass
class ThunderGaitFlatRewardsCfg(RewardsCfg):
    """Thunder Hist rewards + gait-gated velocity tracking terms."""

    # inherited from ThunderHistRewardsCfg
    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )
    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )
    feet_impact_vel = RewTerm(
        func=mdp.feet_impact_vel,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
        },
    )
    adaptive_energy = RewTerm(
        func=mdp.adaptive_energy,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )

    # --- 步态门控奖励（新增） ---
    gait_gated_lin_vel = RewTerm(
        func=gait_rewards.GaitGatedVelocityReward,
        weight=0.0,
        params={
            "std": math.sqrt(0.5),
            "max_err": 0.2,
            "command_name": "base_velocity",
            "tracking_sigma": 1.0,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("", ""), ("", "")),  # __post_init__ 中覆盖
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    gait_gated_ang_vel = RewTerm(
        func=gait_rewards.GaitGatedAngVelReward,
        weight=0.0,
        params={
            "std": math.sqrt(0.5),
            "max_err": 0.2,
            "command_name": "base_velocity",
            "tracking_sigma": 1.0,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("", ""), ("", "")),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    wheel_lateral_slip = RewTerm(
        func=gait_rewards.wheel_lateral_slip,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
        },
    )


# =============================================================================
# Main Environment Config
# =============================================================================

@configclass
class ThunderGaitFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Thunder Gait -- 单帧平地验证任务。

    完全独立，不继承 rough_env_cfg。
    核心目的：验证方向感知步态门控奖励是否消除 spinning-in-place。
    """

    observations: ThunderGaitFlatObservationsCfg = ThunderGaitFlatObservationsCfg()
    actions: ThunderGaitFlatActionsCfg = ThunderGaitFlatActionsCfg()
    rewards: ThunderGaitFlatRewardsCfg = ThunderGaitFlatRewardsCfg()

    reward_weights: ThunderGaitFlatRewardWeights = ThunderGaitFlatRewardWeights()
    event_params: ThunderGaitFlatEventParams = ThunderGaitFlatEventParams()
    command_params: ThunderGaitFlatCommandParams = ThunderGaitFlatCommandParams()
    actuator_gains: ThunderGaitFlatActuatorGains = ThunderGaitFlatActuatorGains()

    base_link_name = "base_link"
    foot_link_name = ".*_foot"

    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names

    def __post_init__(self):
        super().__post_init__()

        # -------------------- Commands --------------------
        from robot_lab.tasks.manager_based.locomotion.velocity.mdp.commands import (
            HeadingBasedVelocityCommandCfg,
        )
        c = self.command_params
        self.commands.base_velocity = HeadingBasedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            heading_control_stiffness=0.5,
            debug_vis=False,
            ranges=HeadingBasedVelocityCommandCfg.Ranges(
                lin_vel_x=c.lin_vel_x,
                lin_vel_y=c.lin_vel_y,
                ang_vel_z=c.ang_vel_z,
                heading=(-math.pi, math.pi),
            ),
        )

        # -------------------- Scene --------------------
        self.scene.robot = THUNDER_CFG_V1.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.55)
        self.scene.robot.init_state.joint_pos = {
            "FR_hip_joint": -0.1, "FR_thigh_joint": -0.8, "FR_calf_joint": 1.8,
            "FL_hip_joint":  0.1, "FL_thigh_joint":  0.8, "FL_calf_joint": -1.8,
            "RR_hip_joint":  0.1, "RR_thigh_joint":  0.8, "RR_calf_joint": -1.8,
            "RL_hip_joint": -0.1, "RL_thigh_joint": -0.8, "RL_calf_joint":  1.8,
            ".*_foot_joint": 0.0,
        }
        self.scene.robot.init_state.joint_vel = {".*": 0.0}

        # 平地地形
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # 无 height scanner（父类有两个 RayCaster，都要禁）
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None

        # -------------------- Actuator Gains --------------------
        g = self.actuator_gains
        self.scene.robot.actuators["hip"].stiffness = g.hip_stiffness
        self.scene.robot.actuators["hip"].damping = g.hip_damping
        self.scene.robot.actuators["thigh"].stiffness = g.thigh_stiffness
        self.scene.robot.actuators["thigh"].damping = g.thigh_damping
        self.scene.robot.actuators["calf"].stiffness = g.calf_stiffness
        self.scene.robot.actuators["calf"].damping = g.calf_damping
        self.scene.robot.actuators["wheel"].stiffness = g.wheel_stiffness
        self.scene.robot.actuators["wheel"].damping = g.wheel_damping

        # -------------------- Observations --------------------
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_pos.params["wheel_asset_cfg"].joint_names = self.wheel_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["wheel_asset_cfg"].joint_names = self.wheel_joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        # -------------------- Actions --------------------
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # -------------------- Events --------------------
        e = self.event_params
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": e.reset_base_pose_range_x, "y": e.reset_base_pose_range_y,
                "z": e.reset_base_pose_range_z, "roll": e.reset_base_pose_range_roll,
                "pitch": e.reset_base_pose_range_pitch, "yaw": e.reset_base_pose_range_yaw,
            },
            "velocity_range": {
                "x": e.reset_base_velocity_range_x, "y": e.reset_base_velocity_range_y,
                "z": e.reset_base_velocity_range_z, "roll": e.reset_base_velocity_range_roll,
                "pitch": e.reset_base_velocity_range_pitch, "yaw": e.reset_base_velocity_range_yaw,
            },
        }
        self.events.randomize_reset_joints.func = reset_joints_with_type_specific_ranges
        self.events.randomize_reset_joints.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range_hip": e.reset_joint_position_range_hip,
            "position_range_thigh": e.reset_joint_position_range_thigh,
            "position_range_calf": e.reset_joint_position_range_calf,
            "position_range_wheel": e.reset_joint_position_range_wheel,
            "velocity_range": e.reset_joint_velocity_range,
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = e.external_force_range
        self.events.randomize_apply_external_force_torque.params["torque_range"] = e.external_torque_range

        # -------------------- Rewards --------------------
        w = self.reward_weights

        self.rewards.is_terminated.weight = w.is_terminated

        # 禁用原始速度追踪
        self.rewards.track_lin_vel_xy_exp.weight = w.track_lin_vel_xy_exp   # 0.0
        self.rewards.track_ang_vel_z_exp.weight = w.track_ang_vel_z_exp     # 0.0
        self.rewards.upward.weight = w.upward

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = w.lin_vel_z_l2
        self.rewards.ang_vel_xy_l2.weight = w.ang_vel_xy_l2
        self.rewards.flat_orientation_l2.weight = w.flat_orientation_l2
        self.rewards.base_height_l2.weight = w.base_height_l2
        self.rewards.base_height_l2.params["target_height"] = 0.55
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_height_l2.params["sensor_cfg"] = None  # flat terrain: no height scanner
        self.rewards.body_lin_acc_l2.weight = w.body_lin_acc_l2
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = w.joint_torques_l2
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = w.joint_torques_wheel_l2
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = w.joint_vel_l2
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = w.joint_vel_wheel_l2
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = w.joint_acc_l2
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = w.joint_acc_wheel_l2
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_pos_limits.weight = w.joint_pos_limits
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = w.joint_vel_limits
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = w.joint_power
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = w.stand_still
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.weight = w.joint_pos_penalty
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_vel_penalty.weight = w.wheel_vel_penalty
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.wheel_vel_zero_cmd = RewTerm(
            func=gait_rewards.wheel_vel_zero_cmd,
            weight=w.wheel_vel_zero_cmd,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
            },
        )
        self.rewards.joint_mirror.weight = w.joint_mirror
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        self.rewards.action_rate_l2.weight = w.action_rate_l2
        self.rewards.undesired_contacts.weight = w.undesired_contacts
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            f"^(?!.*{self.foot_link_name}).*"
        ]
        self.rewards.contact_forces.weight = w.contact_forces
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.contact_forces.params["threshold"] = 500.0
        self.rewards.feet_air_time.weight = w.feet_air_time
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = w.feet_contact
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = w.feet_contact_without_cmd
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = w.feet_stumble
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide = RewTerm(
            func=gait_rewards.wheeled_feet_slide,
            weight=w.feet_slide,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
            },
        )
        self.rewards.feet_impact_vel.weight = w.feet_impact_vel
        self.rewards.feet_impact_vel.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_impact_vel.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.excess_contact_force = RewTerm(
            func=mdp.excess_contact_force_l2,
            weight=w.excess_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.adaptive_energy.weight = w.adaptive_energy
        self.rewards.adaptive_energy.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.feet_height.weight = w.feet_height
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = w.feet_height_body
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]

        # ------ GaitReward: 驱动 trot 学习 ------
        self.rewards.feet_gait.weight = w.feet_gait
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("FL_foot", "RR_foot"),
            ("FR_foot", "RL_foot"),
        )
        self.rewards.feet_gait.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[self.foot_link_name]
        )

        # ------ 门控线速度追踪 ------
        self.rewards.gait_gated_lin_vel.weight = w.gait_gated_lin_vel
        self.rewards.gait_gated_lin_vel.params["synced_feet_pair_names"] = (
            ("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"),
        )
        self.rewards.gait_gated_lin_vel.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[self.foot_link_name]
        )

        # ------ 门控角速度追踪 ------
        self.rewards.gait_gated_ang_vel.weight = w.gait_gated_ang_vel
        self.rewards.gait_gated_ang_vel.params["synced_feet_pair_names"] = (
            ("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"),
        )
        self.rewards.gait_gated_ang_vel.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[self.foot_link_name]
        )

        # ------ 轮子侧向滑动惩罚 ------
        self.rewards.wheel_lateral_slip.weight = w.wheel_lateral_slip
        self.rewards.wheel_lateral_slip.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[self.foot_link_name]
        )
        self.rewards.wheel_lateral_slip.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[self.foot_link_name]
        )
        self.rewards.hip_pos_penalty = RewTerm(
            func=mdp.joint_pos_penalty,
            weight=w.hip_pos_penalty,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]),
                "stand_still_scale": 3.0,
                "velocity_threshold": 0.05,
                "command_threshold": 0.1,
            },
        )
        self.rewards.gait_phase_clock = RewTerm(
            func=gait_rewards.GaitPhaseClock,
            weight=w.gait_phase_clock,
            params={
                "f_base": 2.0,           # 基础步频 Hz
                "f_scale": 0.5,          # speed 每增 1 m/s → +0.5 Hz
                "f_min": 1.5,            # 最低 1.5 Hz
                "f_max": 3.5,            # 最高 3.5 Hz
                "sigma_f": 100.0,        # swing: exp(-|force|²/σ_f)，越小越严格
                "sigma_v": 1.0,          # stance: exp(-|vel|²/σ_v)，越小越严格
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "synced_feet_pair_names": (
                    ("FL_foot", "RR_foot"),
                    ("FR_foot", "RL_foot"),
                ),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
            },
        )
        self.rewards.feet_clearance = RewTerm(
            func=gait_rewards.feet_clearance,
            weight=w.feet_clearance,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
                "command_name": "base_velocity",
                "base_height": 0.03,
                "height_scale": 0.06,     # 速度缩放系数
                "max_height": 0.18,       # 高速时最大抬脚 18cm
            },
        )
        self.rewards.body_orientation_stability = RewTerm(
            func=gait_rewards.body_orientation_stability,
            weight=w.body_orientation_stability,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # ------ Symmetry rewards（默认 weight=0，Phase2 覆盖） ------
        # body_ids 必须按 FL, FR, RL, RR 顺序，preserve_order=True
        _sym_sensor = SceneEntityCfg(
            "contact_forces",
            body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        )
        _sym_asset = SceneEntityCfg(
            "robot",
            body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        )
        self.rewards.gait_contact_symmetry = RewTerm(
            func=gait_rewards.gait_contact_symmetry,
            weight=w.gait_contact_symmetry,
            params={
                "sensor_cfg": _sym_sensor,
                "command_name": "base_velocity",
                "command_threshold": 0.1,
            },
        )
        self.rewards.foot_height_symmetry = RewTerm(
            func=gait_rewards.foot_height_symmetry,
            weight=w.foot_height_symmetry,
            params={
                "sensor_cfg": _sym_sensor,
                "asset_cfg": _sym_asset,
                "command_name": "base_velocity",
                "command_threshold": 0.1,
            },
        )
        self.rewards.morphological_symmetry = RewTerm(
            func=gait_rewards.morphological_symmetry,
            weight=w.morphological_symmetry,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                    ],
                    preserve_order=True,
                ),
                "command_name": "base_velocity",
                "command_threshold": 0.1,
            },
        )
        self.rewards.foot_height_in_swing = RewTerm(
            func=gait_rewards.foot_height_in_swing,
            weight=w.foot_height_in_swing,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
                "command_name": "base_velocity",
                "min_height": 0.04,
                "max_height": 0.15,
                "command_threshold": 0.1,
            },
        )

        # -------------------- Terminations --------------------
        self.terminations.illegal_contact = None

        # -------------------- Curriculum --------------------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        self.curriculum.disturbance_levels = None
        self.curriculum.mass_randomization_levels = None
        self.curriculum.com_randomization_levels = None

        # 清零权重奖励项
        if self.__class__.__name__ in ("ThunderGaitFlatEnvCfg", "ThunderGaitPhase1EnvCfg"):
            self.disable_zero_weight_rewards()


# =============================================================================
# Phase 1 — 纯步态学习（不追踪速度）
# =============================================================================

@configclass
class ThunderGaitPhase1RewardWeights(ThunderGaitFlatRewardWeights):
    """Phase 1: 速度追踪 + 步态奖励同时训练（平地）。

    策略（参考 2401.12389 Stage I）:
      - 速度追踪保持不变（gait_gated_lin/ang_vel 继承基类）
      - feet_gait 加倍             (强力驱动对角 trot)
      - lateral_gated_air_time 加倍 (激励真正抬脚)
      - feet_clearance 加大        (强制摆腿离地)
      - foot_height_in_swing 新增  (持续梯度，打破腿静止陷阱)
      - 保持其余惩罚项不变

    Ref: 2401.12389 Stage I — velocity tracking + gait-related rewards on flat terrain.
    """

    # --- 速度追踪继承基类（8.0 / 3.0），不覆盖 ---

    # --- 加倍步态激励 ---
    feet_gait: float = 3.0                   # 原 1.5 → 3.0
    feet_clearance: float = 0.0             # 原 -1.5 → -2.0（更严格离地要求）

    # --- 保持 upward（鼓励站稳） ---
    upward: float = 2.0

    # --- 轮子约束不变 ---
    wheel_lateral_slip: float = -0.5
    wheel_vel_zero_cmd: float = -0.05

    # --- 机身稳定 ---
    body_orientation_stability: float = -0.5

    # --- 关键：悬空高度正奖励，打破腿静止陷阱 ---
    foot_height_in_swing: float = 2.0        # Phase1 专用，Phase2 (base) 里=0.0


@configclass
class ThunderGaitPhase1EnvCfg(ThunderGaitFlatEnvCfg):
    """Phase 1: 2000 iter 纯步态训练，不追踪速度。

    训练完成后用 checkpoint resume Phase 2（ThunderGaitFlatEnvCfg）。

    训练命令:
      CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/train.py \\
          --task GaitPhase1 --num_envs 4096 --headless --device cuda:0 \\
          --max_iterations 2000 --experiment_name thunder_gait_phase1

    Phase 2 resume 命令:
      CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/train.py \\
          --task Gait --num_envs 4096 --headless --device cuda:0 \\
          --max_iterations 10000 --experiment_name thunder_gait_phase2 \\
          --load_run thunder_gait_phase1 --checkpoint model_2000.pt
    """

    reward_weights: ThunderGaitPhase1RewardWeights = ThunderGaitPhase1RewardWeights()
