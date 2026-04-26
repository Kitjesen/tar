# Copyright (c) 2024-2026 Inovxio (穹沛科技)
# SPDX-License-Identifier: Apache-2.0
"""
Thunder Gait 专属奖励函数库。

这些函数只服务于 thunder_gait 任务，不属于全局 mdp。
如果这套步态门控方案被废弃，直接删除整个 thunder_gait/ 目录即可。

参考论文（见 papers/ 目录）:
  - 2409.15780  Barrier-Based Style Rewards (KAIST HOUND)
  - 2401.12389  Two-Step Reward for Natural Gaits
  - 2403.10723  Symmetry-Guided RL for Gait Transitions
  - 2601.10723  Energy-Efficient Wheeled Quadruped Locomotion
  - 2411.04787  AllGaits: CPG-based Gait Control

各函数按功能分区：
  §1  方向感知步态门控  (Direction-Aware Gait Gate)
  §2  轮子专属奖励     (Wheel-Specific)
  §3  足端轨迹质量     (Foot Quality)
  §4  机身稳定性       (Body Stability)
  §5  对称性奖励       (Symmetry — 2403.10723)
  §6  动作平滑度       (Action Smoothness)
  §7  轮足协同         (Wheel-Leg Coordination)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# §1  方向感知步态门控 (Direction-Aware Gait Gate)
# =============================================================================

class GaitGatedRewardBase(ManagerTermBase):
    """Base class for direction-aware gait-gated velocity rewards on wheeled-legged robots.

    Gate = (1 - lateral_ratio) + lateral_ratio * gait_quality
    where lateral_ratio = (|vy| + |yaw|) / (|vx| + |vy| + |yaw| + 1e-6)

    - Forward commands: gate ≈ 1 (wheels handle it, no trot required)
    - Lateral/yaw commands: gate scales with trot quality [0, 1]

    Ref: custom design, inspired by 2601.10723 wheel-leg energy analysis.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.command_name: str = cfg.params["command_name"]
        self.tracking_sigma: float = cfg.params["tracking_sigma"]
        self.command_threshold: float = cfg.params.get("command_threshold", 0.1)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        self.synced_feet_pairs = [
            self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0],
            self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0],
        ]

    def _compute_gate(self, cmd: torch.Tensor) -> torch.Tensor:
        vx = torch.abs(cmd[:, 0])
        vy = torch.abs(cmd[:, 1])
        yaw = torch.abs(cmd[:, 2])
        cmd_norm = torch.linalg.norm(cmd[:, :3], dim=1)
        gait_quality = self._compute_gait_quality()
        lateral_ratio = (vy + yaw) / (vx + vy + yaw + 1e-6)
        gate = (1.0 - lateral_ratio) + lateral_ratio * gait_quality
        return gate * (cmd_norm > self.command_threshold).float()

    def _compute_gait_quality(self) -> torch.Tensor:
        sync_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        async_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        return sync_0 * sync_1 * async_0 * async_1 * async_2 * async_3

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


class GaitGatedVelocityReward(GaitGatedRewardBase):
    """Gait-gated linear velocity (xy) tracking reward.

    Replaces track_lin_vel_xy_exp. Forward: always full gate. Lateral/yaw: gated by trot quality.
    Includes upright bonus (projected_gravity_b z-component) to discourage falling over while moving.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        command_name: str,
        tracking_sigma: float,
        synced_feet_pair_names,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        command_threshold: float = 0.1,
    ) -> torch.Tensor:
        cmd = env.command_manager.get_command(self.command_name)
        gate = self._compute_gate(cmd)
        lin_vel_error = torch.sum(
            torch.square(cmd[:, :2] - self.asset.data.root_com_lin_vel_b[:, :2]), dim=1
        )
        reward = torch.exp(-lin_vel_error / self.tracking_sigma) * gate
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward


class GaitGatedAngVelReward(GaitGatedRewardBase):
    """Gait-gated angular velocity (yaw) tracking reward.

    Replaces track_ang_vel_z_exp. Yaw commands are gated so spinning in place requires trot.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        command_name: str,
        tracking_sigma: float,
        synced_feet_pair_names,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        command_threshold: float = 0.1,
    ) -> torch.Tensor:
        cmd = env.command_manager.get_command(self.command_name)
        gate = self._compute_gate(cmd)
        ang_vel_error = torch.square(cmd[:, 2] - self.asset.data.root_com_ang_vel_b[:, 2])
        reward = torch.exp(-ang_vel_error / self.tracking_sigma) * gate
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward


# =============================================================================
# §2  轮子专属奖励 (Wheel-Specific)
# =============================================================================

def wheel_lateral_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral slip at wheel contact points.

    Wheels roll along body x-axis. y-component of wheel velocity = lateral slip.
    Penalized only when wheels are in contact.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    )
    cur_footvel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_body = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_body[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, cur_footvel_w[:, i, :])
    lateral_slip = torch.abs(footvel_body[:, :, 1])
    return torch.sum(lateral_slip * contacts.float(), dim=1)


def wheel_vel_zero_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize wheel joint velocity when standing still (command below threshold).

    Teaches the robot to brake cleanly when no velocity command is given.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.linalg.norm(cmd[:, :3], dim=1)
    no_cmd = (cmd_norm < command_threshold).float()
    wheel_vel = torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return wheel_vel * no_cmd


def wheel_rolling_efficiency(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_radius: float = 0.075,
) -> torch.Tensor:
    """Reward pure rolling condition: wheel angular velocity * radius ≈ body forward velocity.

    Pure rolling means no slip — energy-efficient and mechanically sound.
    Positive reward when slip ratio < threshold; penalize excess slip.

    Ref: 2601.10723 — wheeled quadruped energy efficiency analysis.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Body forward velocity in body frame (x-axis)
    body_vx = asset.data.root_com_lin_vel_b[:, 0]  # (num_envs,)

    # Wheel angular velocity → expected linear velocity if pure rolling
    wheel_ang_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]  # (num_envs, num_wheels)
    expected_vx = wheel_ang_vel * wheel_radius  # (num_envs, num_wheels)

    # Contact mask
    in_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    )

    # Slip = deviation from pure rolling
    slip = torch.abs(expected_vx - body_vx.unsqueeze(1))
    rolling_reward = torch.exp(-slip * 2.0)  # smooth reward, peak at slip=0
    return torch.sum(rolling_reward * in_contact.float(), dim=1)


# =============================================================================
# §3  足端轨迹质量 (Foot Quality)
# =============================================================================

def foot_height_in_swing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    min_height: float = 0.04,
    max_height: float = 0.15,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward foot height during swing phase, gated by lateral/yaw command ratio.

    Unlike lateral_gated_air_time (threshold reward at landing), this gives a
    *continuous* gradient signal the moment any foot lifts off the ground.
    Reward = clamp((foot_z - min_height) / range, 0, 1) * lateral_ratio * in_swing

    This is critical for bootstrapping trot: GaitReward alone can be satisfied
    with all legs on the ground. foot_height_in_swing forces actual leg elevation.

    Use in Phase 1 (gait-only) with weight ~2.0 to break the static-leg trap.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.linalg.norm(cmd[:, :3], dim=1)
    vx = torch.abs(cmd[:, 0])
    vy = torch.abs(cmd[:, 1])
    yaw = torch.abs(cmd[:, 2])
    lateral_ratio = (vy + yaw) / (vx + vy + yaw + 1e-6)
    has_cmd = (cmd_norm > command_threshold).float()

    in_swing = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] < 1.0
    )
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_ratio = torch.clamp(
        (foot_z - min_height) / (max_height - min_height + 1e-6), 0.0, 1.0
    )
    reward = torch.sum(height_ratio * in_swing.float(), dim=1)
    return reward * lateral_ratio * has_cmd


def lateral_gated_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward foot air time, scaled by how lateral/yaw-dominant the command is.

    Pure forward → lateral_ratio ≈ 0 → no air time reward (wheels handle it).
    Lateral/yaw → lateral_ratio ≈ 1 → full air time reward (trot required).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.linalg.norm(cmd[:, :3], dim=1)
    vx = torch.abs(cmd[:, 0])
    vy = torch.abs(cmd[:, 1])
    yaw = torch.abs(cmd[:, 2])
    lateral_ratio = (vy + yaw) / (vx + vy + yaw + 1e-6)
    has_cmd = (cmd_norm > command_threshold).float()
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    air_reward = torch.clamp(air_time - threshold, min=0.0)
    base_reward = torch.sum(air_reward * in_contact.float(), dim=1)
    return base_reward * lateral_ratio * has_cmd


def feet_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    base_height: float = 0.03,
    height_scale: float = 0.08,
    max_height: float = 0.14,
) -> torch.Tensor:
    """Penalize insufficient foot clearance during swing phase.

    Target clearance scales with total command speed:
        min_height = clip(base_height + height_scale * sqrt(vx² + vy² + yaw²), base_height, max_height)

    - Stationary: min_height ≈ 0.03m (nearly no constraint)
    - Fast motion: min_height grows to encourage real leg arcs (up to max_height)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.sqrt(cmd[:, 0] ** 2 + cmd[:, 1] ** 2 + cmd[:, 2] ** 2)
    min_height = torch.clamp(base_height + height_scale * cmd_speed, max=max_height)
    in_swing = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] < 1.0
    )
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    clearance_deficit = torch.clamp(min_height.unsqueeze(1) - foot_z, min=0.0)
    return torch.sum(clearance_deficit * in_swing.float(), dim=1)


def feet_clearance_barrier(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    base_height: float = 0.03,
    height_scale: float = 0.08,
    max_height: float = 0.14,
    margin: float = 0.02,
) -> torch.Tensor:
    """Barrier-function version of feet_clearance (smooth, curriculum-friendly).

    Uses log-barrier: r = -log(deficit/margin + 1) instead of hard clamp.
    margin can be annealed over training for automatic curriculum:
        early training: margin=0.05 (soft)
        late training:  margin=0.01 (strict)

    Ref: 2409.15780 Barrier-Based Style Rewards (KAIST HOUND).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    lateral_speed = torch.sqrt(cmd[:, 1] ** 2 + cmd[:, 2] ** 2)
    min_height = torch.clamp(base_height + height_scale * lateral_speed, max=max_height)
    in_swing = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] < 1.0
    )
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    deficit = torch.clamp(min_height.unsqueeze(1) - foot_z, min=0.0)
    barrier = torch.log(deficit / margin + 1.0)
    return torch.sum(barrier * in_swing.float(), dim=1)


def feet_impact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize sudden contact force spikes at foot touchdown (soft landing).

    Computes first-order temporal difference of contact force norm.
    Penalizes only force increases (touchdown impacts), not decreases (liftoff).

    Ref: 2401.12389 Two-Step Reward; general locomotion practice.
    Note: weight should be small (e.g. -0.01) since forces are in Newtons (large values).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    # history dim: [0]=latest, [1]=t-1
    force_now = forces[:, 0, :, :].norm(dim=-1)
    force_prev = forces[:, 1, :, :].norm(dim=-1)
    delta = torch.clamp(force_now - force_prev, min=0.0)  # only increases
    return torch.sum(delta, dim=1)


def wheeled_feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize true foot sliding — correct for wheeled quadrupeds.

    Standard feet_slide uses body-relative velocity, which incorrectly penalizes
    planted stance legs during lateral/yaw motion (stance foot world vel ≈ 0, but
    body moves → relative vel is large → false positive).

    Fix: take min(world_vel, relative_vel). A foot that is either:
      - planted on ground (world vel ≈ 0), OR
      - rolling with body (relative vel ≈ 0)
    is NOT sliding. Only penalize when BOTH are nonzero.

    | Scenario              | world | relative | min  | Correct? |
    |-----------------------|-------|----------|------|----------|
    | Forward, wheel rolls  | 1.5   | 0        | 0    | ✅       |
    | Lateral, foot planted | 0     | 1.0      | 0    | ✅       |
    | Yaw, foot planted     | 0     | 0.4      | 0    | ✅       |
    | True sliding          | 0.7   | 0.7      | 0.7  | ✅       |
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset: RigidObject = env.scene[asset_cfg.name]

    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    body_vel_w = asset.data.root_lin_vel_w[:, :2].unsqueeze(1)

    world_vel = foot_vel_w.norm(dim=-1)
    relative_vel = (foot_vel_w - body_vel_w).norm(dim=-1)
    slide_vel = torch.min(world_vel, relative_vel)

    reward = torch.sum(slide_vel * contacts.float(), dim=1)
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


# =============================================================================
# §3.5  相位时钟 (Gait Phase Clock)
# =============================================================================


class GaitPhaseClock(ManagerTermBase):
    """Speed-adaptive phase clock with force/velocity enforcement (arXiv 2401.12389).

    Maintains phase φ ∈ [0, 2π) per env, updated each step:
        φ += 2π × f(speed) × dt

    Trot pattern:
        - Pair 0 (FL, RR): swing when sin(φ) > 0, stance when sin(φ) < 0
        - Pair 1 (FR, RL): opposite

    Reward = swing_reward + stance_reward (per foot, summed):
        swing:  (1 - C_des) × exp(-|foot_force|² / σ_f)   → 该抬脚时，惩罚接触力
        stance: C_des × exp(-|foot_vel|² / σ_v)            → 该踩地时，惩罚脚速度

    Ref: Table I of "Two-Step Reward for Natural Gaits" (2401.12389)
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.f_base = cfg.params["f_base"]
        self.f_scale = cfg.params["f_scale"]
        self.f_min = cfg.params["f_min"]
        self.f_max = cfg.params["f_max"]
        self.sigma_f = cfg.params.get("sigma_f", 100.0)   # force penalty sharpness
        self.sigma_v = cfg.params.get("sigma_v", 1.0)     # velocity penalty sharpness
        self.command_name = cfg.params["command_name"]
        self.command_threshold = cfg.params.get("command_threshold", 0.1)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: RigidObject = env.scene[cfg.params["asset_cfg"].name]

        synced_pairs = cfg.params["synced_feet_pair_names"]
        self.pair0_ids = self.contact_sensor.find_bodies(synced_pairs[0])[0]
        self.pair1_ids = self.contact_sensor.find_bodies(synced_pairs[1])[0]
        self.all_foot_ids = list(self.pair0_ids) + list(self.pair1_ids)
        self.n_feet = len(self.all_foot_ids)

        # Phase buffer — random init for diversity
        self.phase = torch.rand(env.num_envs, device=env.device) * 2 * math.pi
        env.gait_phase = self.phase

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        f_base: float,
        f_scale: float,
        f_min: float,
        f_max: float,
        command_name: str,
        synced_feet_pair_names,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        sigma_f: float = 100.0,
        sigma_v: float = 1.0,
        command_threshold: float = 0.1,
    ) -> torch.Tensor:
        cmd = env.command_manager.get_command(self.command_name)
        speed = torch.sqrt(cmd[:, 0] ** 2 + cmd[:, 1] ** 2 + cmd[:, 2] ** 2)

        # Adaptive frequency
        freq = torch.clamp(
            self.f_base + self.f_scale * speed, self.f_min, self.f_max
        )

        # Update phase
        self.phase = (self.phase + 2 * math.pi * freq * self.dt) % (2 * math.pi)

        # Reset phase on env reset
        just_reset = env.episode_length_buf == 0
        if just_reset.any():
            n = just_reset.sum()
            self.phase[just_reset] = torch.rand(n, device=self.phase.device) * 2 * math.pi

        env.gait_phase = self.phase

        # --- Per-foot desired contact (C_des) ---
        # shape: (num_envs, n_feet)
        sin_phase = torch.sin(self.phase)
        # Pair 0: stance when sin(φ) < 0 → C_des = 1
        # Pair 1: stance when sin(φ) > 0 → C_des = 1
        c_des = torch.zeros(env.num_envs, self.n_feet, device=env.device)
        n0 = len(self.pair0_ids)
        c_des[:, :n0] = (sin_phase < 0).float().unsqueeze(1).expand(-1, n0)
        c_des[:, n0:] = (sin_phase > 0).float().unsqueeze(1).expand(-1, self.n_feet - n0)

        # --- Actual foot forces (N) ---
        foot_forces = self.contact_sensor.data.net_forces_w[:, self.all_foot_ids, :]
        foot_force_mag = foot_forces.norm(dim=-1)  # (num_envs, n_feet)

        # --- Actual foot velocities (m/s) ---
        foot_vel = self.asset.data.body_lin_vel_w[:, self.all_foot_ids, :2]
        foot_vel_mag = foot_vel.norm(dim=-1)  # (num_envs, n_feet)

        # --- Swing reward: 该抬脚时惩罚接触力 ---
        # (1 - C_des) × exp(-|f|² / σ_f)
        swing_reward = (1.0 - c_des) * torch.exp(-foot_force_mag ** 2 / self.sigma_f)

        # --- Stance reward: 该踩地时惩罚脚速度 ---
        # C_des × exp(-|v|² / σ_v)
        stance_reward = c_des * torch.exp(-foot_vel_mag ** 2 / self.sigma_v)

        # Sum over feet, average
        reward = (swing_reward + stance_reward).sum(dim=1) / self.n_feet

        # Only enforce when command is active
        cmd_norm = torch.linalg.norm(cmd[:, :3], dim=1)
        has_cmd = (cmd_norm > self.command_threshold).float()

        return reward * has_cmd


def gait_phase_observation(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Gait phase clock observation: [sin(φ), cos(φ)].

    For trot, sin(φ+π) = -sin(φ), so 2 dims is sufficient.
    Policy learns: sin(φ)>0 → pair0 swing, pair1 stance; sin(φ)<0 → opposite.

    Must be used with GaitPhaseClock reward (stores phase on env.gait_phase).
    """
    if not hasattr(env, "gait_phase"):
        env.gait_phase = torch.zeros(env.num_envs, device=env.device)
    phase = env.gait_phase
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)


# =============================================================================
# §4  机身稳定性 (Body Stability)
# =============================================================================

def body_orientation_stability(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body roll and pitch — main source of visual unsteadiness during trot.

    More sensitive to small tilts than flat_orientation_l2.
    projected_gravity_b xy components represent body tilt away from upright.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    grav_b = asset.data.projected_gravity_b
    return torch.sum(torch.square(grav_b[:, :2]), dim=1)


# =============================================================================
# §5  对称性奖励 (Symmetry — 2403.10723)
#
# 三种对称：
#   A. gait_contact_symmetry    — 对角接触时序对称（管"节奏均不均衡"）
#   B. foot_height_symmetry     — 左右摆腿高度对称（管"视觉上歪不歪"）
#   C. morphological_symmetry   — 关节镜像对称（管"腿型自不自然"）
#
# 三者均加 yaw 门控：转弯时左右本就不对称，不强压。
# 推荐权重：A=-0.5, B=-0.3, C=-0.1；先加 A+B，效果好再加 C。
# =============================================================================

def gait_contact_symmetry(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """A. 对角接触时序对称 — 管 trot 节奏均不均衡。

    trot 正确时：FL & RR 同时落地、FR & RL 同时落地（对角同相）。
    惩罚：对角对的接触状态不一致。

    sensor_cfg.body_ids 顺序必须是 [FL_foot, FR_foot, RL_foot, RR_foot]。
    转弯时 yaw 越大门控越弱，避免妨碍转向。

    Ref: 2403.10723 Symmetry-Guided RL.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.linalg.norm(cmd[:, :3], dim=1)
    has_cmd = (cmd_norm > command_threshold).float()

    # 接触状态：force > 1N → 在地
    forces = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1).max(dim=1)[0]
    )  # (num_envs, 4)
    contact = (forces > 1.0).float()  # FL=0, FR=1, RL=2, RR=3

    # 对角对：FL(0)-RR(3) 应同相，FR(1)-RL(2) 应同相
    asym = (
        torch.square(contact[:, 0] - contact[:, 3]) +  # FL vs RR
        torch.square(contact[:, 1] - contact[:, 2])    # FR vs RL
    )

    # 转弯时放松对称约束
    gate = torch.exp(-2.0 * torch.abs(cmd[:, 2]))
    return asym * gate * has_cmd


def foot_height_symmetry(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """B. 左右摆腿高度对称 — 管步态视觉上歪不歪。

    摆腿时：前左脚高度 ≈ 前右脚高度，后左 ≈ 后右。
    惩罚：同侧前后腿高度差异（一边拖地一边正常）。

    sensor_cfg.body_ids / asset_cfg.body_ids 顺序：[FL_foot, FR_foot, RL_foot, RR_foot]。
    转弯时放松（内外侧腿高度可以不同）。

    Ref: 2403.10723; general locomotion practice.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.linalg.norm(cmd[:, :3], dim=1)
    has_cmd = (cmd_norm > command_threshold).float()

    # 仅在摆腿期（无接触）计算高度差
    forces = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1).max(dim=1)[0]
    )
    in_swing = (forces < 1.0).float()  # (num_envs, 4)

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, 4)

    # 左右高度差：FL(0) vs FR(1)，RL(2) vs RR(3)
    asym = (
        torch.square(foot_z[:, 0] - foot_z[:, 1]) * in_swing[:, 0] * in_swing[:, 1] +
        torch.square(foot_z[:, 2] - foot_z[:, 3]) * in_swing[:, 2] * in_swing[:, 3]
    )

    gate = torch.exp(-2.0 * torch.abs(cmd[:, 2]))
    return asym * gate * has_cmd


def morphological_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """C. 关节镜像对称 — 管腿型自不自然。

    左右对应关节应镜像：髋关节反号，大腿/小腿同号。
    权重宜小（-0.1），避免把策略锁死在完全对称。

    asset_cfg.joint_ids 顺序：
        [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
         RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]

    Ref: 2403.10723 Symmetry-Guided RL for Quadruped Gait Transitions.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.linalg.norm(cmd[:, :3], dim=1)
    has_cmd = (cmd_norm > command_threshold).float()

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (num_envs, 12)
    if joint_pos.shape[1] < 12:
        return torch.zeros(env.num_envs, device=env.device)

    fl = joint_pos[:, 0:3]   # FL: hip, thigh, calf
    fr = joint_pos[:, 3:6]   # FR
    rl = joint_pos[:, 6:9]   # RL
    rr = joint_pos[:, 9:12]  # RR

    # 髋关节反号（坐标系镜像），大腿/小腿同号
    sign = joint_pos.new_tensor([-1.0, -1.0, -1.0])  # Thunder: URDF左右axis反号编码

    asym_front = torch.sum(torch.square(fl - fr * sign), dim=1)
    asym_rear = torch.sum(torch.square(rl - rr * sign), dim=1)

    gate = torch.exp(-2.0 * torch.abs(cmd[:, 2]))
    return (asym_front + asym_rear) * gate * has_cmd


# =============================================================================
# §6  动作平滑度 (Action Smoothness)
# =============================================================================

class ActionJerkPenalty(ManagerTermBase):
    """Penalize second-order action derivative (jerk) to reduce jerky motion.

    Stores prev_prev_action internally each step.
    jerk = action - 2*prev_action + prev_prev_action

    Ref: general locomotion practice; complements action_rate_l2.
    Note: weight should be small (e.g. -0.001).
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        num_actions = env.action_manager.action.shape[1]
        self._prev_prev_action = torch.zeros(env.num_envs, num_actions, device=env.device)
        self._prev_action = torch.zeros(env.num_envs, num_actions, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor:
        action = env.action_manager.action
        jerk = action - 2.0 * self._prev_action + self._prev_prev_action
        penalty = torch.sum(torch.square(jerk), dim=1)
        # Update history
        self._prev_prev_action = self._prev_action.clone()
        self._prev_action = action.clone()
        return penalty


# =============================================================================
# §7  轮足协同 (Wheel-Leg Coordination)
# =============================================================================

def wheel_leg_sync(
    env: ManagerBasedRLEnv,
    leg_sensor_cfg: SceneEntityCfg,
    wheel_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize lateral wheel forces when the ipsilateral leg is in swing phase.

    When a leg swings, its paired wheel should not resist the motion with lateral force.
    High lateral wheel force during swing = wheel fighting against leg = wasted energy + instability.

    leg_sensor_cfg.body_ids:   [FL_foot, FR_foot, RL_foot, RR_foot]
    wheel_sensor_cfg.body_ids: [FL_wheel, FR_wheel, RL_wheel, RR_wheel] (same order)

    Ref: custom design for wheeled-legged robots; inspired by 2601.10723.
    """
    leg_sensor: ContactSensor = env.scene.sensors[leg_sensor_cfg.name]
    wheel_sensor: ContactSensor = env.scene.sensors[wheel_sensor_cfg.name]

    # Leg in swing = no contact force on leg
    leg_forces = leg_sensor.data.net_forces_w_history[:, :, leg_sensor_cfg.body_ids, :]
    in_swing = (leg_forces.norm(dim=-1).max(dim=1)[0] < 1.0)  # (num_envs, num_legs)

    # Lateral force on wheel (y-axis in world frame)
    wheel_forces = wheel_sensor.data.net_forces_w[:, wheel_sensor_cfg.body_ids, :]
    wheel_lateral = torch.abs(wheel_forces[:, :, 1])  # (num_envs, num_wheels)

    # Penalize lateral wheel force when paired leg is swinging
    return torch.sum(wheel_lateral * in_swing.float(), dim=1)



def joint_pos_penalty_no_fall_filter(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Joint position deviation penalty WITHOUT upright-factor filtering.

    Same as `joint_pos_penalty` but penalises even when robot has fallen
    (we still want joints close to default after a fall).

    - Stand-still scaling preserved (cmd/velocity gated multiplier)
    - Upright modulation REMOVED (was: `reward *= clamp(-grav_z, 0, 0.7) / 0.7`)

    Use this for grouped joint deviation penalties (hip / thigh / calf)
    where falling should not give the robot a free pass to splay joints.
    """
    asset = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    return reward



def wheel_vel_motion_aware(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    command_threshold: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize wheel velocity with stand-still aware multiplier.

    Merges old foot_vel_penalty (always-on L2) and wheel_vel_zero_cmd (cmd=0
    L1) into one term. Light penalty when commanded to move; heavy when
    commanded to stand still (or already nearly static).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    base_reward = torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        base_reward,
        stand_still_scale * base_reward,
    )
    return reward
