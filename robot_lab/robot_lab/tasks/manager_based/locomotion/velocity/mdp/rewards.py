# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2]
            - asset.data.root_lin_vel_b[:, :2]
        ),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.root_ang_vel_b[:, 2]
    )
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_cmd: float = 0.05,  # 防除零
    low_speed_threshold: float = 0.2,  # ★你指定: XY < 0.2 属于低速
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]

    # body frame 速度
    vel_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )

    cmd = env.command_manager.get_command(command_name)[:, :2]
    actual = vel_yaw[:, :2]

    abs_error = torch.abs(cmd - actual)
    rel_error = abs_error / torch.maximum(
        torch.abs(cmd), torch.tensor(min_cmd, device=env.device)
    )

    # ★ 动态权重 α: cmd 小 -> 严格(相对误差); cmd 大 -> 宽容(绝对误差)
    cmd_mag = torch.norm(cmd, dim=1)

    alpha = torch.clamp(cmd_mag / low_speed_threshold, 0.0, 1.0)
    alpha = alpha.unsqueeze(1)  # shape (N, 1)

    # ★ 归一化 (否则量纲不一致)
    max_abs = 0.5  # 线速度最大合理误差 (m/s)
    norm_abs = abs_error / max_abs
    norm_rel = rel_error

    # combin error = 动态权重混合
    combined = alpha * (norm_abs**2) + (1 - alpha) * (norm_rel**2)
    combined = torch.sum(combined, dim=1)  # x,y 两个方向求和

    reward = torch.exp(-combined / std**2)
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_cmd: float = 0.05,
    low_yaw_threshold: float = 0.4,  # ★你指定: yaw < 0.4 属于低速
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]

    cmd = env.command_manager.get_command(command_name)[:, 2]
    actual = asset.data.root_ang_vel_w[:, 2]

    abs_error = torch.abs(cmd - actual)
    rel_error = abs_error / torch.maximum(
        torch.abs(cmd), torch.tensor(min_cmd, device=env.device)
    )

    # ★ 动态权重 α: 低速时更严格 (相对误差权重变大)
    cmd_mag = torch.abs(cmd)  # yaw 指令大小

    alpha = torch.clamp(cmd_mag / low_yaw_threshold, 0.0, 1.0)

    # ★ 归一化: 单位一致性
    max_abs = 0.5  # 最大合理 yaw 误差 rad/s
    norm_abs = abs_error / max_abs
    norm_rel = rel_error

    # dynamic mixed error
    combined = alpha * (norm_abs**2) + (1 - alpha) * (norm_rel**2)

    reward = torch.exp(-combined / std**2)
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def joint_power(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(
            asset.data.joint_vel[:, asset_cfg.joint_ids]
            * asset.data.applied_torque[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= (
        torch.norm(env.command_manager.get_command(command_name), dim=1)
        < command_threshold
    )
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
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
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward



def wheel_vel_zero_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize wheel spinning ONLY when velocity command is near zero.

    Wheels must spin freely during locomotion — only penalize when cmd ~ 0.
    Prevents the policy exploiting free wheel spin for balance/perturbation.

    Returns: sum(|wheel_vel|) * zero_cmd_mask
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_norm = torch.linalg.norm(
        env.command_manager.get_command(command_name), dim=1
    )
    wheel_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    penalty = torch.sum(wheel_vel, dim=1)
    zero_cmd_mask = (cmd_norm < command_threshold).float()
    return penalty * zero_cmd_mask


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[
            cfg.params["sensor_cfg"].name
        ]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError(
                "This reward only supports gaits with two pairs of synchronized feet, like trotting."
            )
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[
            0
        ]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[
            0
        ]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(
            self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1]
        )
        sync_reward_1 = self._sync_reward_func(
            self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1]
        )
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(
            self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0]
        )
        async_reward_1 = self._async_reward_func(
            self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1]
        )
        async_reward_2 = self._async_reward_func(
            self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1]
        )
        async_reward_3 = self._async_reward_func(
            self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1]
        )
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(
            env.command_manager.get_command(self.command_name), dim=1
        )
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(
                cmd > self.command_threshold, body_vel > self.velocity_threshold
            ),
            sync_reward * async_reward,
            0.0,
        )
        reward *= (
            torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7)
            / 0.7
        )
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(
            torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2
        )
        se_contact = torch.clip(
            torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]),
            max=self.max_err**2,
        )
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(
            torch.square(air_time[:, foot_0] - contact_time[:, foot_1]),
            max=self.max_err**2,
        )
        se_act_1 = torch.clip(
            torch.square(contact_time[:, foot_0] - air_time[:, foot_1]),
            max=self.max_err**2,
        )
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


class GatedTrackLinVelXYExp(ManagerTermBase):
    """Track linear velocity reward gated by clock-anchored trot quality.

    V10 upgrade: 4-condition gate (ref: go2_trot_gating.pdf).
    V9 used only conditions 1+2 (instantaneous diagonal sync). Policy gamed this by
    satisfying the condition briefly at bounding transition moments (1-3 frames).

    V10 adds clock-phase conditions 3+4: actual contact must MATCH the gait clock's
    expected stance phase. Bounding has its own phase structure that cannot sustain
    alignment with the trot clock, preventing the gaming.

    Conditions (all must be true per env):
      1. FL contact == RR contact          (diagonal pair 0 in sync)
      2. FR contact == RL contact          (diagonal pair 1 in sync)
      3. FL/RR actual contact == clock expected stance for pair 0
      4. FR/RL actual contact == clock expected stance for pair 1

    Clock: per-env phase in [0, 1).
      - phase < 0.5  → pair-0 (FL+RR) expected in stance, pair-1 in swing
      - phase >= 0.5 → pair-1 (FR+RL) expected in stance, pair-0 in swing
    Phase advances dt/trot_period per step. Resets randomly on episode reset.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.trot_period: float = cfg.params.get("trot_period", 0.5)

        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.fl_id: int = pair_0[0]
        self.rr_id: int = pair_0[1]
        self.fr_id: int = pair_1[0]
        self.rl_id: int = pair_1[1]

        # Per-env gait phase [0, 1), init random so envs are phase-desynchronized
        self.gait_phase = torch.rand(env.num_envs, device=env.device)
        # Step dt = physics sim dt × decimation
        self.dt: float = env.cfg.sim.dt * env.cfg.decimation

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        synced_feet_pair_names,
        trot_period: float = 0.5,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # --- Velocity tracking ---
        asset: RigidObject = env.scene[asset_cfg.name]
        lin_vel_error = torch.sum(
            torch.square(
                env.command_manager.get_command(command_name)[:, :2]
                - asset.data.root_lin_vel_b[:, :2]
            ),
            dim=1,
        )
        tracking = torch.exp(-lin_vel_error / std**2)
        tracking *= (
            torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        )

        # --- Advance gait clock (per-env phase) ---
        self.gait_phase = (self.gait_phase + self.dt / self.trot_period) % 1.0
        # Randomize phase for envs that just reset (episode starts fresh)
        just_reset = env.episode_length_buf <= 1
        self.gait_phase = torch.where(
            just_reset, torch.rand_like(self.gait_phase), self.gait_phase
        )

        # --- Clock-expected contact states ---
        # pair-0 (FL+RR) in stance during phase < 0.5; pair-1 (FR+RL) in stance during >= 0.5
        expected_pair0_stance = self.gait_phase < 0.5   # True → FL+RR should be in contact
        expected_pair1_stance = self.gait_phase >= 0.5  # True → FR+RL should be in contact

        # --- Actual contact state ---
        contact_time = self.contact_sensor.data.current_contact_time
        in_contact = contact_time > 0.0
        fl = in_contact[:, self.fl_id]
        rr = in_contact[:, self.rr_id]
        fr = in_contact[:, self.fr_id]
        rl = in_contact[:, self.rl_id]

        # --- 4-condition trot check ---
        is_trot = (
            (fl == rr)                          # Cond 1: FL-RR diagonal sync
            & (fr == rl)                        # Cond 2: FR-RL diagonal sync
            & (fl == expected_pair0_stance)     # Cond 3: FL/RR match clock phase
            & (fr == expected_pair1_stance)     # Cond 4: FR/RL match clock phase
        )

        # Gate: only apply when moving; stand-still passes through unaffected
        cmd_norm = torch.linalg.norm(
            env.command_manager.get_command(command_name), dim=1
        )
        gate = torch.where(cmd_norm > 0.1, is_trot.float(), torch.ones_like(is_trot.float()))

        # Store phase on env so observation term can access it (V11)
        env._gait_phase = self.gait_phase.detach()

        return tracking * gate


def gait_clock_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sin/cos of gait clock phase for policy observations (V11).

    Allows policy to directly observe the clock phase and synchronize with it,
    raising gate satisfaction from ~50% (unknown phase) to near 100%.

    Returns shape (num_envs, 2): [sin(2π·phase), cos(2π·phase)]
    """
    if not hasattr(env, "_gait_phase"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    phase = env._gait_phase
    return torch.stack(
        [torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], dim=1
    )


def joint_mirror(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if (
        not hasattr(env, "joint_mirror_joints_cache")
        or env.joint_mirror_joints_cache is None
    ):
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair]
            for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                asset.data.joint_pos[:, joint_pair[0][0]]
                + asset.data.joint_pos[:, joint_pair[1][0]]
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def action_mirror(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if (
        not hasattr(env, "action_mirror_joints_cache")
        or env.action_mirror_joints_cache is None
    ):
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair]
            for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def action_sync(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if (
        not hasattr(env, "action_sync_joint_cache")
        or env.action_sync_joint_cache is None
    ):
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group]
            for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [
                torch.abs(env.action_manager.action[:, joint[0]])
                for joint in joint_group
            ],
            dim=1,
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def feet_air_time_positive_biped(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def feet_air_time_variance_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def feet_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    expect_contact_num: int,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= (
        torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    )
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def penalize_all_feet_air(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize when all feet are simultaneously in the air (bounding flight phase).

    Trot always has 2 feet on the ground — this penalty only fires during bounding.
    Uses instantaneous contact force (not first-contact transitions) to avoid
    conflicts with feet_air_time.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, F, 3)
    in_contact = torch.linalg.norm(net_forces, dim=-1) > force_threshold      # (N, F)
    num_in_contact = in_contact.sum(dim=1)                                     # (N,)
    return (num_in_contact == 0).float()                                       # 1 when all feet in air


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward sustained feet contact when the velocity command is near zero.

    This intentionally uses current contact state instead of first-contact
    events. Rewarding first contact lets a policy earn reward by repeatedly
    lifting and tapping feet, which is the opposite of quiet standing.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.linalg.norm(net_forces, dim=-1) > force_threshold
    reward = torch.sum(contact, dim=-1).float()
    reward *= (
        torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    )
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2
    )
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[
        :, asset_cfg.body_ids, :
    ] - asset.data.root_link_pos_w[:, :].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w),
            cur_footsteps_translated[:, i, :],
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones(
        [env.num_envs, 1], device=env.device
    )
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[
        :, asset_cfg.body_ids, :
    ] - asset.data.root_link_pos_w[:, :].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w),
            cur_footsteps_translated[:, i, :],
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones(
        [env.num_envs, 1], device=env.device
    )
    stance_length_tensor = stance_length * torch.ones(
        [env.num_envs, 1], device=env.device
    )

    desired_xs = torch.cat(
        [
            stance_length_tensor / 2,
            stance_length_tensor / 2,
            -stance_length_tensor / 2,
            -stance_length_tensor / 2,
        ],
        dim=1,
    )
    desired_ys = torch.cat(
        [
            stance_width_tensor / 2,
            -stance_width_tensor / 2,
            stance_width_tensor / 2,
            -stance_width_tensor / 2,
        ],
        dim=1,
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= (
        torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    )
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[
        :, asset_cfg.body_ids, :
    ] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(
        env.num_envs, len(asset_cfg.body_ids), 3, device=env.device
    )
    cur_footvel_translated = asset.data.body_lin_vel_w[
        :, asset_cfg.body_ids, :
    ] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(
        env.num_envs, len(asset_cfg.body_ids), 3, device=env.device
    )
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(
        footpos_in_body_frame[:, :, 2] - target_height
    ).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= (
        torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    )
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[
        :, asset_cfg.body_ids, :
    ] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(
        env.num_envs, len(asset_cfg.body_ids), 3, device=env.device
    )
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(
        torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)
    ).view(env.num_envs, -1)
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if (
            torch.isnan(ray_hits).any()
            or torch.isinf(ray_hits).any()
            or torch.max(torch.abs(ray_hits)) > 1e6
        ):
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def lin_vel_z_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def undesired_contacts(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(
            torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1
        )[0]
        > threshold
    )
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def flat_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def track_body_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of body height command using exponential kernel.

    This reward encourages the robot to match the commanded body height,
    enabling behaviors like crouching and standing tall.
    """
    # extract the used quantities
    asset: RigidObject = env.scene[asset_cfg.name]
    # get the commanded height (shape: num_envs,)
    target_height = env.command_manager.get_command(command_name)[:, 0]
    # get current body height
    current_height = asset.data.root_link_pos_w[:, 2]
    # compute height error
    height_error = torch.square(current_height - target_height)
    # exponential reward
    reward = torch.exp(-height_error / std**2)
    # scale by upright penalty (only give reward when robot is upright)
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def track_standing_posture_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of standing posture command using exponential kernel.

    Posture commands (one-hot encoded, shape: num_envs x 5):
    - [1,0,0,0,0]: Normal standing (gravity points down: [0, 0, -1])
    - [0,1,0,0,0]: Handstand (inverted, gravity points up: [0, 0, 1])
    - [0,0,1,0,0]: Left side standing (gravity points left: [0, -1, 0])
    - [0,0,0,1,0]: Right side standing (gravity points right: [0, 1, 0])
    - [0,0,0,0,1]: Front two legs standing (pitch up ~45deg, gravity: [0.7, 0, -0.7])
    """
    # extract the used quantities
    asset: RigidObject = env.scene[asset_cfg.name]
    # get the commanded posture (one-hot encoded, shape: num_envs x 5)
    posture_cmd_onehot = env.command_manager.get_command(command_name)
    # convert one-hot to indices (shape: num_envs,)
    posture_cmd = torch.argmax(posture_cmd_onehot, dim=1)
    # get current gravity direction in body frame (shape: num_envs, 3)
    gravity_b = asset.data.projected_gravity_b

    # Define target gravity directions for each posture
    target_gravity = torch.zeros(env.num_envs, 3, device=env.device)

    # Posture 0: Normal standing (gravity down)
    mask_0 = posture_cmd == 0
    target_gravity[mask_0] = torch.tensor([0.0, 0.0, -1.0], device=env.device)

    # Posture 1: Handstand (gravity up)
    mask_1 = posture_cmd == 1
    target_gravity[mask_1] = torch.tensor([0.0, 0.0, 1.0], device=env.device)

    # Posture 2: Left side standing
    mask_2 = posture_cmd == 2
    target_gravity[mask_2] = torch.tensor([0.0, -1.0, 0.0], device=env.device)

    # Posture 3: Right side standing
    mask_3 = posture_cmd == 3
    target_gravity[mask_3] = torch.tensor([0.0, 1.0, 0.0], device=env.device)

    # Posture 4: Front two legs standing (pitched up ~45deg)
    mask_4 = posture_cmd == 4
    target_gravity[mask_4] = torch.tensor([0.7071, 0.0, -0.7071], device=env.device)

    # Compute error as L2 distance between current and target gravity direction
    gravity_error = torch.sum(torch.square(gravity_b - target_gravity), dim=1)

    # Exponential reward
    reward = torch.exp(-gravity_error / (2 * std**2))

    return reward


def posture_stability_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize angular velocity to encourage stable postures."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # Penalize all angular velocities (roll, pitch, yaw)
    ang_vel = asset.data.root_ang_vel_b
    penalty = torch.sum(torch.square(ang_vel), dim=1)
    return penalty


def feet_air_time_when_needed(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    obstacle_height_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward feet air time only when obstacle is detected ahead (for wheeled-legged robots).

    This reward encourages the robot to:
    - Use wheels on flat terrain (penalize unnecessary leg lifting)
    - Lift legs only when obstacles/stairs are detected ahead

    Args:
        env: Environment instance.
        sensor_cfg: Configuration for the contact sensor.
        threshold: Minimum air time to start giving reward [s].
        obstacle_height_threshold: Height threshold to detect obstacle ahead [m].

    Returns:
        Reward for appropriate feet air time usage.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Compute feet air time (same as original feet_air_time)
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    # Reward only if the air time is above threshold
    air_time_reward = torch.sum((last_air_time - threshold).clip(min=0.0), dim=1)

    # Detect obstacles ahead using height_scan
    # height_scan shape: [num_envs, num_points]
    # We check the front region of the scan
    has_height_scan = (
        hasattr(env.scene, "height_scanner") and env.scene.height_scanner is not None
    )

    if has_height_scan:
        # Get height scan data
        height_scan = (
            env.scene.height_scanner.data.pos_w[:, :, 2]
            - env.scene.height_scanner.data.ray_hits_w[:, :, 2]
        )

        # Focus on front region (assume front is roughly center of scan)
        num_points = height_scan.shape[1]
        center_start = num_points // 3
        center_end = 2 * num_points // 3
        front_heights = height_scan[:, center_start:center_end]

        # Detect obstacle if max height in front region exceeds threshold
        max_front_height = torch.max(front_heights, dim=1)[0]
        has_obstacle = max_front_height > obstacle_height_threshold

        # Conditional reward:
        # - If obstacle ahead: reward air time (encourage leg lifting)
        # - If flat terrain: penalize air time (encourage wheel usage)
        reward = torch.where(
            has_obstacle,
            air_time_reward,  # Reward lifting when needed
            -0.5 * air_time_reward,  # Penalize unnecessary lifting
        )
    else:
        # Fallback: if no height scan, use air time reward as-is
        reward = air_time_reward

    return reward


def feet_height_when_needed(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    obstacle_height_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward feet height only when obstacle is detected ahead (for wheeled-legged robots).

    This encourages appropriate use of legs vs wheels based on terrain.

    Args:
        env: Environment instance.
        command_name: Name of the command for velocity check.
        asset_cfg: Asset configuration for feet.
        target_height: Target height for feet clearance [m].
        obstacle_height_threshold: Height threshold to detect obstacle [m].

    Returns:
        Conditional reward for feet height.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute feet height reward (similar to original feet_height)
    foot_z_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_z_error = torch.square(foot_z_pos - target_height)

    # Weight by foot velocity (only reward when foot is moving)
    foot_velocity = torch.linalg.norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2
    )
    foot_vel_tanh = torch.tanh(2.0 * foot_velocity)

    height_reward = torch.sum(foot_z_error * foot_vel_tanh, dim=1)

    # Detect obstacles ahead
    has_height_scan = (
        hasattr(env.scene, "height_scanner") and env.scene.height_scanner is not None
    )

    if has_height_scan:
        height_scan = (
            env.scene.height_scanner.data.pos_w[:, :, 2]
            - env.scene.height_scanner.data.ray_hits_w[:, :, 2]
        )

        # Front region detection
        num_points = height_scan.shape[1]
        center_start = num_points // 3
        center_end = 2 * num_points // 3
        front_heights = height_scan[:, center_start:center_end]

        max_front_height = torch.max(front_heights, dim=1)[0]
        has_obstacle = max_front_height > obstacle_height_threshold

        # Conditional reward
        reward = torch.where(
            has_obstacle,
            height_reward,  # Reward height when needed
            -0.3 * height_reward,  # Penalize when not needed
        )
    else:
        reward = height_reward

    # Apply velocity and orientation masks (same as original)
    reward *= (
        torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    )
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )

    return reward


def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the second-order derivative (acceleration) of actions using L2 squared kernel.

    This implements: (a_t - 2*a_{t-1} + a_{t-2})^2
    which penalizes sudden changes in action rate, promoting smoother trajectories.
    """
    # Initialize buffer for previous-previous action if not exists
    if not hasattr(env, "_prev_prev_action"):
        env._prev_prev_action = torch.zeros_like(env.action_manager.action)

    # Compute second-order difference (discrete acceleration)
    # a_t - 2*a_{t-1} + a_{t-2}
    action_accel = (
        env.action_manager.action
        - 2.0 * env.action_manager.prev_action
        + env._prev_prev_action
    )

    # Update the buffer for next iteration
    env._prev_prev_action[:] = env.action_manager.prev_action

    return torch.sum(torch.square(action_accel), dim=1)


def standing_still_stability(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Comprehensive penalty for instability when robot should be standing still.

    When the velocity command is near zero, this reward penalizes:
    1. Joint velocities (to prevent oscillations)
    2. Base angular velocities (roll/pitch to prevent wobbling)
    3. Base linear velocities (xy to prevent drifting, z to prevent bouncing)
    4. Action changes (to encourage steady control outputs)

    Args:
        env: The RL environment instance.
        command_name: Name of the velocity command to check.
        command_threshold: Threshold below which the robot should stand still.
        asset_cfg: Asset configuration for the robot.

    Returns:
        Penalty value (higher when more unstable during standing).
    """
    # Extract the robot asset
    asset: Articulation = env.scene[asset_cfg.name]

    # Check if command is near zero (robot should be standing still)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    is_standing = cmd_norm < command_threshold

    # Initialize penalty
    penalty = torch.zeros(env.num_envs, device=env.device)

    # Only apply penalty when robot should be standing still
    if torch.any(is_standing):
        # 1. Penalize joint velocities (prevent oscillations)
        joint_vel_penalty = torch.sum(
            torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
        )

        # 2. Penalize base angular velocities (prevent wobbling)
        ang_vel_penalty = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)

        # 3. Penalize base linear velocities
        # - xy: prevent drifting
        # - z: prevent bouncing
        lin_vel_xy_penalty = torch.sum(
            torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1
        )
        lin_vel_z_penalty = torch.square(asset.data.root_lin_vel_b[:, 2])

        # 4. Penalize action changes (encourage steady control)
        action_change_penalty = torch.sum(
            torch.square(env.action_manager.action - env.action_manager.prev_action),
            dim=1,
        )

        # Combine all penalties with appropriate weights
        total_penalty = (
            1.5 * joint_vel_penalty  # Heavily penalize joint motion
            + 2.0 * ang_vel_penalty  # Strongly penalize wobbling
            + 1.0 * lin_vel_xy_penalty  # Penalize drifting
            + 0.5 * lin_vel_z_penalty  # Lightly penalize z-motion
            + 1.0 * action_change_penalty  # Penalize control changes
        )

        # Apply the penalty only when standing still
        penalty = torch.where(is_standing, total_penalty, penalty)

    # Scale by orientation (only apply when robot is upright)
    penalty *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )

    return penalty


# ---------------------------------------------------------------------------
# Standing recovery reward terms (orientation, contact, stability, motion)
# ---------------------------------------------------------------------------


def recovery_base_orientation(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize horizontal tilt using squared xy components of body gravity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_xy = asset.data.projected_gravity_b[:, :2]
    penalty = torch.sum(torch.square(gravity_xy), dim=1)
    return penalty


def recovery_upright_orientation(
    env: ManagerBasedRLEnv,
    std: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Encourage upright posture using a Gaussian centered at g_z = -1."""
    asset: RigidObject = env.scene[asset_cfg.name]
    g_z = asset.data.projected_gravity_b[:, 2]
    reward = torch.exp(-torch.square(g_z + 1.0) / (2 * std**2))
    return reward


def recovery_target_posture(
    env: ManagerBasedRLEnv,
    target_joint_pos: torch.Tensor | None = None,
    epsilon: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Encourage convergence to a predefined standing posture when nearly upright."""
    asset: Articulation = env.scene[asset_cfg.name]
    current_q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if target_joint_pos is None:
        stand_pose = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    else:
        if target_joint_pos.ndim == 1:
            stand_pose = target_joint_pos.unsqueeze(0).to(current_q.device)
            if stand_pose.shape[0] == 1 and current_q.shape[0] > 1:
                stand_pose = stand_pose.repeat(current_q.shape[0], 1)
        else:
            stand_pose = target_joint_pos.to(current_q.device)
    posture_error = torch.sum(torch.square(current_q - stand_pose), dim=1)
    near_upright = (
        torch.abs(asset.data.projected_gravity_b[:, 2] + 1.0) < epsilon
    ).float()
    reward = torch.exp(-posture_error) * near_upright
    return reward


def recovery_feet_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0
) -> torch.Tensor:
    """Reward appropriate foot placement using binary contact indicators."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.linalg.norm(net_forces, dim=-1) > force_threshold
    reward = torch.sum(in_contact.float(), dim=1)
    return reward


def recovery_body_contact_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0
) -> torch.Tensor:
    """Penalize undesired body contacts excluding feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.linalg.norm(net_forces, dim=-1) > force_threshold
    penalty = torch.any(in_contact, dim=1).float()
    return penalty


def recovery_safety_force_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_clip: float = 20.0
) -> torch.Tensor:
    """Penalize horizontal forces at monitored contacts."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    horizontal_forces = torch.linalg.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=-1
    )
    penalty = torch.sum(torch.clamp(horizontal_forces, max=force_clip), dim=1)
    return penalty


def recovery_body_bias_penalty(
    env: ManagerBasedRLEnv,
    target_xy: torch.Tensor | None = None,
    clip_min: float = 0.0,
    clip_max: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal displacement of the body from a reference point."""
    asset: RigidObject = env.scene[asset_cfg.name]
    current_xy = asset.data.root_pos_w[:, :2]
    if target_xy is None:
        if hasattr(env.scene, "env_origins"):
            reference_xy = env.scene.env_origins[:, :2]
        else:
            reference_xy = torch.zeros_like(current_xy)
    else:
        reference_xy = target_xy.to(current_xy.device)
        if reference_xy.ndim == 1:
            reference_xy = reference_xy.unsqueeze(0).repeat(current_xy.shape[0], 1)
    displacement = torch.linalg.norm(current_xy - reference_xy, dim=1)
    penalty = torch.clamp(displacement, min=clip_min, max=clip_max)
    return penalty


def recovery_position_limit_penalty(
    env: ManagerBasedRLEnv,
    lower_limits: torch.Tensor | float,
    upper_limits: torch.Tensor | float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Count joint position limit violations."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower = (
        lower_limits
        if isinstance(lower_limits, torch.Tensor)
        else torch.tensor(lower_limits, device=joint_pos.device)
    )
    upper = (
        upper_limits
        if isinstance(upper_limits, torch.Tensor)
        else torch.tensor(upper_limits, device=joint_pos.device)
    )
    if lower.ndim == 0:
        lower = lower.expand_as(joint_pos)
    if upper.ndim == 0:
        upper = upper.expand_as(joint_pos)
    violations = torch.logical_or(joint_pos < lower, joint_pos > upper)
    penalty = torch.sum(violations.float(), dim=1)
    return penalty


def recovery_ang_vel_limit_penalty(
    env: ManagerBasedRLEnv,
    limit: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize excessive joint angular velocity beyond a threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    margin = torch.clamp(torch.abs(joint_vel) - limit, min=0.0)
    penalty = torch.sum(margin, dim=1)
    return penalty


def recovery_root_ang_vel_penalty(
    env: ManagerBasedRLEnv,
    limit: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Optional penalty on base/root angular velocity to keep torso calm."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_mag = torch.linalg.norm(asset.data.root_ang_vel_b, dim=1)
    penalty = torch.clamp(ang_vel_mag - limit, min=0.0)
    return penalty


def recovery_joint_velocity_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Quadratic penalty on joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.square(joint_vel), dim=1)
    return penalty


def recovery_joint_acc_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Quadratic penalty on joint accelerations estimated via finite differences."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_standing_prev_joint_vel"):
        env._standing_prev_joint_vel = torch.zeros_like(asset.data.joint_vel)
    joint_acc = (asset.data.joint_vel - env._standing_prev_joint_vel) / env.step_dt
    env._standing_prev_joint_vel = asset.data.joint_vel.clone()
    penalty = torch.sum(torch.square(joint_acc[:, asset_cfg.joint_ids]), dim=1)
    return penalty


def recovery_action_smoothing(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize abrupt action changes."""
    action_diff = env.action_manager.action - env.action_manager.prev_action
    penalty = torch.sum(torch.square(action_diff), dim=1)
    return penalty


def recovery_joint_torque_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Quadratic penalty on joint torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.square(torques), dim=1)
    return penalty


STANDING_RECOVERY_DEFAULT_WEIGHTS = {
    "base_orientation": -0.5,
    "upright_orientation": 6.0,
    "target_posture": 4.0,
    "feet_contact": 0.3,
    "body_contact": -0.2,
    "safety_force": -1.0e-2,
    "body_bias": -0.1,
    "position_limit": -1.0,
    "ang_vel_limit": -0.1,
    "joint_acc": -2.5e-6,
    "joint_vel": -1.0e-2,
    "action_smoothing": -0.01,
    "joint_torque": -5.0e-4,
}


def compute_standing_recovery_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_sensor_cfg: SceneEntityCfg | None = None,
    body_sensor_cfg: SceneEntityCfg | None = None,
    stability_sensor_cfg: SceneEntityCfg | None = None,
    target_joint_pos: torch.Tensor | None = None,
    target_xy: torch.Tensor | None = None,
    joint_pos_lower: torch.Tensor | float | None = None,
    joint_pos_upper: torch.Tensor | float | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Convenience helper that aggregates the proposed standing recovery rewards.

    Returns:
        total_reward: Weighted sum of all configured reward/penalty terms.
        term_dict: Dictionary with each individual (unweighted) term.
    """
    weights = weights or STANDING_RECOVERY_DEFAULT_WEIGHTS

    if foot_sensor_cfg is None and stability_sensor_cfg is None:
        raise ValueError(
            "At least one contact sensor cfg must be provided for standing rewards."
        )

    # Default fallbacks when specific sensor cfgs are omitted
    if foot_sensor_cfg is None:
        foot_sensor_cfg = stability_sensor_cfg
    if body_sensor_cfg is None:
        body_sensor_cfg = stability_sensor_cfg
    if stability_sensor_cfg is None:
        stability_sensor_cfg = foot_sensor_cfg

    terms: dict[str, torch.Tensor] = {}

    terms["base_orientation"] = recovery_base_orientation(env, asset_cfg)
    terms["upright_orientation"] = recovery_upright_orientation(
        env, asset_cfg=asset_cfg
    )
    terms["target_posture"] = recovery_target_posture(
        env, target_joint_pos, asset_cfg=asset_cfg
    )
    terms["feet_contact"] = recovery_feet_contact(env, foot_sensor_cfg)
    terms["body_contact"] = recovery_body_contact_penalty(env, body_sensor_cfg)
    terms["safety_force"] = recovery_safety_force_penalty(env, stability_sensor_cfg)
    terms["body_bias"] = recovery_body_bias_penalty(
        env, target_xy=target_xy, asset_cfg=asset_cfg
    )

    if joint_pos_lower is not None and joint_pos_upper is not None:
        terms["position_limit"] = recovery_position_limit_penalty(
            env, joint_pos_lower, joint_pos_upper, asset_cfg=asset_cfg
        )
    else:
        terms["position_limit"] = torch.zeros(env.num_envs, device=env.device)

    terms["ang_vel_limit"] = recovery_ang_vel_limit_penalty(env, asset_cfg=asset_cfg)
    terms["joint_acc"] = recovery_joint_acc_penalty(env, asset_cfg=asset_cfg)
    terms["joint_vel"] = recovery_joint_velocity_penalty(env, asset_cfg=asset_cfg)
    terms["action_smoothing"] = recovery_action_smoothing(env)
    terms["joint_torque"] = recovery_joint_torque_penalty(env, asset_cfg=asset_cfg)

    total = torch.zeros(env.num_envs, device=env.device)
    for name, value in terms.items():
        weight = weights.get(name, 0.0)
        if weight != 0.0:
            total = total + weight * value

    return total, terms


def flat_orientation_l2_adaptive(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalize non-flat base orientation, but relax constraint on harder terrains.
    Level 0-4: Full penalty (Mode: Elegant)
    Level 5-9: Reduced penalty (Mode: Survival)
    """
    # 1. Calculate standard penalty (projected gravity xy)
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    
    # 2. Get terrain levels (if available)
    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
        levels = env.scene.terrain.terrain_levels.float()
        # Scale factor: 1.0 at Level 0 -> 0.2 at Level 9
        # Formula: max(0.2, 1.0 - level * 0.08)
        scale = torch.clamp(1.0 - levels * 0.08, min=0.2, max=1.0)
    else:
        scale = 1.0
        
    return error * scale


def _terrain_adaptive_scale(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Helper: compute per-env scale factor based on terrain level.

    Level 0 -> 1.0 (full penalty, elegant mode)
    Level 9 -> 0.28 clamped to 0.2 (relaxed penalty, survival mode)
    """
    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
        levels = env.scene.terrain.terrain_levels.float()
        return torch.clamp(1.0 - levels * 0.08, min=0.2, max=1.0)
    return torch.ones(env.num_envs, device=env.device)


def ang_vel_xy_l2_adaptive(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity, relaxed on harder terrains.

    On slopes/stairs the body inevitably rotates more, so we reduce the penalty
    at higher terrain levels to avoid over-constraining the policy.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    # Upright gate: only penalize when roughly upright (same as original ang_vel_xy_l2)
    error *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return error * _terrain_adaptive_scale(env)


def lin_vel_z_l2_adaptive(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize z-axis base linear velocity, relaxed on harder terrains.

    On stairs/rough terrain vertical bouncing is unavoidable, so we reduce
    the penalty at higher terrain levels.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(asset.data.root_lin_vel_b[:, 2])
    # Upright gate (same as original lin_vel_z_l2)
    error *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return error * _terrain_adaptive_scale(env)


def action_rate_l2_adaptive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action rate of change, relaxed on harder terrains.

    On difficult terrains the robot needs fast corrective actions, so we
    reduce the smoothness penalty to allow more agile responses.
    """
    error = torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )
    return error * _terrain_adaptive_scale(env)


def feet_impact_vel(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize high vertical velocity of feet/wheels at the moment of ground contact.

    Only triggers on the first-contact frame (transition from air to ground).
    Penalizes the squared downward velocity, encouraging gentle touchdowns.
    Designed for both legged and wheeled-legged robots.

    Reference: walk-these-ways (Margolis et al., CoRL 2023)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    # Detect first-contact events (was in air, now touching)
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    # Get vertical (z) velocity of the foot/wheel bodies
    foot_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    # Only penalize downward velocity (negative z), clip to avoid rewarding upward
    downward_vel = torch.clamp(foot_vel_z, max=0.0)
    # Squared penalty, only on first-contact frames
    reward = torch.sum(torch.square(downward_vel) * first_contact, dim=1)
    return reward




def adaptive_energy(
    env: ManagerBasedRLEnv,
    sigma_x: float = 1000.0,
    sigma_z: float = 500.0,
    eps: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Adaptive energy regularization: approximate Cost of Transport as a reward.

    Rewards energy-efficient locomotion by normalizing mechanical power by motion magnitude.
    Moving fast allows higher power; moving slow or standing forces minimal power.

    Reference: UC Berkeley, ICRA 2025
    Formula: r = exp( -sum(|tau * dq|) / (sigma_x * |v_x| + sigma_z * |omega_z| + eps) )
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Instantaneous mechanical power: sum(|tau_i * dq_i|)
    power = torch.sum(
        torch.abs(
            asset.data.applied_torque[:, asset_cfg.joint_ids]
            * asset.data.joint_vel[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    # Motion magnitude: forward speed + yaw rate
    v_x = torch.abs(asset.data.root_lin_vel_b[:, 0])
    omega_z = torch.abs(asset.data.root_ang_vel_b[:, 2])
    motion_magnitude = sigma_x * v_x + sigma_z * omega_z + eps
    # Exponential reward: high when power/motion ratio is low
    reward = torch.exp(-power / motion_magnitude)
    return reward


def excess_contact_force_l2(
    env: ManagerBasedRLEnv,
    threshold: float = 0.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """AME-2 style: penalize contact forces exceeding robot weight.

    r = sum(max(|F| - robot_weight, 0)^2)

    Only penalizes impact/crash forces, not normal standing/walking forces.
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # Robot weight = total mass * gravity
    robot_weight = asset.data.default_mass.sum() * 9.81
    # Contact forces on specified bodies
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, B, 3)
    force_mag = torch.norm(forces, dim=-1)  # (N, B)
    # Only penalize excess over robot weight
    excess = torch.clamp(force_mag - robot_weight, min=0.0)
    return torch.sum(excess ** 2, dim=1)
