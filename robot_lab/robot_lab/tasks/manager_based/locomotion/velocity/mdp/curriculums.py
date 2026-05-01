# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> torch.Tensor:
    """Curriculum based on the reward for tracking velocity commands.
    
    This function increases the range of velocity commands when the robot performs well.
    """
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    # 返回当前线性速度上限作为标量值，供课程学习管理器记录
    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    # 🔧 修正：限制在有效范围内 [0, max_terrain_level-1] 后再返回
    max_level = terrain.max_terrain_level - 1
    clamped_levels = torch.clamp(terrain.terrain_levels, 0, max_level)
    return torch.mean(clamped_levels.float())


def disturbance_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    force_range_start: tuple[float, float] = (5.0, 10.0),
    force_range_end: tuple[float, float] = (20.0, 50.0),
    performance_threshold: float = 0.8,
) -> torch.Tensor:
    """Simple disturbance curriculum that gradually increases external force disturbances."""
    # Return zero for now - simplified implementation
    return torch.tensor(0.0, device=env.device)


def mass_randomization_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    mass_range_start: tuple[float, float] = (0.9, 1.1),
    mass_range_end: tuple[float, float] = (0.7, 1.3),
    performance_threshold: float = 0.8,
) -> torch.Tensor:
    """Simple mass randomization curriculum."""
    # Return zero for now - simplified implementation
    return torch.tensor(0.0, device=env.device)


def com_randomization_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    com_range_start: dict[str, tuple[float, float]] = None,
    com_range_end: dict[str, tuple[float, float]] = None,
    performance_threshold: float = 0.8,
) -> torch.Tensor:
    """Simple COM randomization curriculum."""
    # Return zero for now - simplified implementation
    return torch.tensor(0.0, device=env.device)


def reward_weight_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_name: str,
    start_value: float,
    end_value: float,
    start_iter: int = 0,
    end_iter: int = 15000,
) -> torch.Tensor:
    """Progressive reward weight curriculum - gradually increase penalty weights.
    
    This implements a linear interpolation from start_value to end_value over training iterations.
    
    Args:
        env: The RL environment.
        env_ids: Not used (kept for interface compatibility).
        reward_name: Name of the reward term to modify.
        start_value: Initial weight (usually small penalty for exploration).
        end_value: Final weight (full penalty for optimization).
        start_iter: Iteration to start the curriculum (default: 0).
        end_iter: Iteration to reach end_value (default: 15000).
    
    Returns:
        Dummy tensor (actual weight update happens via reward_manager).
    
    Example:
        Early phase (iter 0-5k): Small penalty (-0.01) → free exploration
        Middle phase (iter 5k-15k): Gradual increase → constrain behavior  
        Late phase (iter 15k+): Full penalty (-0.05) → refined control
    """
    # Get current training iteration
    # Note: env.common_step_counter is total steps, need to convert to iterations
    # Assuming 24 steps per iteration (default RL step count)
    current_iter = env.common_step_counter // 24
    
    # Compute interpolation factor (0.0 to 1.0)
    if current_iter <= start_iter:
        alpha = 0.0
    elif current_iter >= end_iter:
        alpha = 1.0
    else:
        alpha = (current_iter - start_iter) / (end_iter - start_iter)
    
    # Linear interpolation
    current_weight = start_value + (end_value - start_value) * alpha
    
    # Update reward weight in the environment
    if hasattr(env, 'reward_manager') and reward_name in env.reward_manager._term_names:
        # Directly modify the weight attribute of the reward term config
        if hasattr(env.reward_manager, '_term_cfgs') and reward_name in env.reward_manager._term_cfgs:
            env.reward_manager._term_cfgs[reward_name].weight = current_weight
        
        # Log every 100 iterations for debugging
        if current_iter % 100 == 0 and current_iter > 0:
            print(f"[Progressive Curriculum] Iter {current_iter}: {reward_name} = {current_weight:.6f} (alpha={alpha:.3f})")
    
    # Return dummy value (curriculum functions must return a tensor)
    return torch.tensor(current_weight, device=env.device)


def ramp_reward_param(
    env: "ManagerBasedRLEnv",
    env_ids,
    term_name: str,
    param_name: str,
    value_start: float,
    value_end: float,
    num_steps: int,
) -> torch.Tensor:
    """Linearly ramp a reward term's config parameter."""
    progress = min(env.common_step_counter / max(num_steps, 1), 1.0)
    new_value = value_start + (value_end - value_start) * progress
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.params[param_name] = new_value
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return torch.tensor(new_value, device=env.device)


def piecewise_ramp_reward_param(
    env: "ManagerBasedRLEnv",
    env_ids,
    term_name: str,
    param_name: str,
    value_start: float,
    value_mid: float,
    value_end: float,
    hold_steps: int,
    mid_steps: int,
    end_steps: int,
) -> torch.Tensor:
    """Hold, then tighten a reward term parameter in two linear ramps."""
    step = env.common_step_counter
    if step <= hold_steps:
        new_value = value_start
    elif step <= mid_steps:
        progress = (step - hold_steps) / max(mid_steps - hold_steps, 1)
        new_value = value_start + (value_mid - value_start) * progress
    elif step <= end_steps:
        progress = (step - mid_steps) / max(end_steps - mid_steps, 1)
        new_value = value_mid + (value_end - value_mid) * progress
    else:
        new_value = value_end

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.params[param_name] = new_value
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return torch.tensor(new_value, device=env.device)
