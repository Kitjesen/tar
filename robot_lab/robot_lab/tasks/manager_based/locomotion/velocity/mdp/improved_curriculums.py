# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Improved curriculum terms for terrain adaptation.

This module provides enhanced curriculum learning strategies that account for
terrain difficulty and provide more fair assessment across different terrain types.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel_weighted(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stairs_weight: float = 1.5,
    slopes_weight: float = 1.2,
    rough_weight: float = 1.0,
    base_threshold: float = 0.5,
) -> torch.Tensor:
    """Improved terrain curriculum with difficulty-weighted thresholds.
    
    This function adjusts the upgrade/downgrade thresholds based on terrain type difficulty.
    Harder terrains (like stairs) get more lenient thresholds, while easier terrains
    (like flat) get stricter thresholds.
    
    Args:
        env: The learning environment.
        asset_cfg: The asset configuration.
        stairs_weight: Difficulty weight for stairs terrains (default 1.5 = 50% more lenient).
        slopes_weight: Difficulty weight for slopes terrains (default 1.2 = 20% more lenient).
        rough_weight: Difficulty weight for rough terrains (default 1.0 = baseline).
        base_threshold: Base threshold multiplier (default 0.5 = half of terrain size).
    
    Returns:
        The mean terrain level across all environments.
    
    Note:
        Terrain type mapping (for Thunder rough curriculum):
        - Type 0: pyramid_stairs (uses stairs_weight)
        - Type 1: pyramid_stairs_inv (uses stairs_weight)
        - Type 2: random_rough (uses rough_weight)
        - Type 3: hf_pyramid_slope (uses slopes_weight)
        - Type 4: hf_pyramid_slope_inv (uses slopes_weight)
    """
    # Build difficulty weights based on terrain types
    # This mapping should match your terrain configuration
    terrain_difficulty_weights = {
        0: stairs_weight,  # pyramid_stairs
        1: stairs_weight,  # pyramid_stairs_inv
        2: rough_weight,   # random_rough
        3: slopes_weight,  # hf_pyramid_slope
        4: slopes_weight,  # hf_pyramid_slope_inv
    }
    
    # Extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    
    # Compute the distance each robot walked
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], 
        dim=1
    )
    
    # Get terrain types for these environments
    terrain_types = terrain.terrain_types[env_ids]
    
    # Apply difficulty-weighted thresholds
    move_up = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    move_down = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    
    for i, (env_id, terrain_type) in enumerate(zip(env_ids, terrain_types)):
        # Get difficulty weight for this terrain type
        difficulty_weight = terrain_difficulty_weights.get(terrain_type.item(), 1.0)
        
        # Adjust thresholds based on difficulty
        # Harder terrain → larger threshold → easier to upgrade
        upgrade_threshold = (terrain.cfg.terrain_generator.size[0] / 2) / difficulty_weight
        
        # Expected distance based on command velocity
        cmd_norm = torch.norm(command[env_id, :2])
        expected_distance = cmd_norm * env.max_episode_length_s * base_threshold
        # Harder terrain → expected distance reduced
        downgrade_threshold = expected_distance / difficulty_weight
        
        # Check upgrade condition
        if distance[i] > upgrade_threshold:
            move_up[i] = True
        # Check downgrade condition (only if not upgrading)
        elif distance[i] < downgrade_threshold:
            move_down[i] = True
    
    # Update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    # 🔧 修正：限制在有效范围内 [0, max_terrain_level-1] 后再返回
    max_level = terrain.max_terrain_level - 1
    clamped_levels = torch.clamp(terrain.terrain_levels, 0, max_level)
    return torch.mean(clamped_levels.float())


def terrain_levels_vel_success_rate(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    success_distance_ratio: float = 0.4,
    failure_distance_ratio: float = 0.2,
) -> torch.Tensor:
    """Terrain curriculum based on success rate rather than absolute distance.
    
    Instead of fixed distance thresholds, this method considers:
    - Success: Walked > X% of commanded distance
    - Failure: Walked < Y% of commanded distance
    
    This naturally adapts to different terrain difficulties because the
    commanded velocity already takes terrain into account.
    
    Args:
        env: The learning environment.
        asset_cfg: The asset configuration.
        success_distance_ratio: Ratio of commanded distance to consider success (default 0.4 = 40%).
        failure_distance_ratio: Ratio of commanded distance to consider failure (default 0.2 = 20%).
    
    Returns:
        The mean terrain level across all environments.
    """
    # Extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    
    # Compute actual distance walked
    actual_distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], 
        dim=1
    )
    
    # Compute expected distance based on command
    cmd_norm = torch.norm(command[env_ids, :2], dim=1)
    expected_distance = cmd_norm * env.max_episode_length_s
    
    # Avoid division by zero
    expected_distance = torch.clamp(expected_distance, min=0.1)
    
    # Compute success rate
    distance_ratio = actual_distance / expected_distance
    
    # Determine upgrade/downgrade
    move_up = distance_ratio > success_distance_ratio
    move_down = distance_ratio < failure_distance_ratio
    move_down = move_down & (~move_up)  # Don't downgrade if already upgrading
    
    # Update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    # 🔧 修正：限制在有效范围内 [0, max_terrain_level-1] 后再返回
    max_level = terrain.max_terrain_level - 1
    clamped_levels = torch.clamp(terrain.terrain_levels, 0, max_level)
    return torch.mean(clamped_levels.float())


def terrain_levels_vel_stability(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_weight: float = 0.6,
    stability_weight: float = 0.4,
    distance_threshold: float = 0.4,
    stability_threshold: float = 0.7,
) -> torch.Tensor:
    """Terrain curriculum considering both distance and stability.
    
    This method combines:
    - Distance metric: How far the robot walked
    - Stability metric: How stable the robot was (low torques, smooth motion)
    
    Args:
        env: The learning environment.
        asset_cfg: The asset configuration.
        distance_weight: Weight for distance in the combined score.
        stability_weight: Weight for stability in the combined score.
        distance_threshold: Distance ratio threshold for success.
        stability_threshold: Stability score threshold for success.
    
    Returns:
        The mean terrain level across all environments.
    """
    # Extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    
    # 1. Distance metric
    actual_distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], 
        dim=1
    )
    cmd_norm = torch.norm(command[env_ids, :2], dim=1)
    expected_distance = torch.clamp(cmd_norm * env.max_episode_length_s, min=0.1)
    distance_score = torch.clamp(actual_distance / expected_distance, 0, 1)
    
    # 2. Stability metric (simple version: based on episode length)
    # Longer episodes without termination = more stable
    episode_length = env.episode_length_buf[env_ids].float()
    stability_score = episode_length / env.max_episode_length
    
    # 3. Combined score
    combined_score = distance_weight * distance_score + stability_weight * stability_score
    
    # Determine upgrade/downgrade based on combined score
    upgrade_threshold = distance_weight * distance_threshold + stability_weight * stability_threshold
    downgrade_threshold = upgrade_threshold * 0.5  # More lenient downgrade
    
    move_up = combined_score > upgrade_threshold
    move_down = combined_score < downgrade_threshold
    move_down = move_down & (~move_up)
    
    # Update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    # 🔧 修正：限制在有效范围内 [0, max_terrain_level-1] 后再返回
    max_level = terrain.max_terrain_level - 1
    clamped_levels = torch.clamp(terrain.terrain_levels, 0, max_level)
    return torch.mean(clamped_levels.float())


def terrain_levels_separate_tracking(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track terrain levels separately for each terrain type.
    
    This method maintains independent terrain levels for each terrain type,
    preventing the "stairs are always hard" problem from affecting other terrains.
    
    Note: This requires custom terrain level management and is more complex to implement.
    
    Returns:
        The mean terrain level across all environments.
    """
    # This would require modifying the terrain importer to track
    # separate levels per terrain type
    # For now, returning standard behavior
    from isaaclab.envs.mdp import terrain_levels_vel
    return terrain_levels_vel(env, env_ids, asset_cfg)

