# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")




def fall_after_stood_up(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_height_threshold: float = 0.30,
    gravity_z_threshold: float = -0.7,
    stable_steps: int = 20,
    fall_height_threshold: float = 0.20,
    fall_gravity_z_threshold: float = 0.5,
) -> torch.Tensor:
    """Terminate when robot falls AFTER having stood up stably.

    Avoids false positives from initial random pose drop (robot needs ~50 steps
    to land + stabilise from z=0.55-0.75 with random roll/pitch/yaw).

    Per-env latch state is stored on env._fall_state and reset via
    `reset_fall_after_stood_up_state` (must register as a "reset" EventTerm).

    Args:
        base_height_threshold: minimum base height (m) to count as standing
        gravity_z_threshold: max projected_gravity_b[z] to count as upright
            (-1 = perfectly upright; -0.7 ≈ within ~45 deg of upright)
        stable_steps: consecutive steps satisfying both above to latch has_stood_up
        fall_height_threshold: base height (m) below which counts as fallen
        fall_gravity_z_threshold: gravity_b[z] above which counts as toppled (>0 = past horizontal)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_z = asset.data.root_pos_w[:, 2]
    grav_z = asset.data.projected_gravity_b[:, 2]

    # Lazy init / re-init on env count change
    if not hasattr(env, "_fall_state") or env._fall_state["stood_up_steps"].shape[0] != env.num_envs:
        env._fall_state = {
            "stood_up_steps": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "has_stood_up": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }

    state = env._fall_state
    is_standing = (base_z > base_height_threshold) & (grav_z < gravity_z_threshold)

    # Accumulate consecutive standing steps; reset to 0 if not standing
    state["stood_up_steps"] = torch.where(
        is_standing,
        state["stood_up_steps"] + 1,
        torch.zeros_like(state["stood_up_steps"]),
    )
    # Latch: once threshold reached, stays True until reset event clears it
    state["has_stood_up"] = state["has_stood_up"] | (state["stood_up_steps"] >= stable_steps)

    is_fallen = (base_z < fall_height_threshold) | (grav_z > fall_gravity_z_threshold)
    return state["has_stood_up"] & is_fallen
