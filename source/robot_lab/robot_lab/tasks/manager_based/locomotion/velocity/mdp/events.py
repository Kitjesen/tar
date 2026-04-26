# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization, or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def reset_joints_with_type_specific_ranges(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_range_hip: tuple[float, float] = (-0.4, 0.4),
    position_range_thigh: tuple[float, float] = (-3.14, 3.14),
    position_range_calf: tuple[float, float] = (-3.14, 3.14),
    position_range_wheel: tuple[float, float] = (-3.14, 3.14),
    velocity_range: tuple[float, float] = (-4.0, 4.0),
) -> None:
    """Reset joint positions and velocities with type-specific ranges.
    
    This function randomizes joint positions and velocities based on joint type (hip, thigh, calf, wheel).
    Different joint types have different motion ranges, so this allows for more realistic fall recovery
    training scenarios.
    
    Args:
        env: The RL environment instance.
        env_ids: Environment IDs to reset. If None, resets all environments.
        asset_cfg: Configuration for the robot asset.
        position_range_hip: Position offset range for hip joints (rad). Default: (-0.4, 0.4).
            Based on Thunder URDF limit: ±0.4 rad (≈±23°), maximum possible.
        position_range_thigh: Position offset range for thigh joints (rad). Default: (-3.14, 3.14).
            Full ±π rad range (≈±180°) for completely random fall poses. Simulator will clamp to URDF limits.
        position_range_calf: Position offset range for calf joints (rad). Default: (-3.14, 3.14).
            Full ±π rad range (≈±180°) for completely random fall poses. Simulator will clamp to URDF limits.
        position_range_wheel: Position offset range for wheel/foot joints (rad). Default: (-3.14, 3.14).
            Full ±π rad range (≈±180°). Foot joints are continuous (unlimited), full range for extreme poses.
        velocity_range: Velocity range for all joints (rad/s). Default: (-4.0, 4.0).
            Large range to simulate fall impact and dynamic recovery scenarios.
        velocity_range: Velocity range for all joints (rad/s). Default: (-3.0, 3.0).
    
    Example:
        .. code-block:: python
        
            # In environment configuration
            self.events.randomize_reset_joints.func = reset_joints_with_type_specific_ranges
            self.events.randomize_reset_joints.params = {
                "asset_cfg": SceneEntityCfg("robot"),
                "position_range_hip": (-0.4, 0.4),      # URDF limit: ±0.4 rad (max)
                "position_range_thigh": (-3.14, 3.14),  # Full ±π rad range for random poses
                "position_range_calf": (-3.14, 3.14),   # Full ±π rad range for random poses
                "position_range_wheel": (-3.14, 3.14),  # Full ±π rad range (continuous joint)
                "velocity_range": (-4.0, 4.0),          # Large range for fall impact
            }
    """
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Resolve environment IDs
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(env.device)
    
    # Get default joint positions and velocities
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    
    # Get joint names from asset
    # Try different ways to get joint names depending on Isaac Lab version
    try:
        joint_names = asset.root_physx_view.joint_names
    except AttributeError:
        # Fallback: use joint indices and infer names from asset configuration
        num_joints = asset.num_joints
        joint_names = [f"joint_{i}" for i in range(num_joints)]
        # Try to get actual names if available
        if hasattr(asset, "joint_names"):
            joint_names = asset.joint_names
        elif hasattr(asset.data, "joint_names"):
            joint_names = asset.data.joint_names
    
    # Apply type-specific randomization
    for i, joint_name in enumerate(joint_names):
        # Determine joint type and apply appropriate range
        joint_name_lower = joint_name.lower()
        if "hip" in joint_name_lower:
            pos_min, pos_max = position_range_hip
        elif "thigh" in joint_name_lower:
            pos_min, pos_max = position_range_thigh
        elif "calf" in joint_name_lower:
            pos_min, pos_max = position_range_calf
        elif "foot" in joint_name_lower or "wheel" in joint_name_lower:
            pos_min, pos_max = position_range_wheel
        else:
            # Default range for unknown joints (use thigh range as middle ground)
            pos_min, pos_max = position_range_thigh
        
        # Randomize joint positions (offset from default, within specified range)
        # pos_min and pos_max define the offset range, e.g., (-1.2, 1.2) means ±1.2 rad offset
        joint_offsets = torch.rand(len(env_ids), device=env.device) * (pos_max - pos_min) + pos_min
        joint_pos[:, i] += joint_offsets
        
        # Randomize joint velocities (same range for all joints)
        vel_min, vel_max = velocity_range
        joint_vel[:, i] = torch.rand(len(env_ids), device=env.device) * (vel_max - vel_min) + vel_min
    
    # Write joint state to simulator
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)



def reset_fall_after_stood_up_state(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
):
    """Reset per-env latch used by `terminations.fall_after_stood_up`.

    Must be registered as an EventTerm with mode="reset" so that
    after each episode reset, the robot starts again with has_stood_up=False
    (otherwise initial random-pose drop would immediately trigger termination).
    """
    if not hasattr(env, "_fall_state"):
        return
    env._fall_state["stood_up_steps"][env_ids] = 0
    env._fall_state["has_stood_up"][env_ids] = False
