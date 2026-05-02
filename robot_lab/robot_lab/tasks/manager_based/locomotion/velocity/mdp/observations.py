# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
import re

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

# --- Helpers ---

DEFAULT_MASS_CONTACT_GROUPS = ["base", "hip", "thigh", "calf"]
DEFAULT_COLLISION_BODY_ORDER = [
    "base_link",
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def _match_body_ids(asset: Articulation, patterns: list[str]) -> list[int]:
    """Find all body indices matching a list of regex patterns."""
    all_ids = []
    for pattern in patterns:
        # find_bodies returns (indices, names)
        ids, _ = asset.find_bodies(pattern)
        all_ids.extend(ids)
    return sorted(list(set(all_ids)))

# --- Observations ---

def center_of_mass_position(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Center of mass position of the entire robot in base frame (xy only)."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get_coms() returns [num_envs, num_links, 3] usually, but check if it includes quat (7)
    link_coms_w = asset.root_physx_view.get_coms().to(env.device)
    if link_coms_w.shape[-1] == 7:
         link_coms_w = link_coms_w[..., :3]
    
    link_masses = asset.root_physx_view.get_masses().to(env.device)
    
    # Weighted sum for System CoM
    total_mass = torch.sum(link_masses, dim=1, keepdim=True)
    system_com_w = torch.sum(link_coms_w * link_masses.unsqueeze(-1), dim=1) / total_mass
    
    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w
    
    rel_pos_w = system_com_w - base_pos_w
    # Use quat_apply_inverse instead of quat_rotate_inverse
    rel_pos_b = quat_apply_inverse(base_quat_w, rel_pos_w)
    
    return rel_pos_b[:, :2]

def mass_distribution_components(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize: bool = False
) -> torch.Tensor:
    """Mass of specific body groups (base, hip, thigh, calf)."""
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses().to(env.device)
    
    # We need 4 components: base, hip, thigh, calf (summed across legs)
    base_ids = _match_body_ids(asset, ["base_link"])
    hip_ids = _match_body_ids(asset, [".*_hip"])
    thigh_ids = _match_body_ids(asset, [".*_thigh"])
    calf_ids = _match_body_ids(asset, [".*_calf"])
    
    base_mass = masses[:, base_ids].sum(dim=1, keepdim=True)
    hip_mass = masses[:, hip_ids].sum(dim=1, keepdim=True)
    thigh_mass = masses[:, thigh_ids].sum(dim=1, keepdim=True)
    calf_mass = masses[:, calf_ids].sum(dim=1, keepdim=True)
    
    dist = torch.cat([base_mass, hip_mass, thigh_mass, calf_mass], dim=-1)
    
    if normalize:
        total_mass = dist.sum(dim=-1, keepdim=True) + 1e-6
        dist = dist / total_mass
        
    return dist

def body_collision_probabilities(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    force_threshold: float = 1.0
) -> torch.Tensor:
    """
    Binary contact probability (0 or 1) for specific bodies based on force threshold.
    
    Returns contact probabilities for bodies in DEFAULT_COLLISION_BODY_ORDER:
    [base_link, FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, 
     RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
    Total: 13 dims
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Try to use ContactSensor if available
    try:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        # ContactSensor.data.net_forces_w shape: [num_envs, num_bodies_in_sensor, 3]
        # Need to map sensor body indices to asset body indices
        forces = sensor.data.net_forces_w  # [env, sensor_bodies, 3]
        force_mags = torch.norm(forces, dim=-1)  # [env, sensor_bodies]
        
        # Get body IDs from sensor config (if available) or use all sensor bodies
        if hasattr(sensor_cfg, 'body_ids') and sensor_cfg.body_ids is not None:
            if isinstance(sensor_cfg.body_ids, list):
                sensor_body_ids = sensor_cfg.body_ids
            elif isinstance(sensor_cfg.body_ids, slice):
                # slice(None, None, None) means all bodies
                sensor_body_ids = list(range(forces.shape[1]))
            else:
                # Try to convert to list if it's iterable
                try:
                    sensor_body_ids = list(sensor_cfg.body_ids)
                except TypeError:
                    # If not iterable, use all sensor bodies
                    sensor_body_ids = list(range(forces.shape[1]))
        else:
            # If sensor has body_names, try to match them
            sensor_body_ids = list(range(forces.shape[1]))
        
        # Map sensor body indices to asset body indices
        # For simplicity, assume sensor bodies match asset bodies in order
        # If sensor only has subset, we'll need to match by name
        if len(sensor_body_ids) == asset.num_bodies:
            # Sensor covers all bodies, use directly
            asset_force_mags = force_mags
        else:
            # Sensor has subset, need to map - fallback to asset view
            asset_force_mags = torch.norm(asset.root_physx_view.get_net_contact_forces(), dim=-1)
    except (KeyError, AttributeError):
        # Fallback to asset view if sensor not available
        forces = asset.root_physx_view.get_net_contact_forces()  # [env, body, 3]
        asset_force_mags = torch.norm(forces, dim=-1)  # [env, body]
    
    # Order: base, FL_hip, FL_thigh, FL_calf, ... (13 bodies usually)
    target_bodies = DEFAULT_COLLISION_BODY_ORDER
    
    probs = []
    for name in target_bodies:
        ids, _ = asset.find_bodies(name)
        if ids:
            # If multiple bodies match (unlikely for exact names), take max force
            f = asset_force_mags[:, ids].max(dim=1)[0]
            probs.append((f > force_threshold).float().unsqueeze(1))
        else:
            probs.append(torch.zeros((env.num_envs, 1), device=env.device))
            
    return torch.cat(probs, dim=-1)

def average_foot_friction(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Average friction coefficient of the ground."""
    if hasattr(env, "friction_coeffs"):
        val = env.friction_coeffs
        if val.dim() > 1:
            val = val.mean(dim=-1, keepdim=True)
        else:
            val = val.view(-1, 1)
        return val
    return torch.ones((env.num_envs, 1), device=env.device)

def generated_actuator_gains(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Return P and D gains for actuators.
    
    Returns [num_envs, num_joints * 2] tensor where each joint has [P, D] gains.
    Order matches joint order in asset.
    
    Note: In Isaac Lab, actuators are typically grouped (hip, thigh, calf, wheel),
    so joints of the same type share the same gains. This function maps actuator
    group gains to individual joints based on joint names.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Try to get from environment if stored (e.g. by randomization event)
    if hasattr(env, "actuator_gains") and env.actuator_gains is not None:
        gains = env.actuator_gains
        if isinstance(gains, torch.Tensor) and gains.shape[1] == asset.num_joints * 2:
            return gains.to(env.device)
    
    # Read from actuator model (current values in simulation)
    # Actuators are grouped by type: hip, thigh, calf, wheel
    joint_names = asset.joint_names
    
    # Map joint names to actuator groups based on naming pattern
    # Pattern: {leg}_{type}_joint (e.g., FL_hip_joint, FR_thigh_joint)
    stiffness_list = []
    damping_list = []
    
    for joint_name in joint_names:
        # Determine actuator group from joint name
        actuator_type = None
        if "_hip_joint" in joint_name:
            actuator_type = "hip"
        elif "_thigh_joint" in joint_name:
            actuator_type = "thigh"
        elif "_calf_joint" in joint_name:
            actuator_type = "calf"
        elif "_foot_joint" in joint_name or "_wheel" in joint_name:
            actuator_type = "wheel"
        
        # Get gains from actuator group
        if actuator_type and actuator_type in asset.actuators:
            actuator = asset.actuators[actuator_type]
            # Actuator stiffness/damping are typically tensors [num_envs, num_joints_in_group]
            # We need to extract one value per environment (they're the same for all joints in group)
            if hasattr(actuator, 'stiffness'):
                stiffness = actuator.stiffness
                if isinstance(stiffness, torch.Tensor):
                    # If tensor, take first joint's value (all same in group)
                    if stiffness.dim() > 1:
                        stiffness_val = stiffness[:, 0:1]  # [env, 1]
                    else:
                        stiffness_val = stiffness.unsqueeze(1)  # [env, 1]
                else:
                    # Scalar value, expand to [env, 1]
                    stiffness_val = torch.full((env.num_envs, 1), float(stiffness), device=env.device)
            else:
                stiffness_val = torch.zeros((env.num_envs, 1), device=env.device)
            
            if hasattr(actuator, 'damping'):
                damping = actuator.damping
                if isinstance(damping, torch.Tensor):
                    if damping.dim() > 1:
                        damping_val = damping[:, 0:1]  # [env, 1]
                    else:
                        damping_val = damping.unsqueeze(1)  # [env, 1]
                else:
                    damping_val = torch.full((env.num_envs, 1), float(damping), device=env.device)
            else:
                damping_val = torch.zeros((env.num_envs, 1), device=env.device)
            
            stiffness_list.append(stiffness_val)
            damping_list.append(damping_val)
        else:
            # Unknown joint type, use zeros
            stiffness_list.append(torch.zeros((env.num_envs, 1), device=env.device))
            damping_list.append(torch.zeros((env.num_envs, 1), device=env.device))
    
    # Stack: [num_envs, num_joints] for each
    if stiffness_list and damping_list:
        stiffness = torch.cat(stiffness_list, dim=1)  # [env, num_joints]
        damping = torch.cat(damping_list, dim=1)  # [env, num_joints]
        # Interleave: [P1, D1, P2, D2, ...]
        gains = torch.stack([stiffness, damping], dim=2)  # [env, num_joints, 2]
        gains = gains.view(env.num_envs, -1)  # [env, num_joints * 2]
        return gains
    else:
        # Fallback: return zeros if actuators not accessible
        return torch.zeros((env.num_envs, asset.num_joints * 2), device=env.device)

def foot_contact_force_magnitudes(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_names_regex: str = ".*_foot"
) -> torch.Tensor:
    """
    Magnitude of contact forces on feet.
    
    Returns contact force magnitudes for feet matching foot_names_regex.
    Typically returns 4 dims (one per foot: FL, FR, RL, RR).
    """
    try:
        asset: Articulation = env.scene[asset_cfg.name]
    except KeyError:
        return torch.zeros((env.num_envs, 4), device=env.device)
    
    # Find foot body IDs matching the regex
    foot_ids, _ = asset.find_bodies(foot_names_regex)
    if not foot_ids:
        return torch.zeros((env.num_envs, 4), device=env.device)
    
    # Try to use ContactSensor if available
    try:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        forces = sensor.data.net_forces_w  # [env, sensor_bodies, 3]
        
        # Map sensor body indices to asset body indices
        # Check if sensor has body_ids or body_names in config
        if hasattr(sensor_cfg, 'body_ids') and sensor_cfg.body_ids is not None:
            if isinstance(sensor_cfg.body_ids, list):
                sensor_body_ids = sensor_cfg.body_ids
            elif isinstance(sensor_cfg.body_ids, slice):
                # slice(None, None, None) means all bodies
                sensor_body_ids = list(range(min(forces.shape[1], asset.num_bodies)))
            else:
                # Try to convert to list if it's iterable
                try:
                    sensor_body_ids = list(sensor_cfg.body_ids)
                except TypeError:
                    # If not iterable, use all sensor bodies
                    sensor_body_ids = list(range(min(forces.shape[1], asset.num_bodies)))
        elif hasattr(sensor, 'cfg') and hasattr(sensor.cfg, 'body_names'):
            # Try to match by name
            sensor_body_ids = []
            for body_name in sensor.cfg.body_names:
                ids, _ = asset.find_bodies(body_name)
                if ids:
                    sensor_body_ids.extend(ids)
        else:
            # Assume sensor covers all bodies in order
            sensor_body_ids = list(range(min(forces.shape[1], asset.num_bodies)))
        
        # Extract forces for foot bodies
        foot_forces = []
        for foot_id in foot_ids:
            if foot_id in sensor_body_ids:
                sensor_idx = sensor_body_ids.index(foot_id)
                if sensor_idx < forces.shape[1]:
                    foot_forces.append(forces[:, sensor_idx, :])
                else:
                    foot_forces.append(torch.zeros((env.num_envs, 3), device=env.device))
            else:
                # Foot not in sensor, use zero
                foot_forces.append(torch.zeros((env.num_envs, 3), device=env.device))
        
        if foot_forces:
            foot_forces_tensor = torch.stack(foot_forces, dim=1)  # [env, num_feet, 3]
            return torch.norm(foot_forces_tensor, dim=-1)  # [env, num_feet]
        else:
            # Fallback to asset view
            forces = asset.root_physx_view.get_net_contact_forces()  # [env, body, 3]
            foot_forces_tensor = forces[:, foot_ids, :]  # [env, num_feet, 3]
            return torch.norm(foot_forces_tensor, dim=-1)  # [env, num_feet]
            
    except (KeyError, AttributeError, IndexError):
        # Fallback to asset view if sensor not available or mapping fails
        forces = asset.root_physx_view.get_net_contact_forces()  # [env, body, 3]
        foot_forces_tensor = forces[:, foot_ids, :]  # [env, num_feet, 3]
        return torch.norm(foot_forces_tensor, dim=-1)  # [env, num_feet]
