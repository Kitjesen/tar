# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class HeadingBasedVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that uses heading-based control instead of direct angular velocity command.
    
    This avoids the robot needing to twist its body to track angular velocity commands.
    Instead, it tracks a target heading and computes angular velocity from heading error.
    """

    cfg: mdp.HeadingBasedVelocityCommandCfg
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands with heading-based control.
        
        Only samples linear velocities and heading target, NOT angular velocity.
        Angular velocity will be computed from heading error in _update_command.
        """
        r = torch.empty(len(env_ids), device=self.device)
        
        # Sample linear velocities
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        
        # DO NOT sample angular velocity - it will be computed from heading error
        # Initialize to zero, will be updated in _update_command
        self.vel_command_b[env_ids, 2] = 0.0
        
        # Sample heading target (this is what we track)
        self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
        
        # All environments use heading control (no need for is_heading_env flag)
        # But we still need to set it for compatibility
        self.is_heading_env[env_ids] = True
        
        # Threshold processing: set small linear velocities to zero
        vel_xy_norm = torch.norm(self.vel_command_b[env_ids, :2], dim=1)
        mask = (vel_xy_norm > self.cfg.deadzone_threshold).unsqueeze(1)
        self.vel_command_b[env_ids, :2] *= mask
        
        # Update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Update command: compute angular velocity from heading error.
        
        This is the key difference: angular velocity is ALWAYS computed from
        heading error, not sampled directly. This prevents the robot from
        needing to twist its body to track arbitrary angular velocity commands.
        """
        import isaaclab.utils.math as math_utils
        
        # Compute angular velocity from heading error for ALL environments
        # (since we always use heading control)
        heading_error = math_utils.wrap_to_pi(self.heading_target - self.robot.data.heading_w)
        
        # Compute angular velocity using proportional control
        ang_vel = self.cfg.heading_control_stiffness * heading_error
        
        # Clip to valid range
        self.vel_command_b[:, 2] = torch.clip(
            ang_vel,
            min=self.cfg.ranges.ang_vel_z[0],
            max=self.cfg.ranges.ang_vel_z[1],
        )
        
        # Enforce standing (zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            self.vel_command_b[standing_env_ids, :] = 0.0


@configclass
class HeadingBasedVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for heading-based velocity command generator.
    
    This command generator uses heading tracking instead of direct angular velocity commands.
    The angular velocity is computed from heading error, which is more natural for locomotion.
    """

    class_type: type = HeadingBasedVelocityCommand
    
    # Override parent defaults to ensure heading control is always enabled
    heading_command: bool = True
    """Always True for heading-based command. Angular velocity is computed from heading error."""
    
    rel_heading_envs: float = 1.0
    """Always 1.0 for heading-based command. All environments use heading control."""
    deadzone_threshold: float = 0.1
    """xy 指令 norm < deadzone_threshold 时清零，防低速蠕动和真机手柄噪声。"""


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


class UniformBodyHeightCommand(CommandTerm):
    """Command generator for body height from uniform distribution."""

    cfg: UniformBodyHeightCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformBodyHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Create buffer to store the height command (num_envs, 1)
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)

    def __str__(self) -> str:
        msg = "UniformBodyHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.height_command.shape)}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeight range: {self.cfg.ranges.height}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired body height command. Shape is (num_envs, 1)."""
        return self.height_command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # Sample random height from uniform distribution
        height_min, height_max = self.cfg.ranges.height
        self.height_command[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (
            height_max - height_min
        ) + height_min

    def _update_command(self):
        pass


@configclass
class UniformBodyHeightCommandCfg(CommandTermCfg):
    """Configuration for uniform body height command generator."""

    class_type: type = UniformBodyHeightCommand

    @configclass
    class Ranges:
        """Uniform distribution ranges for body height command."""

        height: tuple[float, float] = (0.3, 0.7)  # Min and max body height in meters

    ranges: Ranges = Ranges()


class StandingPostureCommand(CommandTerm):
    """
    Command generator for standing posture with one-hot encoding.

    Generates 5-dimensional one-hot encoded posture commands:
    - [1,0,0,0,0]: Normal standing (upright)
    - [0,1,0,0,0]: Handstand (inverted)
    - [0,0,1,0,0]: Left side standing
    - [0,0,0,1,0]: Right side standing
    - [0,0,0,0,1]: Front two legs standing
    """

    cfg: StandingPostureCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: StandingPostureCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Create buffer to store the one-hot posture command (num_envs, 5)
        self.posture_command = torch.zeros(self.num_envs, 5, device=self.device)
        # Internal buffer for posture indices
        self._posture_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def __str__(self) -> str:
        msg = "StandingPostureCommand (one-hot encoded):\n"
        msg += f"\tCommand dimension: {tuple(self.posture_command.shape)}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += "\tPosture options:\n"
        msg += "\t  [1,0,0,0,0] = Normal standing\n"
        msg += "\t  [0,1,0,0,0] = Handstand\n"
        msg += "\t  [0,0,1,0,0] = Left side standing\n"
        msg += "\t  [0,0,0,1,0] = Right side standing\n"
        msg += "\t  [0,0,0,0,1] = Front two legs standing"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired posture command (one-hot encoded). Shape is (num_envs, 5)."""
        return self.posture_command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # Sample random posture indices from [0, 1, 2, 3, 4]
        self._posture_indices[env_ids] = torch.randint(
            0, 5, (len(env_ids),), device=self.device, dtype=torch.long
        )
        # Convert to one-hot encoding
        self.posture_command[env_ids] = 0.0  # Clear previous values
        self.posture_command[env_ids, self._posture_indices[env_ids]] = 1.0

    def _update_command(self):
        pass


@configclass
class StandingPostureCommandCfg(CommandTermCfg):
    """Configuration for standing posture command generator."""

    class_type: type = StandingPostureCommand

    resampling_time_range: tuple[float, float] = (10.0, 10.0)  # Time between posture changes
