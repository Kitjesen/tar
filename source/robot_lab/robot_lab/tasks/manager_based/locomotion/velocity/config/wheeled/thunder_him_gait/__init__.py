# Copyright (c) 2024-2026 Inovxio (穹沛科技)
# SPDX-License-Identifier: Apache-2.0
"""Thunder Gait HIM family — rough-terrain RL tasks with different architectures.

Tasks registered:
  - GaitHimRough:         baseline HIM (SwAV), uses existing HIMActorCritic + train_him.py
  - GaitTarRough:         simplified TAR variant (triplet loss)
  - GaitTarOfficialRough: paper-accurate TARLoco port (contrastive hinge + encoder_critic)
  - GaitTerRough:         TerAdapt (VQ-VAE codebook alignment)
"""

import gymnasium as gym

from . import agents


gym.register(
    id="GaitHimRough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.him_rough_env_cfg:ThunderGaitHimRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ThunderGaitHimRoughPPORunnerCfg",
    },
)


gym.register(
    id="GaitTarRough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tar_rough_env_cfg:ThunderGaitTarRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_tar_cfg:ThunderGaitTarRoughPPORunnerCfg",
    },
)


gym.register(
    id="GaitTarOfficialRough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tar_official_rough_env_cfg:ThunderGaitTarOfficialRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_tar_official_cfg:ThunderGaitTarOfficialRoughPPORunnerCfg",
    },
)


gym.register(
    id="GaitTerRough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teradapt_rough_env_cfg:ThunderGaitTerAdaptRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_teradapt_cfg:ThunderGaitTerAdaptRoughPPORunnerCfg",
    },
)
