# Thunder gait task registrations for methods that are present in this repository.
#
# Tasks:
#   GaitHimRough:  HIM baseline (SwAV contrastive on proprio history; Long et al. 2024)
#   GaitTarRough:  TAR MLP variant adapted from TARLoco (Mousa et al. 2025)
#   GaitTerRough:  TerAdapt (VQ-VAE codebook alignment; 2026 RA-L)

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
    id="GaitTerRough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teradapt_rough_env_cfg:ThunderGaitTerAdaptRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_teradapt_cfg:ThunderGaitTerAdaptRoughPPORunnerCfg",
    },
)
