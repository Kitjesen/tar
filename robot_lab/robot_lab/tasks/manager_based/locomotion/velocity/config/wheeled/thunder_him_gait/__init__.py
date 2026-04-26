# Thunder gait task registrations for methods that are present in this repository.
#
# Tasks:
#   GaitTarRough: current TAR MLP variant adapted from TARLoco
#   GaitTerRough: TerAdapt (VQ-VAE codebook alignment)
#
# Notes:
#   The HIM baseline is not registered here because the corresponding
#   train_him/env_cfg/agent_cfg files are not part of this repository snapshot.

import gymnasium as gym

from . import agents


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
