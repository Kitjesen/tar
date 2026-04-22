# Copyright (c) 2024-2026 Inovxio (穹沛科技)
"""PPO runner cfg for thunder_him_gait TAR Official variant."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ThunderGaitTarOfficialRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """TAR Official (paper-accurate TARLoco port) training config."""

    num_steps_per_env = 48
    max_iterations = 20000
    save_interval = 500
    experiment_name = "thunder_gait_tar_official_rough"
    class_name = "OnPolicyRunner"  # train_tar_official.py overrides to TAROfficialOnPolicyRunner
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic", "height_scan_group"],
    }

    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # Official TARLoco dims: actor [256,128,128], critic [512,256,256]
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[512, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,    # Official TARLoco default (not our 20)
        num_mini_batches=4,        # Official TARLoco default (not our 16)
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
