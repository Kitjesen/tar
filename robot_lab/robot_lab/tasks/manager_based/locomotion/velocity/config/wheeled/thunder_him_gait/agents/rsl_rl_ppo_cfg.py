# Copyright (c) 2024-2026 Inovxio (穹沛科技)
# SPDX-License-Identifier: Apache-2.0
"""PPO runner cfg for thunder_him_gait (used by train_him.py).

train_him.py hardcodes HIMActorCritic/HIMPPO/HIMOnPolicyRunner regardless of
class_name here; this cfg is only used for hyperparameters + obs_groups.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ThunderGaitHimRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """HIM training config — mirrors thunder_hist_rough but for gait-gated rewards."""

    num_steps_per_env = 48
    max_iterations = 20000
    save_interval = 200
    experiment_name = "thunder_gait_him_rough"
    class_name = "OnPolicyRunner"
    obs_groups = {"policy": ["policy"], "critic": ["critic", "height_scan_group"]}

    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=20,
        num_mini_batches=16,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
