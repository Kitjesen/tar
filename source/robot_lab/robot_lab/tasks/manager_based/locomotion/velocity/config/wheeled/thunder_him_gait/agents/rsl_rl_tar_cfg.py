# PPO runner cfg for thunder_him_gait TAR variant.
# Matches ammousa/TARLoco `Go1RoughPpoTarRunnerCfg` + `tar_algo_cfg` exactly.
# Reference: reference/TARLoco/exts/tarloco/tasks/agents/rsl_rl_cfg.py lines 109-146.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ThunderGaitTarRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Paper-accurate TAR training config — mirrors TARLoco Go1RoughPpoTarRunnerCfg."""

    num_steps_per_env = 24                        # TARLoco Go1RoughPpoRunnerCfg
    max_iterations = 1500                         # TARLoco official Go1 default
    save_interval = 100                           # TARLoco default
    experiment_name = "thunder_gait_tar_rough"
    empirical_normalization = True                # TARLoco uses obs whitening
    class_name = "OnPolicyRunner"                 # train_tar.py swaps in TAROnPolicyRunner
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic", "height_scan_group"],
    }

    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # Actor/Critic dims: [512, 256, 128] (TARLoco Go1RoughPpoTarRunnerCfg policy override)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # TARLoco `tar_algo_cfg`
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,                        # TARLoco actual: 0.01 (not 0.0)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,                     # lr_max; lr_min=5e-5 not exposed in rsl_rl
        schedule="adaptive",
        gamma=0.99,                               # TARLoco actual: 0.99 (not 0.998)
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
