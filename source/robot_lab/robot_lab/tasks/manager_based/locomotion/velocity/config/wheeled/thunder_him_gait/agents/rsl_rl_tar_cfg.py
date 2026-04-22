# PPO runner cfg for thunder_him_gait TAR variant (paper-accurate TARLoco port).

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ThunderGaitTarRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Paper-accurate TAR training config.

    Matches ammousa/TARLoco settings:
      - actor [256, 128, 128], critic [512, 256, 256]
      - num_learning_epochs=5, num_mini_batches=4
      - max_iterations=7500 (paper: TAR converges ~7500 iter vs HIM 12500-17500)
      - single-stage end-to-end PPO (no teacher pretraining)
    """

    num_steps_per_env = 48
    max_iterations = 7500
    save_interval = 500
    experiment_name = "thunder_gait_tar_rough"
    class_name = "OnPolicyRunner"  # train_tar.py overrides to TAROnPolicyRunner
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic", "height_scan_group"],
    }

    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # TARLoco dims: actor [256,128,128], critic [512,256,256]
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[512, 256, 256],
        activation="elu",
    )

    # PPO hyperparameters from TARLoco defaults
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,        # TARLoco default
        num_learning_epochs=5,    # TARLoco default
        num_mini_batches=4,       # TARLoco default
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.998,              # TARLoco default (not 0.99)
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
