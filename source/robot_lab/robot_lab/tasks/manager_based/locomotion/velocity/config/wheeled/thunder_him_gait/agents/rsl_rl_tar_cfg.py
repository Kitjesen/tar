# PPO runner cfg for the current Thunder TAR MLP variant.
# Hyperparameters follow the TARLoco Go1 TAR MLP runner where applicable,
# but this repository does not currently include the recurrent TAR student.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ThunderGaitTarRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """TAR MLP training config for Thunder."""

    num_steps_per_env = 24                        # TARLoco Go1 TAR MLP runner
    max_iterations = 1500                         # TARLoco Go1 default used in this repo
    save_interval = 100                           # TARLoco default
    experiment_name = "thunder_gait_tar_rough"
    empirical_normalization = True                # TARLoco uses observation whitening
    class_name = "OnPolicyRunner"                 # train_tar.py swaps in TAROnPolicyRunner
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic", "height_scan_group"],
    }

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
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,                     # tar_ppo.py clamps adaptive LR to [5e-5, 1e-3]
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
