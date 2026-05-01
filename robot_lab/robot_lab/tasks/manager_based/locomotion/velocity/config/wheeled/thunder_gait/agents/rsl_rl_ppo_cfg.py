from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ThunderRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 200
    experiment_name = "thunder_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
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
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ThunderFlatPPORunnerCfg(ThunderRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 50000
        self.experiment_name = "thunder_flat"


@configclass
class ThunderGaitFlatPPORunnerCfg(ThunderRoughPPORunnerCfg):
    """PPO runner for gait-gated flat terrain validation experiment.

    单帧观测，网络规模与 thunder_flat 一致。
    10k iter 足够验证步态门控效果。
    """
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10000
        self.save_interval = 500
        self.experiment_name = "thunder_gait_flat"


@configclass
class ThunderCrouchPPORunnerCfg(ThunderRoughPPORunnerCfg):
    max_iterations = 15000
    experiment_name = "thunder_crouch"
    class_name = "OnPolicyRunner"


@configclass
class ThunderMultiSkillPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for Thunder multi-skill learning."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "thunder_multiskill"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
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
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ThunderRoughCurriculumPPORunnerCfg(ThunderRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 40000
        self.experiment_name = "thunder_rough_curriculum"
        self.save_interval = 200


@configclass
class ThunderStandPPORunnerCfg(ThunderRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 40000
        self.experiment_name = "thunder_stand_dynamic"


@configclass
class ThunderFlatHeightPPORunnerCfg(ThunderCrouchPPORunnerCfg):
    experiment_name = "thunder_flat_height"
    max_iterations = 20000


@configclass
class ThunderGaitRoughPPORunnerCfg(ThunderRoughPPORunnerCfg):
    """PPO runner for gait rough terrain training (Stage II).

    Asymmetric actor-critic: actor blind (policy obs only), critic gets privileged
    height_scan_group (+ base_lin_vel/foot_contact/friction/mass already in critic).
    """
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic", "height_scan_group"],
    }

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20000
        self.save_interval = 500
        self.experiment_name = "thunder_gait_rough"
