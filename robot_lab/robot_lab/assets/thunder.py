import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

THUNDER_V3_MASS_INERTIA_URDF = (
    f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/thunder_v3_assets/urdf/"
    "thunder_v3_cad_inertia.urdf"
)

OW_WHEELED_LEG_DOG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/ow/wheeled_leg_dogv2/urdf/wheeled_leg_dogv2.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*_hip_joint": 0.0,
            "FL_thigh_joint": 0.0,
            "FR_thigh_joint": 0.0,
            "RL_thigh_joint": 0.0,
            "RR_thigh_joint": 0.0,
            "FL_calf_joint": -1.2,
            "FR_calf_joint": 1.2,
            "RL_calf_joint": -1.2,
            "RR_calf_joint": 1.2,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_joint",
                ".*_thigh_joint",
                ".*_calf_joint",
            ],
            effort_limit=120.0,
            stiffness=200.0,
            damping=5,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=16.965,
            stiffness=0.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)

OW_WHEEL_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/ow/ow_wheel_description/urdf/ow_wheel.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.7,
            "FL_calf_joint": -1.5,
            "FL_foot_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.7,
            "FR_calf_joint": 1.5,
            "FR_foot_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -0.7,
            "RL_calf_joint": 1.5,
            "RL_foot_joint": 0.0,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.7,
            "RR_calf_joint": -1.5,
            "RR_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_joint",
                ".*_thigh_joint",
                ".*_calf_joint",
            ],
            effort_limit=120.0,
            stiffness=160.0,
            damping=5,
            friction=0.0,
            min_delay=0,
            max_delay=8,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=16.965,
            stiffness=0.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)

THUNDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=THUNDER_V3_MASS_INERTIA_URDF,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.8,
            "FR_calf_joint": 1.7,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.7,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.7,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -0.8,
            "RL_calf_joint": 1.7,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=100.0,
            damping=5.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=100.0,
            damping=5.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=100.0,
            damping=5.0,
            friction=0.0,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=16.956,
            stiffness=0.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)


THUNDER_CFG_V1 = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=THUNDER_V3_MASS_INERTIA_URDF,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.8,
            "FR_calf_joint": 1.7,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.7,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.7,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -0.8,
            "RL_calf_joint": 1.7,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=100.0,
            damping=5.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=100.0,
            damping=5.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=100.0,
            damping=5.0,
            friction=0.0,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=16.956,
            stiffness=0.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)

THUNDER_NOHEAD_POSE2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=THUNDER_V3_MASS_INERTIA_URDF,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        # joint_pos={
        #     "FR_hip_joint": 0.0,
        #     "FR_thigh_joint": -0.64,
        #     "FR_calf_joint": 1.6,
        #     "FL_hip_joint": 0.0,
        #     "FL_thigh_joint": 0.64,
        #     "FL_calf_joint": -1.6,
        #     "RR_hip_joint": 0.0,
        #     "RR_thigh_joint": 0.64,
        #     "RR_calf_joint": -1.6,
        #     "RL_hip_joint": 0.0,
        #     "RL_thigh_joint": -0.64,
        #     "RL_calf_joint": 1.6,
        #     ".*_foot_joint": 0.0,
        # },
        joint_pos={
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": -0.9,
            "FR_calf_joint": 1.8,
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": 0.1,
            "RR_thigh_joint": 2.2,
            "RR_calf_joint": 1.8,
            "RL_hip_joint": -0.1,
            "RL_thigh_joint": -2.2,
            "RL_calf_joint": -1.8,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=100.0,
            damping=5.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=100.0,
            damping=5.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=100.0,
            damping=5.0,
            friction=0.0,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=16.956,
            stiffness=0.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)
