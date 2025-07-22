# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaacsim.core.utils.torch as torch_utils
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.envs import ViewerCfg
from isaaclab.utils.math import quat_from_euler_xyz

from .factory_tasks_cfg import ASSET_DIR, FactoryTask, PegInsert
import torch
import math

CAMERA_WIDTH = 128
CAMERA_HEIGHT = 128

# 定義多個相機視角
#  1.(x正後 y正? z正上) 2.(x正左 y正後 z正上)
CAMERA_POS = [(.9, 0, .2), (0.45, .7, .4)]

roll = [torch.tensor(0), torch.tensor(0)] 

# 1.(值越大角度越低 => 數字小角度低) 2.(值越大角度越高 => 數字小角度高)
pitch = [torch.tensor(2*math.pi / 12), torch.tensor(2*math.pi / 10)]

yaw = [torch.tensor(math.pi), torch.tensor(math.pi / 2)]
camera_quat = [quat_from_euler_xyz(roll[i], pitch[i], -yaw[i]) for i in range(len(CAMERA_POS))]
CAMERA_ROT = [tuple(camera_quat[i].tolist()) for i in range(len(CAMERA_POS))]  # 轉換為元組格式 (w, x, y, z)


CAMERA_EYE = (1.0, 1.0, 1.0)  # Viewer eye position
num_envs = 64

# 每個相機的RGBD數據大小
CAMERA_CHANNELS = 4  # RGBD
SINGLE_CAMERA_DATA_SIZE = CAMERA_HEIGHT * CAMERA_WIDTH * CAMERA_CHANNELS
# 多相機的總數據大小 + 動作維度
MULTI_CAMERA_DATA_SIZE = SINGLE_CAMERA_DATA_SIZE * len(CAMERA_POS)
ACTION_DIM = 6
TOTAL_OBS_SIZE = MULTI_CAMERA_DATA_SIZE + ACTION_DIM

OBS_CAMERA_CFG = {
    'rgbd': TOTAL_OBS_SIZE
}
OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]

    reset_joints = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null = 10.0
    kd_null = 6.3246


@configclass
class FactoryEnvCfg(DirectRLEnvCfg):
    decimation = 8
    action_space = 6
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 21
    state_space = 72
    obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]

    task_name: str = "peg_insert"  # peg_insert, gear_mesh, nut_thread
    task: FactoryTask = FactoryTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()

    episode_length_s = 10.0  # Probably need to override.
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=87,
                velocity_limit=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=12,
                velocity_limit=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit=40.0,
                velocity_limit=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )


@configclass
class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 10.0


@configclass
class FactoryRGBCameraEnvCfg(FactoryEnvCfg):
    """Configuration for Factory environment with RGB camera."""
    task_name = "peg_insert_rgb_camera"
    # camera
    tiled_camera_front: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=CAMERA_POS[0], rot=CAMERA_ROT[0], convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0)
        ),
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )
    write_image_to_file = False

    # spaces - 與 CartpoleRGBCameraEnvCfg 保持一致的格式
    observation_space = [CAMERA_HEIGHT, CAMERA_WIDTH, 3]  # [height, width, channels]

    # change viewer settings
    viewer = ViewerCfg(eye=CAMERA_EYE)

    # reduce number of environments for camera-based training
    scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.0)

    # Use PegInsert task configuration by default
    task = PegInsert()


@configclass
class FactoryDepthCameraEnvCfg(FactoryRGBCameraEnvCfg):
    """Configuration for Factory environment with depth camera."""
    task_name = "peg_insert_depth_camera"
    # camera
    tiled_camera_front: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=CAMERA_POS[0], rot=CAMERA_ROT[0], convention="world"),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0)
        ),
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )

    # spaces
    observation_space = [CAMERA_HEIGHT, CAMERA_WIDTH, 1]  # [height, width, channels]


@configclass
class FactoryRGBDCameraEnvCfg(FactoryEnvCfg):
    """Configuration for Factory environment with multiple RGBD cameras."""
    task_name = "peg_insert_multi_rgbd_camera"

    # 定義多個相機
    tiled_camera_1: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera1",
        offset=TiledCameraCfg.OffsetCfg(pos=CAMERA_POS[0], rot=CAMERA_ROT[0], convention="world"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0)
        ),
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )

    tiled_camera_2: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera2",
        offset=TiledCameraCfg.OffsetCfg(pos=CAMERA_POS[1], rot=CAMERA_ROT[1], convention="world"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0)
        ),
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )

    write_image_to_file = False

    # 觀測空間為所有相機數據的總和 + 動作維度
    observation_space = TOTAL_OBS_SIZE

    # change viewer settings
    viewer = ViewerCfg(eye=CAMERA_EYE)

    # reduce number of environments for camera-based training
    scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.0)

    # Use PegInsert task configuration by default
    task = PegInsert()
