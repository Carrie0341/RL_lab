# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Factory insertion environment.
"""

import gymnasium as gym

from . import agents
from .factory_camera_env import FactoryCameraEnv
from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryEnvCfg, FactoryTaskPegInsertCfg, FactoryRGBDCameraEnvCfg
from .factory_tasks_cfg import PegInsert

##
# Register Gym environments.
##

# gym.register(
#     id="Custom-Factory-PegInsert-Direct-v0",
#     entry_point=f"{__name__}.factory_env:FactoryEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskPegInsertCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Custom-Factory-PegInsert-RGB-Camera-Direct-v0",
#     entry_point=f"{__name__}.factory_camera_env:FactoryCameraEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.factory_camera_env:FactoryRGBCameraEnvCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Custom-Factory-PegInsert-Depth-Camera-Direct-v0",
#     entry_point=f"{__name__}.factory_camera_env:FactoryCameraEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.factory_camera_env:FactoryDepthCameraEnvCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
#     },
# )

# Register RGBD camera-based environment
gym.register(
    id="Custom-Factory-PegInsert-RGBD-Camera-Direct-v0",
    entry_point=f"{__name__}.factory_camera_env:FactoryCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryRGBDCameraEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rgbd_camera_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rgbd_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rgbd_camera_ppo_cfg.yaml",
    },
)
