# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.sensors import TiledCamera

from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryRGBCameraEnvCfg, FactoryDepthCameraEnvCfg


class FactoryCameraEnv(FactoryEnv):
    """Factory environment with camera observations."""

    cfg: FactoryRGBCameraEnvCfg  # 可以是 RGB 或 Depth 相機配置

    def __init__(self, cfg: FactoryRGBCameraEnvCfg, render_mode: str | None = None, **kwargs):
        # 不需要更新 observation_space，因為它已經在配置中設置為 [128, 128, 3] 或 [128, 128, 1]
        self.cfg_task = cfg.task
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        """Initialize simulation scene with camera."""
        # 調用父類方法設置基本場景
        super()._setup_scene()

        # 添加相機
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["camera"] = self._tiled_camera

    def _get_observations(self):
        """Get observations from camera for policy and state vectors for critic."""

        print("[!!!Debug] CameraEnv._get_observations()")
        # 獲取相機數據
        data_type = self.cfg.tiled_camera.data_types[0]  # 使用配置中的第一個數據類型

        if data_type == "rgb":
            # 獲取RGB圖像並歸一化到[0,1]範圍
            camera_data = self._tiled_camera.data.output[data_type] / 255.0
            # 標準化處理
            mean = torch.mean(camera_data, dim=(1, 2), keepdim=True)
            std = torch.std(camera_data, dim=(1, 2), keepdim=True) + 1e-6  # 避免除以零
            camera_data = (camera_data - mean) / std
        elif data_type == "depth":
            # 獲取深度圖像
            camera_data = self._tiled_camera.data.output[data_type]
            # 處理無限深度值
            camera_data[camera_data == float("inf")] = 0.0
            # 歸一化深度值到 [0, 1] 範圍
            max_depth = self.cfg.tiled_camera.spawn.clipping_range[1]
            camera_data = camera_data / max_depth
            # 添加通道維度
            camera_data = camera_data.unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported camera data type: {data_type}")

        # 可選：保存圖像到文件進行調試
        if self.cfg.write_image_to_file and self.episode_length_buf[0] % 100 == 0:
            import os
            import numpy as np
            from PIL import Image

            os.makedirs("debug_images", exist_ok=True)
            img_data = camera_data[0].detach().cpu().numpy()

            if data_type == "rgb":
                # 將標準化的RGB圖像轉換回[0,255]範圍
                img_data = (img_data * std[0].cpu().numpy() + mean[0].cpu().numpy())
                img_data = np.clip(img_data * 255, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_data)
            else:  # depth
                # 將深度圖像轉換為灰度圖
                img_data = (img_data * 255).astype(np.uint8).squeeze()
                img = Image.fromarray(img_data, mode="L")

            img.save(f"debug_images/frame_{self.episode_length_buf[0]}.png")

        # 獲取critic的狀態向量（與原始FactoryEnv相同）
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }
        state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order + ["prev_actions"]]
        state_tensors = torch.cat(state_tensors, dim=-1)

        print("[Debug] Camera data shape:", camera_data.shape)
        # 返回相機數據給policy，狀態向量給critic
        return {"policy": camera_data, "critic": state_tensors}
