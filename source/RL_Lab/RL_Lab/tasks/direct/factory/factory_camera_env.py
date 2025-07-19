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
        self.is_rgbd_task = cfg.task_name == "peg_insert_rgbd_camera"

    def _setup_scene(self):
        """Initialize simulation scene with camera."""
        # 調用父類方法設置基本場景
        super()._setup_scene()

        # 添加相機
        self._tiled_camera_front = TiledCamera(self.cfg.tiled_camera_front)
        self.scene.sensors["camera"] = self._tiled_camera_front

    def _get_observations(self):
        """Get observations from camera for policy and state vectors for critic."""
        # 獲取相機數據
        data_type = self.cfg.tiled_camera_front.data_types[0]  # 使用配置中的第一個數據類型

        if self.is_rgbd_task:
            # 分別處理 RGB 和 Depth 圖像
            rgb_data = self._tiled_camera_front.data.output["rgb"] / 255.0
            # 標準化 RGB 數據
            rgb_mean = torch.mean(rgb_data, dim=(1, 2), keepdim=True)
            rgb_std = torch.std(rgb_data, dim=(1, 2), keepdim=True) + 1e-6  # 避免除以零
            rgb_data = (rgb_data - rgb_mean) / rgb_std

            depth_data = self._tiled_camera_front.data.output["depth"]
            # 處理無限深度值
            depth_data[depth_data == float("inf")] = 0.0
            # 歸一化深度值到 [0, 1] 範圍
            max_depth = self.cfg.tiled_camera_front.spawn.clipping_range[1]
            depth_data = depth_data / max_depth

            # 將 RGB 和 Depth 合併為一個張量
            camera_data = torch.cat([rgb_data, depth_data], dim=-1)  # [B, H, W, 4]
        elif data_type == "rgb":
            # 獲取RGB圖像並歸一化到[0,1]範圍
            camera_data = self._tiled_camera_front.data.output[data_type] / 255.0
            # 標準化處理
            mean = torch.mean(camera_data, dim=(1, 2), keepdim=True)
            std = torch.std(camera_data, dim=(1, 2), keepdim=True) + 1e-6  # 避免除以零
            camera_data = (camera_data - mean) / std
        elif data_type == "depth":
            # 獲取深度圖像
            camera_data = self._tiled_camera_front.data.output[data_type]
            # 處理無限深度值
            camera_data[camera_data == float("inf")] = 0.0
            # 歸一化深度值到 [0, 1] 範圍
            max_depth = self.cfg.tiled_camera_front.spawn.clipping_range[1]
            camera_data = camera_data / max_depth
        else:
            raise ValueError(f"Unsupported camera data type: {data_type}")

        # 可選：保存圖像到文件進行調試
        if hasattr(self.cfg, 'write_image_to_file') and self.cfg.write_image_to_file and self.episode_length_buf[0] % 100 == 0:
            import os
            import numpy as np
            from PIL import Image
            os.makedirs("debug_images", exist_ok=True)
            for image_index in range(min(self.num_envs, 5)):  # 只保存前5個環境的圖像以節省空間
                if self.is_rgbd_task:
                    # 保存 RGB 和 Depth 圖像
                    rgb_img_data = camera_data[image_index, :, :, :3].detach().cpu().numpy()
                    # 反標準化 RGB 數據
                    rgb_img_data = np.clip(rgb_img_data * rgb_std[image_index].cpu().numpy()
                                           + rgb_mean[image_index].cpu().numpy(), 0, 1) * 255
                    rgb_img_data = rgb_img_data.astype(np.uint8)
                    rgb_img = Image.fromarray(rgb_img_data)
                    rgb_img.save(f"debug_images/frame_{self.episode_length_buf[0]}_index_{image_index}_rgb.png")

                    depth_img_data = camera_data[image_index, :, :, 3].detach().cpu().numpy() * 255
                    depth_img_data = depth_img_data.astype(np.uint8)
                    depth_img = Image.fromarray(depth_img_data, mode="L")
                    depth_img.save(f"debug_images/frame_{self.episode_length_buf[0]}_index_{image_index}_depth.png")
                elif data_type == "rgb":
                    img_data = camera_data[image_index].detach().cpu().numpy()
                    img_data = np.clip(img_data * std[image_index].cpu().numpy()
                                       + mean[image_index].cpu().numpy(), 0, 1) * 255
                    img_data = img_data.astype(np.uint8)
                    img = Image.fromarray(img_data)
                    img.save(f"debug_images/frame_{self.episode_length_buf[0]}_index_{image_index}_rgb.png")
                else:  # depth
                    img_data = (camera_data[image_index].detach().cpu().numpy() * 255).astype(np.uint8).squeeze()
                    img = Image.fromarray(img_data, mode="L")
                    img.save(f"debug_images/frame_{self.episode_length_buf[0]}_index_{image_index}_depth.png")

        # 獲取前一步動作
        prev_actions = self.actions.clone()

        # 修改：將相機數據攤平並與前一步動作連接成一個張量
        batch_size = camera_data.shape[0]
        camera_data_flat = camera_data.reshape(batch_size, -1)  # 攤平相機數據

        # 連接相機數據和前一步動作
        policy_obs = torch.cat([camera_data_flat, prev_actions], dim=1)

        # 獲取critic的狀態向量（與原始FactoryEnv相同）
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

        # 返回攤平的張量給policy，狀態向量給critic
        return {"policy": policy_obs, "critic": state_tensors}
