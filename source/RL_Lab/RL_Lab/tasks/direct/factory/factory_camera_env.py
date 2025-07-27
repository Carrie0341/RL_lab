# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.sensors import TiledCamera

from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryRGBDCameraEnvCfg


class FactoryCameraEnv(FactoryEnv):
    """Factory environment with camera observations."""

    cfg: FactoryRGBDCameraEnvCfg  # 可以是 RGB 或 Depth 相機配置

    def __init__(self, cfg: FactoryRGBDCameraEnvCfg, render_mode: str | None = None, **kwargs):
        # 不需要更新 observation_space，因為它已經在配置中設置為 [128, 128, 3] 或 [128, 128, 1]
        self.cfg_task = cfg.task
        super().__init__(cfg, render_mode, **kwargs)
        self.is_rgbd_task = cfg.task_name == "peg_insert_rgbd_camera"
        self.is_multi_camera_task = True

    def _setup_scene(self):
        """Initialize simulation scene with camera."""
        # 調用父類方法設置基本場景
        super()._setup_scene()

        # 添加三個相機
        self._tiled_camera_1 = TiledCamera(self.cfg.tiled_camera_1)
        self._tiled_camera_2 = TiledCamera(self.cfg.tiled_camera_2)
        self._tiled_camera_3 = TiledCamera(self.cfg.tiled_camera_3)  # 新增第三個相機
        self.scene.sensors["camera1"] = self._tiled_camera_1
        self.scene.sensors["camera2"] = self._tiled_camera_2
        self.scene.sensors["camera3"] = self._tiled_camera_3  # 新增第三個相機

    def _get_observations(self):
        """Get observations from camera for policy and state vectors for critic."""

        if self.is_multi_camera_task:
            # 處理多相機情況
            camera_data_list = []

            # 處理第一個相機
            rgb_data1 = self._tiled_camera_1.data.output["rgb"] / 255.0
            rgb_mean1 = torch.mean(rgb_data1, dim=(1, 2), keepdim=True)
            rgb_std1 = torch.std(rgb_data1, dim=(1, 2), keepdim=True) + 1e-6
            rgb_data1 = (rgb_data1 - rgb_mean1) / rgb_std1

            # 獲取原始深度數據
            raw_depth_data1 = self._tiled_camera_1.data.output["depth"]
            raw_depth_data1[raw_depth_data1 == float("inf")] = 0.0
            
            # 修改：使用新的深度處理方法 - 限制範圍在0.2~1.2米之間
            max_depth1 = self.cfg.tiled_camera_1.spawn.clipping_range[1]
            # 將深度值限制在0.2~1.2米範圍內，然後正規化到[0,1]
            depth_data1 = torch.clamp(raw_depth_data1, min=0.2, max=1.2)
            depth_data1 = (depth_data1 - 0.2) / 1.0  # 從[0.2,1.2]正規化到[0,1]

            # 處理第二個相機
            rgb_data2 = self._tiled_camera_2.data.output["rgb"] / 255.0
            rgb_mean2 = torch.mean(rgb_data2, dim=(1, 2), keepdim=True)
            rgb_std2 = torch.std(rgb_data2, dim=(1, 2), keepdim=True) + 1e-6
            rgb_data2 = (rgb_data2 - rgb_mean2) / rgb_std2

            # 獲取原始深度數據
            raw_depth_data2 = self._tiled_camera_2.data.output["depth"]
            raw_depth_data2[raw_depth_data2 == float("inf")] = 0.0
            
            # 修改：使用新的深度處理方法 - 限制範圍在0.2~1.2米之間
            max_depth2 = self.cfg.tiled_camera_2.spawn.clipping_range[1]
            # 將深度值限制在0.2~1.2米範圍內，然後正規化到[0,1]
            depth_data2 = torch.clamp(raw_depth_data2, min=0.2, max=1.2)
            depth_data2 = (depth_data2 - 0.2) / 1.0  # 從[0.2,1.2]正規化到[0,1]

            # 處理第三個相機 (新增)
            rgb_data3 = self._tiled_camera_3.data.output["rgb"] / 255.0
            rgb_mean3 = torch.mean(rgb_data3, dim=(1, 2), keepdim=True)
            rgb_std3 = torch.std(rgb_data3, dim=(1, 2), keepdim=True) + 1e-6
            rgb_data3 = (rgb_data3 - rgb_mean3) / rgb_std3

            # 獲取原始深度數據
            raw_depth_data3 = self._tiled_camera_3.data.output["depth"]
            raw_depth_data3[raw_depth_data3 == float("inf")] = 0.0
            
            # 修改：使用新的深度處理方法 - 限制範圍在0.2~1.2米之間
            max_depth3 = self.cfg.tiled_camera_3.spawn.clipping_range[1]
            # 將深度值限制在0.2~1.2米範圍內，然後正規化到[0,1]
            depth_data3 = torch.clamp(raw_depth_data3, min=0.2, max=1.2)
            depth_data3 = (depth_data3 - 0.2) / 1.0  # 從[0.2,1.2]正規化到[0,1]

            # 合併RGB和Depth為RGBD
            camera_data1 = torch.cat([rgb_data1, depth_data1], dim=-1)  # [B, H, W, 4]
            camera_data2 = torch.cat([rgb_data2, depth_data2], dim=-1)  # [B, H, W, 4]
            camera_data3 = torch.cat([rgb_data3, depth_data3], dim=-1)  # [B, H, W, 4]

            # 攤平每個相機的數據
            batch_size = camera_data1.shape[0]
            camera_data1_flat = camera_data1.reshape(batch_size, -1)  # 攤平相機1數據
            camera_data2_flat = camera_data2.reshape(batch_size, -1)  # 攤平相機2數據
            camera_data3_flat = camera_data3.reshape(batch_size, -1)  # 攤平相機3數據

            # 獲取前一步動作
            prev_actions = self.actions.clone()

            # 連接所有相機數據和前一步動作
            policy_obs = torch.cat([camera_data1_flat, camera_data2_flat, camera_data3_flat, prev_actions], dim=1)

            # 可選：保存圖像到文件進行調試
            if hasattr(self.cfg, 'write_image_to_file') and self.cfg.write_image_to_file and self.episode_length_buf[0] % 999 == 0:
                # 保存原始深度數據以便在_save_debug_images中使用
                self._save_debug_images(
                    camera_data1, camera_data2, camera_data3,
                    rgb_data1, rgb_data2, rgb_data3,
                    depth_data1, depth_data2, depth_data3,
                    rgb_mean1, rgb_std1, rgb_mean2, rgb_std2, rgb_mean3, rgb_std3,
                    raw_depth_data1, raw_depth_data2, raw_depth_data3
                )

        else:
           assert False, "This environment is configured for multi-camera tasks only."
           
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
    
    
    def _save_debug_images(self, camera_data1, camera_data2, camera_data3, 
                          rgb_data1, rgb_data2, rgb_data3, 
                          depth_data1, depth_data2, depth_data3, 
                          rgb_mean1, rgb_std1, rgb_mean2, rgb_std2, rgb_mean3, rgb_std3, 
                          raw_depth_data1=None, raw_depth_data2=None, raw_depth_data3=None):
        """保存調試圖像到文件"""
        import os
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        os.makedirs("debug_images", exist_ok=True)
        show = [True, True, True]  # 是否顯示圖像，默認不顯示
        show_rgb = True  # 是否顯示RGB圖像，默認顯示
        show_gray = False  # 是否顯示灰階圖像，默認不顯示
        show_depth = True  # 是否顯示深度圖像，默認顯示
        for image_index in range(min(self.num_envs, 5)):  # 只保存前5個環境的圖像以節省空間
            if show[0]:
                # 保存第一個相機的圖像
                # RGB
                if show_rgb:
                    rgb_img_data1 = rgb_data1[image_index].detach().cpu().numpy()
                    rgb_img_data1 = np.clip(rgb_img_data1 * rgb_std1[image_index].cpu().numpy()
                                            + rgb_mean1[image_index].cpu().numpy(), 0, 1) * 255
                    rgb_img_data1 = rgb_img_data1.astype(np.uint8)
                    rgb_img1 = Image.fromarray(rgb_img_data1)
                    rgb_img1.save(f"debug_images/frame_{self.episode_length_buf[0]}_cam1_rgb_index_{image_index}.png")
                
                # 新增：RGB轉灰階圖像
                if show_gray:
                    rgb_gray_img1 = Image.fromarray(rgb_img_data1).convert('L')
                    rgb_gray_img1.save(f"debug_images/frame_{self.episode_length_buf[0]}_cam1_rgb_gray_index_{image_index}.png")
                
                # Depth - 新的處理方式（0.2-1.2米範圍）
                if show_depth:    
                    depth_img_data1 = depth_data1[image_index].detach().cpu().numpy() * 255
                    depth_img_data1 = depth_img_data1.astype(np.uint8).squeeze()
                    depth_img1 = Image.fromarray(depth_img_data1, mode="L")
                    depth_img1.save(f"debug_images/frame_{self.episode_length_buf[0]}_cam1_depth_index_{image_index}.png")
            if show[1]:
                # 保存第二個相機的圖像
                # RGB
                if show_rgb:
                    rgb_img_data2 = rgb_data2[image_index].detach().cpu().numpy()
                    rgb_img_data2 = np.clip(rgb_img_data2 * rgb_std2[image_index].cpu().numpy()
                                            + rgb_mean2[image_index].cpu().numpy(), 0, 1) * 255
                    rgb_img_data2 = rgb_img_data2.astype(np.uint8)
                    rgb_img2 = Image.fromarray(rgb_img_data2)
                    rgb_img2.save(f"debug_images/frame_{self.episode_length_buf[0]}_cam2_rgb_index_{image_index}.png")
                
                # 新增：RGB轉灰階圖像
                if show_gray:
                    rgb_gray_img2 = Image.fromarray(rgb_img_data2).convert('L')
                    rgb_gray_img2.save(f"debug_images/frame_{self.episode_length_buf[0]}_cam2_rgb_gray_index_{image_index}.png")
                
                # Depth - 新的處理方式（0.2-1.2米範圍）
                if show_depth:    
                    depth_img_data2 = depth_data2[image_index].detach().cpu().numpy() * 255
                    depth_img_data2 = depth_img_data2.astype(np.uint8).squeeze()
                    depth_img2 = Image.fromarray(depth_img_data2, mode="L")
                    depth_img2.save(f"debug_images/frame_{self.episode_length_buf[0]}_cam2_depth_index_{image_index}.png")
            if show[2]:
                # 保存第三個相機的圖像 (新增)
                # RGB
                if show_rgb:
                    rgb_img_data3 = rgb_data3[image_index].detach().cpu().numpy()
                    rgb_img_data3 = np.clip(rgb_img_data3 * rgb_std3[image_index].cpu().numpy()
                                            + rgb_mean3[image_index].cpu().numpy(), 0, 1) * 255
                    rgb_img_data3 = rgb_img_data3.astype(np.uint8)
                    rgb_img3 = Image.fromarray(rgb_img_data3)
                    rgb_img3.save(f"debug_images/frame_{self.episode_length_buf[0]}_cam3_rgb_index_{image_index}.png")
                
                # 新增：RGB轉灰階圖像
                if show_gray:
                    rgb_gray_img3 = Image.fromarray(rgb_img_data3).convert('L')
                    rgb_gray_img3.save(f"debug_images/frame_{self.episode_length_buf[0]}_cam3_rgb_gray_index_{image_index}.png")
                
                # Depth - 新的處理方式（0.2-1.2米範圍）
                if show_depth:    
                    depth_img_data3 = depth_data3[image_index].detach().cpu().numpy() * 255
                    depth_img_data3 = depth_img_data3.astype(np.uint8).squeeze()
                    depth_img3 = Image.fromarray(depth_img_data3, mode="L")
                    depth_img3.save(f"debug_images/frame_{self.episode_length_buf[0]}_cam3_depth_index_{image_index}.png")
            
            # 如果提供了原始深度數據，則創建各種深度可視化
            # if raw_depth_data1 is not None and raw_depth_data2 is not None and raw_depth_data3 is not None:
            #     # 獲取原始深度值
            #     raw_depth_np1 = raw_depth_data1[image_index].detach().cpu().numpy().squeeze()
            #     raw_depth_np2 = raw_depth_data2[image_index].detach().cpu().numpy().squeeze()
            #     raw_depth_np3 = raw_depth_data3[image_index].detach().cpu().numpy().squeeze()
                
            #     # 保存原始深度值的熱力圖
            #     plt.figure(figsize=(15, 5))
                
            #     # 第一個相機原始深度圖
            #     plt.subplot(1, 3, 1)
            #     depth_map1_raw = plt.imshow(raw_depth_np1, cmap='plasma', vmin=0, vmax=2.0)
            #     plt.colorbar(depth_map1_raw, label='Depth (meters)')
            #     plt.title(f'Camera 1 Raw Depth - Frame {self.episode_length_buf[0]}')
                
            #     # 第二個相機原始深度圖
            #     plt.subplot(1, 3, 2)
            #     depth_map2_raw = plt.imshow(raw_depth_np2, cmap='plasma', vmin=0, vmax=2.0)
            #     plt.colorbar(depth_map2_raw, label='Depth (meters)')
            #     plt.title(f'Camera 2 Raw Depth - Frame {self.episode_length_buf[0]}')
                
            #     # 第三個相機原始深度圖 (新增)
            #     plt.subplot(1, 3, 3)
            #     depth_map3_raw = plt.imshow(raw_depth_np3, cmap='plasma', vmin=0, vmax=2.0)
            #     plt.colorbar(depth_map3_raw, label='Depth (meters)')
            #     plt.title(f'Camera 3 Raw Depth - Frame {self.episode_length_buf[0]}')
                
            #     plt.tight_layout()
            #     plt.savefig(f"debug_images/frame_{self.episode_length_buf[0]}_index_{image_index}_raw_depth_heatmap.png", dpi=150)
            #     plt.close()
                
            #     # 保存新的深度處理方式的熱力圖（0.2-1.2米範圍）
            #     plt.figure(figsize=(15, 5))
                
            #     # 第一個相機的新深度處理
            #     plt.subplot(1, 3, 1)
            #     # 顯示限制在0.2-1.2米範圍內的深度值
            #     depth_map1_new = plt.imshow(np.clip(raw_depth_np1, 0.2, 1.2), cmap='turbo', vmin=0.2, vmax=1.2)
            #     plt.colorbar(depth_map1_new, label='Depth (meters)')
            #     plt.title(f'Camera 1 New Depth (0.2-1.2m) - Frame {self.episode_length_buf[0]}')
                
            #     # 第二個相機的新深度處理
            #     plt.subplot(1, 3, 2)
            #     depth_map2_new = plt.imshow(np.clip(raw_depth_np2, 0.2, 1.2), cmap='turbo', vmin=0.2, vmax=1.2)
            #     plt.colorbar(depth_map2_new, label='Depth (meters)')
            #     plt.title(f'Camera 2 New Depth (0.2-1.2m) - Frame {self.episode_length_buf[0]}')
                
            #     # 第三個相機的新深度處理 (新增)
            #     plt.subplot(1, 3, 3)
            #     depth_map3_new = plt.imshow(np.clip(raw_depth_np3, 0.2, 1.2), cmap='turbo', vmin=0.2, vmax=1.2)
            #     plt.colorbar(depth_map3_new, label='Depth (meters)')
            #     plt.title(f'Camera 3 New Depth (0.2-1.2m) - Frame {self.episode_length_buf[0]}')
                
            #     plt.tight_layout()
            #     plt.savefig(f"debug_images/frame_{self.episode_length_buf[0]}_index_{image_index}_new_depth_range_heatmap.png", dpi=150)
            #     plt.close()