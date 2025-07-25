import torch
import torch.nn as nn
import numpy as np
from rl_games.algos_torch import network_builder
from rl_games.algos_torch import model_builder


class VisualFactoryTestNetwork(network_builder.NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)

        self.actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.device = kwargs.get('device', 'cuda:0')

        # 固定參數設置
        self.num_seqs = 64  # 固定批次大小
        self.action_dim = 6  # 固定動作維度

        # 固定相機參數
        self.img_height = 128+64
        self.img_width = 128+64
        self.img_channels = 4  # RGBD
        self.rgb_channels = 3  # RGB通道數
        self.depth_channels = 1  # 深度通道數
        self.num_cameras = 2  # 固定為2個相機

        # 計算相機數據大小
        self.single_camera_size = self.img_height * self.img_width * self.img_channels
        self.camera_size = self.single_camera_size * self.num_cameras  # 2個相機的總大小
        self.input_size = self.camera_size + self.action_dim

        # 固定使用卷積編碼器
        self.use_linear_encoder = False
        self.is_multi_camera = True  # 設置為多相機模式

        # 創建共用的RGB和深度編碼器
        self.rgb_encoder = self._build_rgb_encoder()
        self.depth_encoder = self._build_depth_encoder()

        # 計算CNN輸出尺寸 - RGB編碼器
        with torch.no_grad():
            rgb_sample = torch.zeros((1, self.rgb_channels, self.img_height, self.img_width))
            rgb_out = self.rgb_encoder(rgb_sample)
            self.rgb_out_shape = rgb_out.shape[1:]
            self.rgb_encoder_out_size = np.prod(self.rgb_out_shape)

            # 深度編碼器
            depth_sample = torch.zeros((1, self.depth_channels, self.img_height, self.img_width))
            depth_out = self.depth_encoder(depth_sample)
            self.depth_out_shape = depth_out.shape[1:]
            self.depth_encoder_out_size = np.prod(self.depth_out_shape)

            # 單個相機的特徵大小 (RGB + 深度)
            self.single_camera_feature_size = self.rgb_encoder_out_size + self.depth_encoder_out_size

        # 為每個相機視角創建獨立的MLP
        self.camera_mlps = nn.ModuleList()
        for _ in range(self.num_cameras):
            self.camera_mlps.append(nn.Sequential(
                nn.Linear(self.single_camera_feature_size, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU()
            ))

        # 每個相機MLP的輸出大小
        self.camera_mlp_out_size = 64

        # 所有相機特徵的總大小
        self.visual_out_size = self.camera_mlp_out_size * self.num_cameras

        # 動作編碼器 (固定結構)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU()
        )
        action_out_size = 32

        # 計算特徵融合後的大小
        combined_features_size = self.visual_out_size + action_out_size

        # 固定不使用RNN
        self._is_rnn = False
        self.rnn_units = 0
        self.rnn_layers = 0

        # 融合MLP層 (固定結構)
        self.mlp = nn.Sequential(
            nn.Linear(combined_features_size, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )

        # 輸出層
        self.value = nn.Linear(128, 1)
        self.mu = nn.Linear(128, self.actions_num)
        self.sigma = nn.Parameter(torch.zeros(self.actions_num))

        # 初始化權重
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.orthogonal_(m.weight, gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 特別初始化輸出層
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.zeros_(self.mu.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

        # 備用編碼器 (簡化版)
        self.fallback_encoder = nn.Sequential(
            nn.Linear(self.camera_size, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        ).to(self.device)

    def _build_rgb_encoder(self):
        """構建RGB影像的卷積編碼器"""
        return nn.Sequential(
            # 第一層卷積 - 減小kernel size以保留更多細節
            nn.Conv2d(self.rgb_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            # 第二層卷積
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # 第三層卷積
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )

    def _build_depth_encoder(self):
        """構建深度影像的卷積編碼器 - 增強深度處理能力"""
        return nn.Sequential(
            # 第一層卷積 - 使用較小的kernel以保留精細深度變化
            nn.Conv2d(self.depth_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            # 第二層卷積
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # 第三層卷積
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 增加通道數
            nn.ELU(),
        )

    def is_rnn(self):
        """返回網絡是否使用RNN"""
        return False  # 固定不使用RNN

    def get_default_rnn_state(self):
        """返回默認的RNN狀態"""
        return None  # 固定不使用RNN

    def forward(self, obs_tensor, rnn_states=None):
        """
        處理觀測輸入，固定處理邏輯
        """
        # 從輸入中提取相機數據和前一步動作
        if isinstance(obs_tensor, dict):
            if 'obs' in obs_tensor:
                obs = obs_tensor['obs']
                if isinstance(obs, dict):
                    camera_data = obs['camera_data']
                    prev_actions = obs['prev_actions']
                else:
                    camera_data = obs[:, :-self.action_dim]
                    prev_actions = obs[:, -self.action_dim:]
            else:
                camera_data = obs_tensor[:, :-self.action_dim]
                prev_actions = obs_tensor[:, -self.action_dim:]
        else:
            camera_data = obs_tensor[:, :-self.action_dim]
            prev_actions = obs_tensor[:, -self.action_dim:]

        batch_size = camera_data.shape[0]

        try:
            # 處理多相機數據
            all_camera_features = []

            for i in range(self.num_cameras):
                # 提取每個相機的數據
                start_idx = i * self.single_camera_size
                end_idx = (i + 1) * self.single_camera_size
                single_camera_data = camera_data[:, start_idx:end_idx]

                # 重塑為[B, H, W, C]格式
                single_camera_data = single_camera_data.reshape(batch_size, self.img_height, self.img_width, self.img_channels)

                # 分離RGB和深度通道
                rgb_data = single_camera_data[:, :, :, :3]  # 前3個通道是RGB
                depth_data = single_camera_data[:, :, :, 3:4]  # 第4個通道是深度

                # 轉換為[B, C, H, W]格式
                rgb_data = rgb_data.permute(0, 3, 1, 2)  # [B, 3, H, W]
                depth_data = depth_data.permute(0, 3, 1, 2)  # [B, 1, H, W]

                # 使用共用編碼器處理RGB和深度數據
                rgb_features = self.rgb_encoder(rgb_data)
                depth_features = self.depth_encoder(depth_data)

                # 扁平化特徵
                rgb_features = rgb_features.reshape(batch_size, -1)
                depth_features = depth_features.reshape(batch_size, -1)

                # 連接RGB和深度特徵
                camera_features = torch.cat([rgb_features, depth_features], dim=1)

                # 使用該相機視角的MLP處理特徵
                camera_processed = self.camera_mlps[i](camera_features)

                all_camera_features.append(camera_processed)

            # 連接所有相機特徵
            visual_features = torch.cat(all_camera_features, dim=1)

        except RuntimeError:
            # 使用備用編碼器
            visual_features = self.fallback_encoder(camera_data)

        # 動作特徵提取
        action_features = self.action_encoder(prev_actions)

        # 特徵融合
        combined_features = torch.cat([visual_features, action_features], dim=1)

        # MLP處理
        mlp_out = self.mlp(combined_features)

        # 輸出層
        value = self.value(mlp_out)
        mu = self.mu(mlp_out)
        sigma = self.sigma.expand_as(mu)

        return mu, sigma, value, None  # 固定返回None作為RNN狀態


class VisualFactoryTestNetworkBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        network_builder.NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params
        return self

    def build(self, name, **kwargs):
        return VisualFactoryTestNetwork(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)


# 註冊網絡
model_builder.register_network('visual_factory_test', VisualFactoryTestNetworkBuilder)
