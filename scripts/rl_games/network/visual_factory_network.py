import torch
import torch.nn as nn
import numpy as np
from rl_games.algos_torch import network_builder
from rl_games.algos_torch import model_builder


class VisualFactoryNetwork(network_builder.NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)

        self.actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.device = kwargs.get('device', 'cuda:0')

        # 儲存配置參數
        self.params = params
        # 默認批次大小為16
        self.num_seqs = 16
        if 'config' in params and 'num_actors' in params['config']:
            self.num_seqs = params['config']['num_actors']

        # 處理輸入形狀
        # 獲取動作空間大小
        self.action_dim = 6  # 固定為6維動作

        # 獲取實際輸入大小
        if isinstance(input_shape, int):
            # 如果輸入是一個整數，表示攤平的觀測
            self.input_size = input_shape
            # 計算相機數據大小 = 總大小 - 動作維度
            self.camera_size = self.input_size - self.action_dim

            # 使用實際的256x256x4相機尺寸
            self.img_height = 256
            self.img_width = 256
            self.img_channels = 4  # RGBD

            # 檢查計算出的相機大小是否正確
            expected_camera_size = self.img_height * self.img_width * self.img_channels
            if expected_camera_size != self.camera_size:
                # print(f"# [Warning] Calculated camera size {self.camera_size} doesn't match expected {expected_camera_size}")
                # 嘗試調整尺寸使其匹配
                self.use_linear_encoder = True
            else:
                self.use_linear_encoder = False
        else:
            # 如果是(H, W, C)形式或其他格式，使用默認值
            self.img_height = 256
            self.img_width = 256
            self.img_channels = 4  # RGBD
            self.camera_size = self.img_height * self.img_width * self.img_channels
            self.input_size = self.camera_size + self.action_dim
            self.use_linear_encoder = False

        # print(f"# [Debug] Network initialized with camera_size={self.camera_size}, input_size={self.input_size}")
        # print(f"# [Debug] Image dimensions: {self.img_height}x{self.img_width}x{self.img_channels}")

        # 構建視覺編碼器
        if self.use_linear_encoder:
            # 使用線性層處理攤平的輸入
            visual_encoder = []

            # 創建3層全連接網絡
            hidden_sizes = [1024, 512, 256]

            for i, hidden_size in enumerate(hidden_sizes):
                if i == 0:
                    visual_encoder.append(nn.Linear(self.camera_size, hidden_size))
                else:
                    visual_encoder.append(nn.Linear(hidden_sizes[i - 1], hidden_size))

                # 添加激活函數
                visual_encoder.append(nn.ELU())

            self.visual_encoder = nn.Sequential(*visual_encoder)

            # 計算視覺編碼器輸出大小
            self.visual_out_size = hidden_sizes[-1]
        else:
            # 使用卷積網絡處理圖像
            visual_encoder = []

            # 添加卷積層
            conv_configs = params['visual_encoder']['convs']
            for i, conv_config in enumerate(conv_configs):
                if i == 0:
                    input_channels = self.img_channels
                else:
                    input_channels = conv_configs[i - 1]['filters']

                visual_encoder.append(
                    nn.Conv2d(
                        input_channels,
                        conv_config['filters'],
                        kernel_size=conv_config['kernel_size'],
                        stride=conv_config['strides'],
                        padding=conv_config['padding']
                    )
                )

                # 添加激活函數
                if params['visual_encoder']['activation'] == 'elu':
                    visual_encoder.append(nn.ELU())
                elif params['visual_encoder']['activation'] == 'relu':
                    visual_encoder.append(nn.ReLU())
                elif params['visual_encoder']['activation'] == 'tanh':
                    visual_encoder.append(nn.Tanh())

            self.visual_encoder = nn.Sequential(*visual_encoder)

            # 計算CNN輸出尺寸
            with torch.no_grad():
                # 創建一個樣本輸入 [B, C, H, W]
                sample_input = torch.zeros((1, self.img_channels, self.img_height, self.img_width))
                cnn_out = self.visual_encoder(sample_input)
                self.cnn_out_shape = cnn_out.shape[1:]  # [C', H', W']
                self.visual_out_size = np.prod(self.cnn_out_shape)

        # 動作編碼器 (MLP)
        action_encoder = []

        action_units = params['action_encoder']['units']
        for i, units in enumerate(action_units):
            if i == 0:
                action_encoder.append(nn.Linear(self.action_dim, units))
            else:
                action_encoder.append(nn.Linear(action_units[i - 1], units))

            # 添加激活函數
            if params['action_encoder']['activation'] == 'elu':
                action_encoder.append(nn.ELU())
            elif params['action_encoder']['activation'] == 'relu':
                action_encoder.append(nn.ReLU())
            elif params['action_encoder']['activation'] == 'tanh':
                action_encoder.append(nn.Tanh())

        self.action_encoder = nn.Sequential(*action_encoder)
        action_out_size = action_units[-1]

        # 注意力機制 (可選)
        self.attention = None
        if 'attention' in params['visual_encoder'] and params['visual_encoder']['attention']['enabled']:
            self.attention = nn.MultiheadAttention(
                embed_dim=conv_configs[-1]['filters'],
                num_heads=params['visual_encoder']['attention']['heads'],
                kdim=params['visual_encoder']['attention']['key_dim'],
                vdim=params['visual_encoder']['attention']['key_dim']
            )

        # 計算特徵融合後的大小
        combined_features_size = self.visual_out_size + action_out_size

        # LSTM層 (可選)
        self._is_rnn = False
        self.rnn_units = 0
        self.rnn_layers = 0

        if 'rnn' in params:
            self._is_rnn = True
            self.rnn_units = params['rnn']['units']
            self.rnn_layers = params['rnn']['layers']
            self.rnn_name = params['rnn']['name']

            if self.rnn_name == 'lstm':
                self.rnn = nn.LSTM(
                    combined_features_size,
                    self.rnn_units,
                    self.rnn_layers,
                    batch_first=True
                )
            elif self.rnn_name == 'gru':
                self.rnn = nn.GRU(
                    combined_features_size,
                    self.rnn_units,
                    self.rnn_layers,
                    batch_first=True
                )

            mlp_input_size = self.rnn_units
        else:
            mlp_input_size = combined_features_size

        # 融合MLP層
        mlp_layers = []

        mlp_units = params['mlp']['units']
        for i, units in enumerate(mlp_units):
            if i == 0:
                mlp_layers.append(nn.Linear(mlp_input_size, units))
            else:
                mlp_layers.append(nn.Linear(mlp_units[i - 1], units))

            # 添加激活函數
            if params['mlp']['activation'] == 'elu':
                mlp_layers.append(nn.ELU())
            elif params['mlp']['activation'] == 'relu':
                mlp_layers.append(nn.ReLU())
            elif params['mlp']['activation'] == 'tanh':
                mlp_layers.append(nn.Tanh())

        self.mlp = nn.Sequential(*mlp_layers)

        # 輸出層
        self.value = nn.Linear(mlp_units[-1], 1)
        self.mu = nn.Linear(mlp_units[-1], self.actions_num)
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

        # print(f"# [Debug] initialized VisualFactoryNetwork with params: {params}")

    def is_rnn(self):
        """返回網絡是否使用RNN"""
        return self._is_rnn

    def get_default_rnn_state(self):
        """返回默認的RNN狀態 - 符合RL-Games的期望接口"""
        if not self._is_rnn:
            return None

        num_layers = self.rnn_layers
        # 使用self.num_seqs作為批次大小
        batch_size = self.num_seqs
        hidden_size = self.rnn_units
        device = self.device

        if self.rnn_name == 'lstm':
            return (
                torch.zeros((num_layers, batch_size, hidden_size), device=device),
                torch.zeros((num_layers, batch_size, hidden_size), device=device)
            )
        else:  # GRU
            return torch.zeros((num_layers, batch_size, hidden_size), device=device)

    def _get_rnn_state(self, batch_size, device):
        """內部方法，用於在forward過程中獲取RNN狀態"""
        if not self._is_rnn:
            return None

        num_layers = self.rnn_layers
        hidden_size = self.rnn_units

        if self.rnn_name == 'lstm':
            return (
                torch.zeros((num_layers, batch_size, hidden_size), device=device),
                torch.zeros((num_layers, batch_size, hidden_size), device=device)
            )
        else:  # GRU
            return torch.zeros((num_layers, batch_size, hidden_size), device=device)

    def forward(self, obs_tensor, rnn_states=None):
        """
        處理觀測輸入，可以是tensor或dict格式

        Args:
            obs_tensor: 可以是tensor或dict格式的觀測值
            rnn_states: RNN狀態 (如果使用RNN)
        """

        # print(f"# [Debug] VisualFactoryNetwork forward called with obs_tensor: {obs_tensor.shape if isinstance(obs_tensor, torch.Tensor) else obs_tensor['obs'].shape}")

        # 處理輸入 - 從攤平的tensor中提取相機數據和前一步動作
        if isinstance(obs_tensor, dict):
            if 'obs' in obs_tensor:
                obs = obs_tensor['obs']
                if isinstance(obs, dict):
                    camera_data = obs['camera_data']
                    prev_actions = obs['prev_actions']
                else:
                    # 處理攤平的觀測向量
                    camera_data = obs[:, :-self.action_dim]
                    prev_actions = obs[:, -self.action_dim:]
            else:
                # 舊格式，需要拆分
                camera_data = obs_tensor[:, :-self.action_dim]
                prev_actions = obs_tensor[:, -self.action_dim:]
        else:
            # 直接從tensor中提取數據
            camera_data = obs_tensor[:, :-self.action_dim]
            prev_actions = obs_tensor[:, -self.action_dim:]

        batch_size = camera_data.shape[0]
        device = camera_data.device

        # 視覺特徵提取
        if self.use_linear_encoder:
            # 直接使用線性層處理攤平的數據
            visual_features = self.visual_encoder(camera_data)
        else:
            try:
                # 重塑相機數據為 [B, C, H, W] 格式
                camera_data = camera_data.reshape(batch_size, self.img_height, self.img_width, self.img_channels)
                camera_data = camera_data.permute(0, 3, 1, 2)  # [B, C, H, W]

                # 通過CNN處理
                visual_features = self.visual_encoder(camera_data)
                visual_features = visual_features.reshape(batch_size, -1)  # 扁平化
            except RuntimeError as e:
                # print(f"Error reshaping camera data: {e}")
                # print(f"Camera data shape: {camera_data.shape}")
                # print(f"Expected shape after reshape: [{batch_size}, {self.img_height}, {self.img_width}, {self.img_channels}]")
                # print(f"Total elements in camera_data: {camera_data.numel()}")
                # print(f"Expected elements: {batch_size * self.img_height * self.img_width * self.img_channels}")

                # 如果沒有備用編碼器，創建一個
                if not hasattr(self, 'fallback_encoder'):
                    fallback_encoder = []
                    input_size = camera_data.shape[1]
                    hidden_sizes = [512, 256, 128]

                    for i, hidden_size in enumerate(hidden_sizes):
                        if i == 0:
                            fallback_encoder.append(nn.Linear(input_size, hidden_size))
                        else:
                            fallback_encoder.append(nn.Linear(hidden_sizes[i - 1], hidden_size))
                        fallback_encoder.append(nn.ELU())

                    self.fallback_encoder = nn.Sequential(*fallback_encoder).to(device)

                # 使用備用編碼器
                visual_features = self.fallback_encoder(camera_data)

        # 動作特徵提取
        action_features = self.action_encoder(prev_actions)

        # 特徵融合
        combined_features = torch.cat([visual_features, action_features], dim=1)

        # LSTM處理 (如果有)
        if self._is_rnn:
            if len(combined_features.shape) == 2:
                combined_features = combined_features.unsqueeze(1)  # 添加時間維度

            if rnn_states is None:
                # 獲取默認的RNN狀態
                rnn_states = self._get_rnn_state(batch_size, device)
            else:
                # 確保RNN狀態在正確的設備上
                if isinstance(rnn_states, tuple):  # LSTM
                    h, c = rnn_states
                    if h.device != device:
                        h = h.to(device)
                        c = c.to(device)
                    rnn_states = (h, c)
                else:  # GRU
                    if rnn_states.device != device:
                        rnn_states = rnn_states.to(device)

            # 使用RNN處理
            combined_features, new_rnn_states = self.rnn(combined_features, rnn_states)

            if len(combined_features.shape) == 3:
                combined_features = combined_features[:, -1]  # 取最後一個時間步
        else:
            new_rnn_states = None

        # MLP處理
        mlp_out = self.mlp(combined_features)

        # 輸出層
        value = self.value(mlp_out)
        mu = self.mu(mlp_out)
        sigma = self.sigma.expand_as(mu)

        # print(f"# [Debug] mu shape: {mu.shape}, sigma shape: {sigma.shape}, value shape: {value.shape}")

        return mu, sigma, value, new_rnn_states


class VisualFactoryNetworkBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        network_builder.NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params
        return self

    def build(self, name, **kwargs):
        return VisualFactoryNetwork(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)


# 註冊網絡
model_builder.register_network('visual_factory', VisualFactoryNetworkBuilder)
