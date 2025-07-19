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
        self.num_seqs = 32
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

            # 檢查是否是多相機設置
            expected_single_camera_size = self.img_height * self.img_width * self.img_channels
            self.num_cameras = self.camera_size // expected_single_camera_size

            if self.num_cameras > 1 and self.camera_size % expected_single_camera_size == 0:
                # 多相機設置
                self.is_multi_camera = True
                self.single_camera_size = expected_single_camera_size
            else:
                # 單相機或不規則大小
                self.is_multi_camera = False
                self.single_camera_size = self.camera_size

            # 檢查計算出的相機大小是否正確
            if self.is_multi_camera:
                # 多相機設置
                self.use_linear_encoder = False
            elif expected_single_camera_size != self.camera_size:
                # 單相機但大小不匹配
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
            self.is_multi_camera = False
            self.single_camera_size = self.camera_size
            self.num_cameras = 1

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
            # 如果是多相機，為每個相機創建一個編碼器
            if self.is_multi_camera:
                self.camera_encoders = nn.ModuleList()
                for _ in range(self.num_cameras):
                    encoder = self._build_conv_encoder(params)
                    self.camera_encoders.append(encoder)

                # 計算單個相機編碼器的輸出大小
                with torch.no_grad():
                    # 創建一個樣本輸入 [B, C, H, W]
                    sample_input = torch.zeros((1, self.img_channels, self.img_height, self.img_width))
                    cnn_out = self.camera_encoders[0](sample_input)
                    self.cnn_out_shape = cnn_out.shape[1:]  # [C', H', W']
                    self.single_encoder_out_size = np.prod(self.cnn_out_shape)

                # 總視覺輸出大小是所有相機編碼器輸出的總和
                self.visual_out_size = self.single_encoder_out_size * self.num_cameras
            else:
                # 單相機設置
                self.visual_encoder = self._build_conv_encoder(params)

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
            if self.is_multi_camera:
                # 對於多相機，使用注意力機制融合不同相機的特徵
                conv_configs = params['visual_encoder']['convs']
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

    def _build_conv_encoder(self, params):
        """構建卷積編碼器"""
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

        return nn.Sequential(*visual_encoder)

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
                if self.is_multi_camera:
                    # 多相機處理
                    all_visual_features = []

                    for i in range(self.num_cameras):
                        # 提取每個相機的數據
                        start_idx = i * self.single_camera_size
                        end_idx = (i + 1) * self.single_camera_size
                        single_camera_data = camera_data[:, start_idx:end_idx]

                        # 重塑為[B, C, H, W]格式
                        single_camera_data = single_camera_data.reshape(batch_size, self.img_height, self.img_width, self.img_channels)
                        single_camera_data = single_camera_data.permute(0, 3, 1, 2)  # [B, C, H, W]

                        # 使用對應的編碼器處理
                        features = self.camera_encoders[i](single_camera_data)
                        features = features.reshape(batch_size, -1)  # 扁平化
                        all_visual_features.append(features)

                    # 如果啟用了注意力機制，使用它來融合特徵
                    if self.attention is not None:
                        # 重塑特徵以適應注意力機制 [seq_len, batch, embed_dim]
                        stacked_features = torch.stack(all_visual_features, dim=0)
                        # 應用注意力
                        attn_output, _ = self.attention(stacked_features, stacked_features, stacked_features)
                        # 重塑回原始形狀並攤平
                        visual_features = attn_output.sum(dim=0)  # [batch, embed_dim]
                    else:
                        # 否則，簡單地連接所有特徵
                        visual_features = torch.cat(all_visual_features, dim=1)
                else:
                    # 單相機處理
                    # 重塑相機數據為 [B, C, H, W] 格式
                    camera_data = camera_data.reshape(batch_size, self.img_height, self.img_width, self.img_channels)
                    camera_data = camera_data.permute(0, 3, 1, 2)  # [B, C, H, W]

                    # 通過CNN處理
                    visual_features = self.visual_encoder(camera_data)
                    visual_features = visual_features.reshape(batch_size, -1)  # 扁平化
            except RuntimeError as e:
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

            # 檢查輸入尺寸是否匹配
            if combined_features.size(-1) != self.rnn.input_size:
                # 打印調試信息
                print(f"WARNING: LSTM input size mismatch. Expected {self.rnn.input_size}, got {combined_features.size(-1)}")
                print(f"Visual features size: {visual_features.size()}, Action features size: {action_features.size()}")

                # 使用線性層調整尺寸
                if not hasattr(self, 'lstm_adapter'):
                    self.lstm_adapter = nn.Linear(combined_features.size(-1), self.rnn.input_size).to(device)
                    # 使用正交初始化
                    nn.init.orthogonal_(self.lstm_adapter.weight, gain=1.4142)
                    nn.init.zeros_(self.lstm_adapter.bias)
                    print("Created LSTM adapter layer")

                # 調整尺寸
                combined_features = self.lstm_adapter(combined_features)
                print(f"Adjusted combined features size: {combined_features.size()}")

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
