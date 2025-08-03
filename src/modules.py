import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticEncoder(nn.Module):
    """
    语义编码器，用于将文本标签转换为语义特征向量。
    该模块在训练时为视觉模型提供"剧透"信息，在推理时被禁用。
    """
    def __init__(self, num_classes, embed_size=128, hidden_size=256):
        super(SemanticEncoder, self).__init__()
        self.embedding = nn.Embedding(num_classes, embed_size)
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False  # 单向即可，我们只需要一个最终的上下文表示
        )
        self.hidden_size = hidden_size

    def forward(self, text_indices, text_lengths):
        """
        Args:
            text_indices (torch.Tensor): 形状为 [B, max_len] 的文本索引
            text_lengths (torch.Tensor): 形状为 [B] 的文本实际长度

        Returns:
            torch.Tensor: 形状为 [B, hidden_size] 的语义特征向量
        """
        # 嵌入
        embedded = self.embedding(text_indices)  # [B, max_len, embed_size]

        # 为了处理可变长度序列，我们打包序列
        # 注意: pack_padded_sequence 需要 length 在 cpu上
        packed = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # RNN前向传播
        _, hidden = self.rnn(packed)  # hidden shape: [num_layers*num_directions, B, hidden_size]
        
        # 我们只需要最后一个时间步的隐藏状态
        # GRU的hidden是最后一个时间步的隐藏状态
        # 形状为 [1, B, hidden_size]，需要调整为 [B, hidden_size]
        semantic_feature = hidden.squeeze(0)
        
        return semantic_feature


class VGG_FeatureExtractor(nn.Module):
    """VGG网络特征提取器"""
    def __init__(self, input_channel=1, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = output_channel

        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, output_channel, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.ConvNet(x)


class SELayer(nn.Module):
    """Squeeze-and-Excitation注意力机制"""
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MobileNetV3Block(nn.Module):
    """MobileNetV3的基础模块，使用SE注意力机制"""
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(MobileNetV3Block, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.use_hs = use_hs
        self.inp = inp
        self.oup = oup

        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # 逐点卷积
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True) if not use_hs else nn.Hardswish(inplace=True),
            # 深度可分离卷积
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True) if not use_hs else nn.Hardswish(inplace=True),
            # SE模块
            SELayer(hidden_dim) if use_se else nn.Identity(),
            # 逐点卷积
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )
        
        # 在stride=1且输入输出通道数相同时使用跳跃连接
        # 在stride=1且输入输出通道数不同时使用1x1卷积进行维度匹配
        if stride == 1:
            if inp == oup:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup)
                )
        else:
            self.shortcut = nn.Sequential()  # 空序列，不进行跳跃连接

    def forward(self, x):
        if self.stride == 1:
            return self.conv(x) + self.shortcut(x)
        else:
            return self.conv(x)


class MobileNetV3_FeatureExtractor(nn.Module):
    """MobileNetV3特征提取器"""
    def __init__(self, input_channel=1, output_channel=512):
        super(MobileNetV3_FeatureExtractor, self).__init__()
        self.output_channel = output_channel
        
        # 第一层
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True)
        )
        
        # MobileNetV3 blocks
        self.blocks = nn.Sequential(
            MobileNetV3Block(16, 16, 16, 3, 1, False, False),
            MobileNetV3Block(16, 64, 24, 3, 2, False, False),
            MobileNetV3Block(24, 72, 24, 3, 1, False, False),
            MobileNetV3Block(24, 72, 40, 5, 2, True, False),
            MobileNetV3Block(40, 120, 40, 5, 1, True, False),
            MobileNetV3Block(40, 120, 40, 5, 1, True, False),
            MobileNetV3Block(40, 240, 80, 3, 2, False, True),
            MobileNetV3Block(80, 200, 80, 3, 1, False, True),
            MobileNetV3Block(80, 184, 80, 3, 1, False, True),
            MobileNetV3Block(80, 184, 80, 3, 1, False, True),
            MobileNetV3Block(80, 480, 112, 3, 1, True, True),
            MobileNetV3Block(112, 672, 112, 3, 1, True, True),
            MobileNetV3Block(112, 672, 160, 5, 1, True, True),
            MobileNetV3Block(160, 672, 160, 5, 2, True, True),
            MobileNetV3Block(160, 960, 160, 5, 1, True, True),
        )
        
        # 最后一层
        self.conv2 = nn.Sequential(
            nn.Conv2d(160, self.output_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.output_channel),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        return x


class MobileNetV3M_FeatureExtractor(nn.Module):
    """自定义轻量化MobileNetV3特征提取器"""
    def __init__(self, input_channel=1, output_channel=512):
        super(MobileNetV3M_FeatureExtractor, self).__init__()
        self.output_channel = output_channel
        
        # 第一层
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 简化的MobileNetV3 blocks
        self.blocks = nn.Sequential(
            MobileNetV3Block(16, 64, 24, 3, 1, False, False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MobileNetV3Block(24, 72, 40, 3, 1, True, False),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            MobileNetV3Block(40, 120, 80, 3, 1, True, True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        
        # 最后一层
        self.conv2 = nn.Sequential(
            nn.Conv2d(80, self.output_channel, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.output_channel),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        return x
