"""
LCRNN (Lightweight CRNN) 模型
基于MobileNetV3的轻量级CRNN实现，适用于移动端部署
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardSwish(nn.Module):
    """HardSwish激活函数"""
    
    def forward(self, x):
        return x * F.hardsigmoid(x)


class SqueezeExcitation(nn.Module):
    """SE模块（Squeeze-and-Excitation）"""
    
    def __init__(self, in_channels, se_channels):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, se_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_channels, in_channels, 1),
            nn.Hardsigmoid(inplace=True)
        )
    
    def forward(self, x):
        return x * self.se(x)


class MobileBottleneck(nn.Module):
    """MobileNetV3的反向残差块"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0, activation='relu'):
        super(MobileBottleneck, self).__init__()
        
        hidden_dim = int(in_channels * expand_ratio)
        self.identity = stride == 1 and in_channels == out_channels
        
        # 激活函数
        if activation == 'relu':
            act_layer = nn.ReLU
        elif activation == 'hardswish':
            act_layer = HardSwish
        else:
            raise NotImplementedError
        
        layers = []
        
        # 扩展层
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act_layer(inplace=True)
            ])
        
        # 深度卷积
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_layer(inplace=True)
        ])
        
        # SE模块
        if se_ratio > 0:
            layers.append(SqueezeExcitation(hidden_dim, int(hidden_dim * se_ratio)))
        
        # 压缩层
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3Backbone(nn.Module):
    """MobileNetV3特征提取骨干网络"""
    
    def __init__(self):
        super(MobileNetV3Backbone, self).__init__()
        
        # 第一层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )
        
        # MobileBottleneck层配置
        # [in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, activation]
        configs = [
            [16, 16, 3, 1, 1, 0, 'relu'],
            [16, 24, 3, 2, 4, 0, 'relu'],
            [24, 24, 3, 1, 3, 0, 'relu'],
            [24, 40, 5, 2, 3, 0.25, 'relu'],
            [40, 40, 5, 1, 3, 0.25, 'relu'],
            [40, 40, 5, 1, 3, 0.25, 'relu'],
            [40, 80, 3, 2, 6, 0, 'hardswish'],
            [80, 80, 3, 1, 2.5, 0, 'hardswish'],
            [80, 80, 3, 1, 2.3, 0, 'hardswish'],
            [80, 80, 3, 1, 2.3, 0, 'hardswish'],
            [80, 112, 3, 1, 6, 0.25, 'hardswish'],
            [112, 112, 3, 1, 6, 0.25, 'hardswish'],
            [112, 160, 5, 1, 6, 0.25, 'hardswish'],
            [160, 160, 5, 1, 6, 0.25, 'hardswish'],
            [160, 160, 5, 1, 6, 0.25, 'hardswish'],
        ]
        
        layers = []
        for config in configs:
            layers.append(MobileBottleneck(*config))
        
        self.layers = nn.Sequential(*layers)
        
        # 最后的卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            HardSwish()
        )
        
        # 池化层，用于降低高度维度
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.conv2(x)
        x = self.pool(x)  # (batch, 960, 1, width)
        
        # 重塑为序列
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, width, channels)  # (batch, width, 960)
        
        return x


class LCRNN(nn.Module):
    """轻量级CRNN模型"""
    
    def __init__(self, num_classes, hidden_size=128, num_layers=2):
        super(LCRNN, self).__init__()
        
        # MobileNetV3特征提取器
        self.backbone = MobileNetV3Backbone()
        
        # 双向LSTM
        self.rnn = nn.LSTM(
            input_size=960,  # MobileNetV3输出通道数
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # 特征提取
        features = self.backbone(x)  # (batch, width, 960)
        
        # RNN处理
        rnn_output, _ = self.rnn(features)  # (batch, width, hidden_size * 2)
        
        # 分类
        output = self.classifier(rnn_output)  # (batch, width, num_classes)
        
        return output


def create_lcrnn(num_classes=100, **kwargs):
    """创建LCRNN模型的工厂函数"""
    return LCRNN(num_classes, **kwargs)


if __name__ == "__main__":
    # 测试LCRNN模型
    model = create_lcrnn(num_classes=100)
    
    # 创建测试输入
    x = torch.randn(2, 3, 64, 256)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {output.shape}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 计算模型大小（MB）
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size = (param_size + buffer_size) / 1024 / 1024
        print(f"模型大小: {model_size:.2f} MB")