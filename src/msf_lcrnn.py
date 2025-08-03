"""
MSF_LCRNN (Multi-Scale Feature LCRNN) 模型
多尺度特征融合的轻量级CRNN，适合处理化学符号的特殊字符
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取模块"""
    
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # 不同尺度的卷积分支
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1, padding=0),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1, padding=0),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1, padding=0),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 5, padding=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, 1, padding=0),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch_1x1(x)
        branch2 = self.branch_3x3(x)
        branch3 = self.branch_5x5(x)
        branch4 = self.branch_pool(x)
        
        # 连接所有分支
        output = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return output


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 深度卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        
        # 点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        return x


class AttentionGate(nn.Module):
    """注意力门控机制"""
    
    def __init__(self, channels):
        super(AttentionGate, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class MSF_LCRNN_Backbone(nn.Module):
    """多尺度特征融合的轻量级骨干网络"""
    
    def __init__(self):
        super(MSF_LCRNN_Backbone, self).__init__()
        
        # 第一层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 32x128
        )
        
        # 多尺度特征提取层
        self.msf1 = MultiScaleFeatureExtractor(32, 64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 16x64
        
        # 深度可分离卷积层
        self.dsc1 = DepthwiseSeparableConv(64, 128)
        self.pool2 = nn.MaxPool2d((2, 1), (2, 1))  # 8x64
        
        # 第二个多尺度特征提取
        self.msf2 = MultiScaleFeatureExtractor(128, 256)
        self.pool3 = nn.MaxPool2d((2, 1), (2, 1))  # 4x64
        
        # 注意力机制
        self.attention = AttentionGate(256)
        
        # 深度可分离卷积层
        self.dsc2 = DepthwiseSeparableConv(256, 512)
        self.pool4 = nn.MaxPool2d((2, 1), (2, 1))  # 2x64
        
        # 最终特征提取
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, 512, (2, 1), padding=0),  # 1x64
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 第一层卷积
        x = self.conv1(x)  # (batch, 32, 32, 128)
        
        # 多尺度特征提取
        x = self.msf1(x)   # (batch, 64, 32, 128)
        x = self.pool1(x)  # (batch, 64, 16, 64)
        
        # 深度可分离卷积
        x = self.dsc1(x)   # (batch, 128, 16, 64)
        x = self.pool2(x)  # (batch, 128, 8, 64)
        
        # 第二个多尺度特征提取
        x = self.msf2(x)   # (batch, 256, 8, 64)
        x = self.pool3(x)  # (batch, 256, 4, 64)
        
        # 注意力机制
        x = self.attention(x)  # (batch, 256, 4, 64)
        
        # 深度可分离卷积
        x = self.dsc2(x)   # (batch, 512, 4, 64)
        x = self.pool4(x)  # (batch, 512, 2, 64)
        
        # 最终卷积
        x = self.final_conv(x)  # (batch, 512, 1, 64)
        
        # 重塑为序列
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, width, channels)  # (batch, 64, 512)
        
        return x


class MSF_LCRNN(nn.Module):
    """多尺度特征融合的轻量级CRNN模型"""
    
    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.1):
        super(MSF_LCRNN, self).__init__()
        
        # 多尺度特征提取骨干
        self.backbone = MSF_LCRNN_Backbone()
        
        # 双向GRU（比LSTM更轻量）
        self.rnn = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 序列注意力机制
        self.seq_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 特征提取
        features = self.backbone(x)  # (batch, width, 512)
        
        # RNN处理
        rnn_output, _ = self.rnn(features)  # (batch, width, hidden_size * 2)
        
        # 应用dropout
        rnn_output = self.dropout(rnn_output)
        
        # 分类
        output = self.classifier(rnn_output)  # (batch, width, num_classes)
        
        return output


def create_msf_lcrnn(num_classes=100, **kwargs):
    """创建MSF_LCRNN模型的工厂函数"""
    return MSF_LCRNN(num_classes, **kwargs)


if __name__ == "__main__":
    # 测试MSF_LCRNN模型
    model = create_msf_lcrnn(num_classes=100)
    
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
        
        # 计算FLOPs（大致估算）
        input_size = x.numel() * x.element_size()
        print(f"输入大小: {input_size / 1024 / 1024:.2f} MB")