"""
CRNN (Convolutional Recurrent Neural Network) 模型
经典的OCR识别架构，结合CNN和RNN
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """CRNN模型实现"""
    
    def __init__(self, img_height, img_width, num_classes, num_channels=3, hidden_size=256, num_layers=2):
        super(CRNN, self).__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        
        # CNN特征提取部分
        self.cnn = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x32x128
            
            # 第二层卷积块
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x16x64
            
            # 第三层卷积块
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 256x8x64
            
            # 第四层卷积块
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 512x4x64
            
            # 第五层卷积块
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),  # 512x3x63
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # RNN部分
        self.rnn = nn.LSTM(
            input_size=512 * 3,  # 根据CNN输出计算
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # CNN特征提取
        conv_features = self.cnn(x)  # (batch, 512, height, width)
        
        # 重塑为序列形式
        batch_size, channels, height, width = conv_features.size()
        conv_features = conv_features.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        conv_features = conv_features.view(batch_size, width, channels * height)
        
        # RNN处理
        rnn_output, _ = self.rnn(conv_features)
        
        # 分类
        output = self.classifier(rnn_output)
        
        return output


class BidirectionalLSTM(nn.Module):
    """双向LSTM模块"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, input_tensor):
        recurrent, _ = self.rnn(input_tensor)
        output = self.linear(recurrent)
        return output


def create_crnn(img_height=64, img_width=256, num_classes=100, **kwargs):
    """创建CRNN模型的工厂函数"""
    return CRNN(img_height, img_width, num_classes, **kwargs)


if __name__ == "__main__":
    # 测试CRNN模型
    model = create_crnn(img_height=64, img_width=256, num_classes=100)
    
    # 创建测试输入
    x = torch.randn(2, 3, 64, 256)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {output.shape}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")