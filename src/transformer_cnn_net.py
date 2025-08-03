"""
基于Transformer和CNN的化学方程式OCR识别模型
结合CNN特征提取和Transformer序列建模的主力架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # 残差连接的输入
        residual = x
        
        # 计算Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 输出投影
        output = self.w_o(context)
        
        # 残差连接和层归一化
        output = self.layer_norm(residual + self.dropout(output))
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
    def forward(self, src):
        # 自注意力
        src2 = self.self_attn(src)
        
        # 前馈网络
        src3 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src2 = self.norm(src2 + self.dropout1(src3))
        
        return src2


class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器"""
    
    def __init__(self, input_channels=3):
        super(CNNFeatureExtractor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 第一组卷积
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二组卷积
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三组卷积
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            # 第四组卷积
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            # 第五组卷积
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        # 将特征图重塑为序列
        batch_size, channels, height, width = x.size()
        # 经过多次pooling后，height应该是3（64->32->16->8->4->2->1，但最后一个卷积核是2x1，所以实际是3）
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.view(batch_size, width, channels * height)  # (B, W, C*H)
        return x


class BiRNN(nn.Module):
    """双向RNN模块，支持LSTM和GRU"""
    
    def __init__(self, input_size, hidden_size, num_layers=2, rnn_type='lstm', dropout=0.1):
        super(BiRNN, self).__init__()
        self.rnn_type = rnn_type.lower()
        
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, bidirectional=True, dropout=dropout
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers,
                batch_first=True, bidirectional=True, dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
            
        self.output_size = hidden_size * 2  # 双向
    
    def forward(self, x):
        output, _ = self.rnn(x)
        return output


class SemanticEncoder(nn.Module):
    """语义编码器：将文本标签编码为语义特征"""
    
    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.1):
        super(SemanticEncoder, self).__init__()
        
        # 字符嵌入层
        self.embedding = nn.Embedding(num_classes, hidden_size)
        
        # 双向LSTM编码器
        self.lstm = nn.LSTM(
            hidden_size, hidden_size // 2, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_indices, text_lengths):
        # 嵌入文本
        embedded = self.embedding(text_indices)  # (B, L, hidden_size)
        
        # 处理空序列，将长度为0的序列长度设置为1，避免pack_padded_sequence错误
        text_lengths_clamped = torch.clamp(text_lengths, min=1)
        
        # 打包变长序列
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths_clamped.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM编码
        output, (hidden, cell) = self.lstm(packed)
        
        # 解包
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # 使用最后一个时间步的隐藏状态作为语义表示
        # hidden: (num_layers * 2, B, hidden_size // 2)
        # 拼接前向和后向的最后一层隐藏状态
        semantic_features = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, hidden_size)
        
        # 对于原始长度为0的序列，将其语义特征置零
        zero_mask = (text_lengths == 0).unsqueeze(1).float()
        semantic_features = semantic_features * (1 - zero_mask)
        
        # 输出投影
        semantic_features = self.output_proj(semantic_features)
        semantic_features = self.dropout(semantic_features)
        
        return semantic_features


class TransformerCNN(nn.Module):
    """
    TransformerCNN主力模型
    结合CNN特征提取和Transformer序列建模，包含语义指导模块
    """
    
    def __init__(self, num_classes, rnn_type='lstm', hidden_size=256, 
                 nhead=8, num_layers=6, dropout=0.1):
        super(TransformerCNN, self).__init__()
        
        # CNN特征提取器
        self.cnn = CNNFeatureExtractor()
        
        # 计算CNN输出的特征维度
        # CNN最后一层输出512通道，高度经过pooling后剩余3
        cnn_output_size = 512 * 3  # 512通道 * 3高度
        
        # 双向RNN
        self.birnn = BiRNN(cnn_output_size, hidden_size, rnn_type=rnn_type, dropout=dropout)
        
        # Transformer编码器
        d_model = self.birnn.output_size
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        # 语义指导模块（训练时使用）
        self.use_semantic_guide = True
        self.semantic_encoder = SemanticEncoder(num_classes, d_model, num_layers=2, dropout=dropout)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # 语义预测头（用于辅助损失）
        self.semantic_predictor = nn.Linear(d_model, num_classes)
        
    def forward(self, x, text_indices=None, text_lengths=None):
        # CNN特征提取
        cnn_features = self.cnn(x)  # (B, W, C*H)
        
        # 双向RNN处理
        rnn_output = self.birnn(cnn_features)  # (B, W, hidden_size*2)
        
        # 语义指导（仅训练时）
        if self.training and self.use_semantic_guide and text_indices is not None and text_lengths is not None:
            # 获取语义特征
            semantic_features = self.semantic_encoder(text_indices, text_lengths)  # (B, d_model)
            
            # 扩展语义特征到序列长度
            seq_len = rnn_output.size(1)
            semantic_features_expanded = semantic_features.unsqueeze(1).expand(-1, seq_len, -1)  # (B, W, d_model)
            
            # 特征融合：将语义特征与视觉特征拼接并融合
            combined_features = torch.cat([rnn_output, semantic_features_expanded], dim=2)  # (B, W, d_model*2)
            fused_features = self.feature_fusion(combined_features)  # (B, W, d_model)
            
            # 使用融合后的特征进行后续处理
            features_for_transformer = fused_features
        else:
            features_for_transformer = rnn_output
        
        # 添加位置编码
        features_for_transformer = features_for_transformer.transpose(0, 1)  # (W, B, d_model)
        encoded = self.pos_encoder(features_for_transformer)
        
        # Transformer编码器
        for layer in self.transformer_layers:
            encoded = layer(encoded)
        
        # 转回 (B, W, d_model)
        encoded = encoded.transpose(0, 1)
        
        # 应用dropout
        encoded = self.dropout(encoded)
        
        # 分类输出
        output = self.classifier(encoded)
        
        # 训练时返回额外的语义预测用于辅助损失
        if self.training and self.use_semantic_guide and text_indices is not None:
            # 使用融合特征预测语义
            semantic_pred = self.semantic_predictor(encoded)
            return output, semantic_pred
        
        return output
    
    def set_inference_mode(self):
        """设置为推理模式，移除语义指导"""
        self.use_semantic_guide = False
        self.eval()


def create_transformer_cnn(num_classes, rnn_type='lstm', **kwargs):
    """创建TransformerCNN模型的工厂函数"""
    return TransformerCNN(num_classes, rnn_type=rnn_type, **kwargs)


if __name__ == "__main__":
    # 测试模型
    model = create_transformer_cnn(num_classes=100, rnn_type='lstm')
    
    # 创建测试输入
    batch_size = 2
    channels = 3
    height = 64
    width = 256
    max_text_length = 25
    
    x = torch.randn(batch_size, channels, height, width)
    
    # 创建文本标签（用于测试语义指导）
    text_indices = torch.randint(0, 100, (batch_size, max_text_length))
    text_lengths = torch.tensor([20, 15])  # 实际文本长度
    
    print(f"输入形状: {x.shape}")
    print(f"文本标签形状: {text_indices.shape}")
    
    # 测试训练模式（带语义指导）
    model.train()
    print("\n训练模式测试:")
    with torch.no_grad():
        output = model(x, text_indices, text_lengths)
        if isinstance(output, tuple):
            main_output, semantic_output = output
            print(f"主输出形状: {main_output.shape}")
            print(f"语义输出形状: {semantic_output.shape}")
        else:
            print(f"输出形状: {output.shape}")
    
    # 测试推理模式（无语义指导）
    model.set_inference_mode()
    print("\n推理模式测试:")
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {output.shape}")
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")