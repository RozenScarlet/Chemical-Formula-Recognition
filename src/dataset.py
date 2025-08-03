"""
数据集处理模块
包含数据加载、预处理和CTC标签转换器
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import re


def chemical_tokenizer(text, character_set):
    """
    化学符号分词器 - 将化学方程式文本正确分割为符号序列
    保持复合符号的完整性，如 _2, |+, \~= 等
    """
    tokens = []
    i = 0
    
    # 将字符集按长度排序，优先匹配长符号
    sorted_chars = sorted(character_set, key=len, reverse=True)
    
    while i < len(text):
        matched = False
        
        # 尝试匹配最长的符号
        for char in sorted_chars:
            if text[i:i+len(char)] == char:
                tokens.append(char)
                i += len(char)
                matched = True
                break
        
        if not matched:
            # 如果没有匹配到预定义符号，跳过这个字符（或者添加到未知符号）
            print(f"警告: 未识别的字符序列 '{text[i]}' 在位置 {i}")
            i += 1
    
    return tokens


class CTCLabelConverter:
    """CTC标签转换器"""
    
    def __init__(self, character_set):
        # 添加特殊字符
        self.BLANK = '[blank]'  # CTC专用空白字符，必须是索引0
        self.SPACE = '[s]'
        self.UNKNOWN = '[UNK]'
        
        # 保存原始字符集用于分词
        self.original_character_set = character_set
        
        # 构建字符集 - BLANK必须是第一个（索引0）
        list_token = [self.BLANK, self.SPACE, self.UNKNOWN]
        list_character = list(character_set)
        
        self.character = list_token + list_character
        self.dict = {char: i for i, char in enumerate(self.character)}
        
        # 特殊字符索引
        self.SPECIAL_TOKENS = {
            'blank': self.dict[self.BLANK],    # 0
            'pad': self.dict[self.SPACE],      # 1  
            'unk': self.dict[self.UNKNOWN]     # 2
        }
        
    def encode(self, text, batch_max_length):
        """将文本编码为数字序列 - 使用化学符号分词器"""
        # 使用化学符号分词器将文本分割为符号序列
        tokens = chemical_tokenizer(text, self.original_character_set)
        
        text_list = []
        for token in tokens:
            if token in self.dict:
                text_list.append(self.dict[token])
            else:
                text_list.append(self.dict[self.UNKNOWN])
        
        # 填充到最大长度 - 注意：不使用BLANK（索引0）填充，使用SPACE
        if len(text_list) > batch_max_length:
            text_list = text_list[:batch_max_length]
        else:
            # 使用SPACE字符填充，而不是BLANK
            text_list = text_list + [self.dict[self.SPACE]] * (batch_max_length - len(text_list))
            
        return text_list
    
    def ctc_greedy_decode(self, indices):
        """统一的CTC贪心解码：去重 + 过滤特殊符号"""
        seq = []
        prev = -1
        
        # 处理indices，可能是tensor或list
        if hasattr(indices, 'cpu'):
            indices = indices.cpu().numpy() if hasattr(indices, 'numpy') else indices.cpu().tolist()
        
        for ch in indices:
            # 转换为int
            ch = int(ch.item()) if hasattr(ch, 'item') else int(ch)
            
            # CTC去重：跳过连续重复的字符
            if ch == prev:
                continue
            prev = ch
            
            # 过滤特殊字符：blank(0) / pad(1) / unk(2)
            if ch in self.SPECIAL_TOKENS.values():
                continue
                
            seq.append(ch)
        
        # 转换为文本
        return ''.join(self.character[i] for i in seq)
    
    def decode(self, text_index, length):
        """将数字序列解码为文本 - 使用统一的CTC解码"""
        texts = []
        
        # 如果是多批次的数据，处理每个批次
        if len(text_index.shape) > 1:
            for batch_idx in range(text_index.shape[0]):
                t = text_index[batch_idx][:length[batch_idx]]
                text = self.ctc_greedy_decode(t)
                texts.append(text)
        else:
            # 单序列处理
            index = 0
            for l in length:
                t = text_index[index:index + l]
                text = self.ctc_greedy_decode(t)
                texts.append(text)
                index += l
        return texts


class ChemicalDataset(Dataset):
    """化学方程式数据集"""
    
    def __init__(self, data_dir, mode='train', transform=None, max_length=25):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.max_length = max_length
        
        # 读取标签文件
        label_file = os.path.join(data_dir, 'labels.txt')
        self.samples = []
        
        if os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            # parts[0]已经包含了images/前缀
                            img_path = os.path.join(data_dir, parts[0])
                            if os.path.exists(img_path):
                                self.samples.append((img_path, parts[1]))
        
        # 构建正确的字符集（保持复合符号完整性）
        self.character_set = build_character_set(data_dir)
        
        # 创建标签转换器
        self.label_converter = CTCLabelConverter(self.character_set)
        
        # 随机划分数据，确保训练和验证集分布一致
        if len(self.samples) > 0:
            # 设置随机种子确保可重现
            random_state = 42
            
            # 计算标签长度用于分层采样
            labels_for_stratify = []
            for _, text in self.samples:
                # 根据文本长度分组，确保长短文本在训练/验证集中均衡分布
                text_length_group = min(len(text) // 5, 4)  # 分为5组：0-4, 5-9, 10-14, 15-19, 20+
                labels_for_stratify.append(text_length_group)
            
            # 使用分层采样进行训练/验证划分
            try:
                train_samples, val_samples = train_test_split(
                    self.samples, 
                    test_size=0.1,  # 10%作为验证集
                    random_state=random_state,
                    stratify=labels_for_stratify,  # 按文本长度分层
                    shuffle=True
                )
            except ValueError:
                # 如果分层采样失败（某些类别样本太少），使用普通随机划分
                print("分层采样失败，使用普通随机划分")
                train_samples, val_samples = train_test_split(
                    self.samples,
                    test_size=0.1,
                    random_state=random_state,
                    shuffle=True
                )
            
            # 根据模式选择数据
            if mode == 'train':
                self.samples = train_samples
            else:
                self.samples = val_samples
        
        print(f"{mode}模式: {len(self.samples)}个样本")
        print(f"字符集大小: {len(self.character_set)}")
        
        # 打印数据分布信息
        if len(self.samples) > 0:
            text_lengths = [len(text) for _, text in self.samples]
            print(f"文本长度分布: 最小={min(text_lengths)}, 最大={max(text_lengths)}, 平均={np.mean(text_lengths):.1f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, text = self.samples[index]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 如果图像加载失败，返回第一个样本
            return self.__getitem__(0)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 编码标签
        text_encoded = self.label_converter.encode(text, self.max_length)
        # 使用编码后文本的实际长度（去掉填充）
        actual_length = min(len(text), self.max_length)
        text_length = actual_length
        
        # 返回图像、编码的文本、文本长度和原始文本
        return image, torch.LongTensor(text_encoded), torch.LongTensor([text_length]), text
    
    def get_original_image(self, index):
        """获取原始图像（未经变换）"""
        img_path, _ = self.samples[index]
        try:
            image = Image.open(img_path).convert('RGB')
            return image
        except:
            # 如果加载失败，返回第一个图像
            return self.get_original_image(0)


class Resize(object):
    """调整图像大小"""
    
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img):
        return img.resize(self.size, self.interpolation)


class NormalizePAD(object):
    """标准化并填充图像"""
    
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = max_size[2] // 2
        self.PAD_type = PAD_type
    
    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        
        return Pad_img


class DataAugmentation:
    """加强的数据增强"""
    
    def __init__(self, prob=0.7, strong=True):
        self.prob = prob
        self.strong = strong  # 是否使用强数据增强
    
    def __call__(self, image):
        if random.random() > self.prob:
            return image
        
        # 根据强度选择增强方式
        if self.strong:
            # 强数据增强：更多变换，更强的变换强度
            aug_types = ['brightness', 'contrast', 'sharpness', 'blur', 'noise', 'rotation', 'perspective', 'elastic', 'shadow', 'cutout']
            # 随机应用1-3个增强
            num_augs = random.randint(1, 3)
            for _ in range(num_augs):
                aug_type = random.choice(aug_types)
                image = self._apply_augmentation(image, aug_type, strong=True)
        else:
            # 轻度数据增强
            aug_types = ['brightness', 'contrast', 'sharpness', 'blur', 'noise']
            aug_type = random.choice(aug_types)
            image = self._apply_augmentation(image, aug_type, strong=False)
        
        return image
    
    def _apply_augmentation(self, image, aug_type, strong=False):
        """应用具体的增强方法"""
        if aug_type == 'brightness':
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.6, 1.4) if strong else random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        elif aug_type == 'contrast':
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.6, 1.4) if strong else random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        elif aug_type == 'sharpness':
            enhancer = ImageEnhance.Sharpness(image)
            factor = random.uniform(0.5, 1.5) if strong else random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        elif aug_type == 'blur':
            # 转换为OpenCV格式进行模糊
            img_array = np.array(image)
            sigma = random.uniform(0.5, 2.5) if strong else random.uniform(0.5, 1.5)
            kernel_size = random.choice([3, 5]) if strong else 3
            img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
            image = Image.fromarray(img_array)
        elif aug_type == 'noise':
            # 添加高斯噪声
            img_array = np.array(image).astype(np.float32)
            noise_level = random.uniform(0.01, 0.08) if strong else random.uniform(0.01, 0.03)
            noise = np.random.normal(0, noise_level, img_array.shape) * 255
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        elif aug_type == 'rotation' and strong:
            # 轻微旋转（仅强增强模式）
            angle = random.uniform(-3, 3)  # 增加旋转角度
            image = image.rotate(angle, expand=False, fillcolor=(255, 255, 255))
        elif aug_type == 'perspective' and strong:
            # 轻微透视变换（仅强增强模式）
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            # 创建轻微的透视变换
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            offset = min(w, h) * 0.03  # 增加偏移量到3%
            pts2 = np.float32([
                [random.uniform(-offset, offset), random.uniform(-offset, offset)],
                [w + random.uniform(-offset, offset), random.uniform(-offset, offset)],
                [random.uniform(-offset, offset), h + random.uniform(-offset, offset)],
                [w + random.uniform(-offset, offset), h + random.uniform(-offset, offset)]
            ])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            img_array = cv2.warpPerspective(img_array, matrix, (w, h), borderValue=(255, 255, 255))
            image = Image.fromarray(img_array)
        elif aug_type == 'elastic' and strong:
            # 弹性变换（仅强增强模式）
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            # 简单的弹性变换：随机位移
            dx = np.random.uniform(-0.02, 0.02, (h, w)) * w
            dy = np.random.uniform(-0.02, 0.02, (h, w)) * h
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x = (x + dx).astype(np.float32)
            y = (y + dy).astype(np.float32)
            img_array = cv2.remap(img_array, x, y, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            image = Image.fromarray(img_array)
        elif aug_type == 'shadow' and strong:
            # 添加阴影效果（仅强增强模式）
            img_array = np.array(image).astype(np.float32)
            # 创建随机阴影区域
            h, w = img_array.shape[:2]
            shadow_factor = random.uniform(0.3, 0.7)
            # 随机选择阴影区域
            shadow_type = random.choice(['corner', 'stripe', 'circle'])
            if shadow_type == 'corner':
                # 角落阴影
                corner = random.choice(['tl', 'tr', 'bl', 'br'])
                shadow_size = random.uniform(0.2, 0.4)
                mask = np.ones((h, w), dtype=np.float32)
                if corner == 'tl':
                    mask[:int(h*shadow_size), :int(w*shadow_size)] = shadow_factor
                elif corner == 'tr':
                    mask[:int(h*shadow_size), int(w*(1-shadow_size)):] = shadow_factor
                elif corner == 'bl':
                    mask[int(h*(1-shadow_size)):, :int(w*shadow_size)] = shadow_factor
                elif corner == 'br':
                    mask[int(h*(1-shadow_size)):, int(w*(1-shadow_size)):] = shadow_factor
            elif shadow_type == 'stripe':
                # 条纹阴影
                stripe_width = random.uniform(0.1, 0.3)
                start_pos = random.uniform(0, 1-stripe_width)
                mask = np.ones((h, w), dtype=np.float32)
                if random.choice([True, False]):  # 垂直条纹
                    mask[:, int(w*start_pos):int(w*(start_pos+stripe_width))] = shadow_factor
                else:  # 水平条纹
                    mask[int(h*start_pos):int(h*(start_pos+stripe_width)), :] = shadow_factor
            else:  # circle
                # 圆形阴影
                center_x = random.uniform(0.2, 0.8) * w
                center_y = random.uniform(0.2, 0.8) * h
                radius = random.uniform(0.1, 0.3) * min(w, h)
                y, x = np.ogrid[:h, :w]
                mask = np.ones((h, w), dtype=np.float32)
                circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                mask[circle_mask] = shadow_factor
            
            # 应用阴影
            for c in range(img_array.shape[2]):  # RGB三个通道
                img_array[:, :, c] *= mask
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        elif aug_type == 'cutout' and strong:
            # Cutout增强（仅强增强模式）
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            # 随机选择1-3个切除区域
            num_holes = random.randint(1, 3)
            for _ in range(num_holes):
                hole_size = random.uniform(0.05, 0.15)  # 孔洞大小占图像的5%-15%
                hole_w = int(w * hole_size)
                hole_h = int(h * hole_size)
                # 随机位置
                x = random.randint(0, max(1, w - hole_w))
                y = random.randint(0, max(1, h - hole_h))
                # 用白色填充（化学公式背景通常是白色）
                img_array[y:y+hole_h, x:x+hole_w] = 255
            image = Image.fromarray(img_array)
        
        return image


def create_dataset(data_dir, img_height=64, img_width=256, batch_size=32, num_workers=4):
    """创建数据加载器"""
    
    # 训练数据变换（使用强数据增强）
    train_transform = transforms.Compose([
        DataAugmentation(prob=0.8, strong=True),  # 进一步增加增强概率到80%
        Resize((img_width, img_height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证数据变换（不使用数据增强，确保结果可重复）
    val_transform = transforms.Compose([
        Resize((img_width, img_height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = ChemicalDataset(data_dir, mode='train', transform=train_transform)
    val_dataset = ChemicalDataset(data_dir, mode='val', transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, train_dataset.label_converter


def collate_fn(batch):
    """自定义批处理函数"""
    images, texts, text_lengths, raw_texts = zip(*batch)
    
    images = torch.stack(images)
    texts = torch.stack(texts)
    text_lengths = torch.cat(text_lengths)
    
    return images, texts, text_lengths, raw_texts


def build_character_set(data_dir):
    """构建字符集 - 使用预定义的化学符号，保持复合符号完整性"""
    # 从预定义classes.txt中加载实际使用的46个字符
    classes_file = os.path.join(os.path.dirname(data_dir), '数据集', 'classes.txt')
    if not os.path.exists(classes_file):
        # 如果找不到classes.txt，使用备用路径
        classes_file = os.path.join(data_dir, '..', '数据集', 'classes.txt')
    
    predefined_chars = []
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    predefined_chars.append(line)
    
    # 验证哪些字符在实际数据中被使用
    label_file = os.path.join(data_dir, 'labels.txt')
    labels_content = ""
    if os.path.exists(label_file):
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        labels_content += parts[1] + " "
    
    # 过滤出实际使用的字符（保持复合符号完整性）
    used_chars = []
    for char in predefined_chars:
        if char in labels_content:
            used_chars.append(char)
    
    print(f"从预定义的{len(predefined_chars)}个字符中，实际使用了{len(used_chars)}个字符")
    print(f"使用的字符: {used_chars}")
    
    return used_chars


if __name__ == "__main__":
    # 测试数据集
    data_dir = "dataset"
    
    if os.path.exists(data_dir):
        # 构建字符集
        chars = build_character_set(data_dir)
        print(f"字符集: {chars}")
        print(f"字符数量: {len(chars)}")
        
        # 创建数据加载器
        train_loader, val_loader, label_converter = create_dataset(data_dir, batch_size=4)
        
        # 测试一个批次
        for batch_idx, (images, texts, text_lengths) in enumerate(train_loader):
            print(f"图像形状: {images.shape}")
            print(f"文本形状: {texts.shape}")
            print(f"文本长度: {text_lengths}")
            
            # 解码文本
            decoded_texts = label_converter.decode(texts.view(-1), text_lengths)
            print(f"解码文本: {decoded_texts}")
            break
    else:
        print(f"数据目录 {data_dir} 不存在")