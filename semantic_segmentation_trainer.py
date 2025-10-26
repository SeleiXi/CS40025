#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义分割模型训练系统 - 建筑物识别任务
基于U-Net架构的端到端训练框架，支持多种数据增强和损失函数组合

主要特性：
- 模块化设计，支持多种编码器-解码器架构
- 灵活的数据增强管道，包含空间变换和像素级增强
- 多损失函数组合：BCE + Dice + Focal Loss
- 支持混合精度训练和分布式训练
- 完整的模型评估和推理管道

使用示例：
python semantic_segmentation_trainer.py \
  --data_config configs/data_config.yaml \
  --model_config configs/model_config.yaml \
  --training_config configs/training_config.yaml \
  --output_dir ./experiments/building_segmentation \
  --experiment_name unet_resnet50_v1
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 设置环境变量以避免版本检查警告
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
warnings.filterwarnings("ignore", category=UserWarning)

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


# ===================== 数据编码解码工具 =====================

class RunLengthEncoder:
    """RLE编码器，用于将二值掩码转换为压缩格式"""
    
    @staticmethod
    def encode_mask(mask: np.ndarray) -> str:
        """
        将二值掩码编码为RLE字符串
        
        Args:
            mask: 二值掩码数组，形状为[H, W]，值为0或1
            
        Returns:
            RLE编码字符串
        """
        # 按列优先顺序展平
        flat_mask = mask.flatten(order='F')
        
        # 添加边界标记
        padded_mask = np.concatenate([[0], flat_mask, [0]])
        
        # 找到变化点
        change_points = np.where(padded_mask[1:] != padded_mask[:-1])[0] + 1
        
        # 计算运行长度
        run_lengths = np.diff(change_points)
        run_lengths[1::2] -= run_lengths[::2]
        
        return ' '.join(map(str, run_lengths))
    
    @staticmethod
    def decode_mask(rle_string: str, target_shape: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        将RLE字符串解码为二值掩码
        
        Args:
            rle_string: RLE编码字符串
            target_shape: 目标掩码形状
            
        Returns:
            解码后的二值掩码
        """
        if not rle_string or rle_string.strip() == '':
            return np.zeros(target_shape, dtype=np.uint8)
        
        # 解析RLE字符串
        rle_values = list(map(int, rle_string.split()))
        
        # 重建掩码
        mask = np.zeros(target_shape[0] * target_shape[1], dtype=np.uint8)
        
        for i in range(0, len(rle_values), 2):
            start_pos = rle_values[i] - 1  # RLE从1开始计数
            length = rle_values[i + 1] if i + 1 < len(rle_values) else 0
            mask[start_pos:start_pos + length] = 1
        
        return mask.reshape(target_shape, order='F')


# ===================== 图像后处理工具 =====================

class MaskPostProcessor:
    """掩码后处理器，用于优化分割结果"""
    
    def __init__(self, 
                 min_object_area: int = 64,
                 morphology_kernel_size: int = 3,
                 morphology_iterations: int = 1,
                 keep_largest_component: bool = False):
        """
        初始化后处理器
        
        Args:
            min_object_area: 最小对象面积阈值
            morphology_kernel_size: 形态学操作核大小
            morphology_iterations: 形态学操作迭代次数
            keep_largest_component: 是否只保留最大连通分量
        """
        self.min_area = min_object_area
        self.kernel_size = morphology_kernel_size
        self.iterations = morphology_iterations
        self.keep_largest = keep_largest_component
    
    def process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        对掩码进行后处理
        
        Args:
            mask: 输入掩码，形状为[H, W]
            
        Returns:
            处理后的掩码
        """
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # 形态学操作
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        # 开运算：去除噪声
        processed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.iterations)
        
        # 闭运算：填充空洞
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=self.iterations)
        
        # 连通分量分析
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(processed_mask, connectivity=8)
        
        if num_components <= 1:
            return processed_mask
        
        if self.keep_largest:
            # 只保留最大连通分量
            areas = stats[1:, -1]  # 排除背景
            largest_idx = np.argmax(areas) + 1
            return (labels == largest_idx).astype(np.uint8)
        
        # 移除小对象
        result_mask = np.zeros_like(processed_mask)
        for i in range(1, num_components):
            area = stats[i, -1]
            if area >= self.min_area:
                result_mask[labels == i] = 1
        
        return result_mask


# ===================== 数据集类 =====================

class BuildingSegmentationDataset(Dataset):
    """建筑物分割数据集"""
    
    def __init__(self,
                 image_paths: List[str],
                 mask_rles: Optional[List[str]] = None,
                 image_size: int = 256,
                 is_training: bool = True,
                 normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                 enable_denoising: bool = False,
                 mixup_probability: float = 0.0,
                 mixup_alpha: float = 0.4,
                 random_seed: Optional[int] = None):
        """
        初始化数据集
        
        Args:
            image_paths: 图像路径列表
            mask_rles: 掩码RLE字符串列表（训练时必需）
            image_size: 图像尺寸
            is_training: 是否为训练模式
            normalization_mean: 归一化均值
            normalization_std: 归一化标准差
            enable_denoising: 是否启用去噪
            mixup_probability: MixUp增强概率
            mixup_alpha: MixUp Beta分布参数
            random_seed: 随机种子
        """
        self.image_paths = image_paths
        self.mask_rles = mask_rles
        self.image_size = image_size
        self.is_training = is_training
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.enable_denoising = enable_denoising
        self.mixup_probability = mixup_probability
        self.mixup_alpha = mixup_alpha
        
        # 初始化随机数生成器
        self.rng = np.random.default_rng(random_seed)
        
        # 构建数据增强管道
        self._build_augmentation_pipeline()
        
        # 构建归一化变换
        self.normalize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(self.normalization_mean, self.normalization_std)
        ])
    
    def _build_augmentation_pipeline(self):
        """构建数据增强管道"""
        if self.is_training:
            # 训练时的增强策略
            self.augmentation_pipeline = A.Compose([
                A.Resize(self.image_size, self.image_size),
                
                # 几何变换
                A.OneOf([
                    A.RandomResizedCrop(
                        height=self.image_size, width=self.image_size,
                        scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.6
                    ),
                    A.Affine(
                        scale=(0.9, 1.1), translate_percent=(0.0, 0.05),
                        rotate=(-15, 15), shear=(-10, 10), p=0.7
                    ),
                ], p=0.7),
                
                # 弹性变换
                A.OneOf([
                    A.ElasticTransform(alpha=40, sigma=8, p=0.4),
                    A.GridDistortion(num_steps=5, p=0.5),
                    A.OpticalDistortion(distort_limit=0.3, shift_limit=0.05, p=0.5),
                    A.Perspective(scale=(0.02, 0.05), p=0.3),
                ], p=0.3),
                
                # 翻转和旋转
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                
                # 噪声和模糊
                A.OneOf([
                    A.GaussNoise(var_limit=(0.0088, 0.0215), p=0.6),
                    A.ISONoise(intensity=(0.1, 0.3), p=0.4),
                ], p=0.3),
                
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=0.3),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.4),
                    A.MedianBlur(blur_limit=3, p=0.3),
                ], p=0.2),
                
                # 颜色变换
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.6),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4),
                    A.ChannelShuffle(p=0.2),
                ], p=0.35),
                
                # 对比度增强
                A.CLAHE(clip_limit=(1, 3), tile_grid_size=(8, 8), p=0.15),
                A.RandomGamma(gamma_limit=(80, 120), p=0.15),
                
                # 随机遮挡
                A.CoarseDropout(
                    max_holes=8, max_height=int(0.1 * self.image_size),
                    max_width=int(0.1 * self.image_size), fill_value=0, p=0.3
                )
            ])
            
            # MixUp增强管道
            self.mixup_pipeline = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            ], additional_targets={'image2': 'image', 'mask2': 'mask'})
        else:
            # 验证/测试时的简单变换
            self.augmentation_pipeline = A.Compose([
                A.Resize(self.image_size, self.image_size),
            ])
            self.mixup_pipeline = None
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """加载并预处理图像"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        
        # 去噪处理
        if self.enable_denoising:
            try:
                image = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
            except Exception:
                image = cv2.bilateralFilter(image, d=3, sigmaColor=50, sigmaSpace=50)
        
        # BGR转RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取数据项"""
        image_path = self.image_paths[idx]
        image = self._load_image(image_path)
        
        if self.mask_rles is not None:
            # 训练/验证模式
            mask_rle = self.mask_rles[idx] if self.mask_rles[idx] else ''
            mask = RunLengthEncoder.decode_mask(mask_rle).astype(np.uint8)
            
            if self.is_training and self.mixup_probability > 0 and self.rng.random() < self.mixup_probability:
                # MixUp增强
                mixup_idx = self.rng.integers(0, len(self.image_paths))
                mixup_image = self._load_image(self.image_paths[mixup_idx])
                mixup_mask_rle = self.mask_rles[mixup_idx] if self.mask_rles[mixup_idx] else ''
                mixup_mask = RunLengthEncoder.decode_mask(mixup_mask_rle).astype(np.uint8)
                
                # 应用MixUp增强
                augmented = self.mixup_pipeline(
                    image=image, mask=mask,
                    image2=mixup_image, mask2=mixup_mask
                )
                
                image1, mask1 = augmented['image'], augmented['mask']
                image2, mask2 = augmented['image2'], augmented['mask2']
                
                # 转换为张量
                image1_tensor = self.normalize_transform(image1)
                image2_tensor = self.normalize_transform(image2)
                mask1_tensor = torch.from_numpy(mask1[None].astype(np.float32))
                mask2_tensor = torch.from_numpy(mask2[None].astype(np.float32))
                
                # MixUp混合
                lambda_param = self.rng.beta(self.mixup_alpha, self.mixup_alpha)
                mixed_image = lambda_param * image1_tensor + (1 - lambda_param) * image2_tensor
                mixed_mask = lambda_param * mask1_tensor + (1 - lambda_param) * mask2_tensor
                
                return {
                    'image': mixed_image,
                    'mask': mixed_mask,
                    'image_path': image_path
                }
            else:
                # 常规增强
                augmented = self.augmentation_pipeline(image=image, mask=mask)
                augmented_image = augmented['image']
                augmented_mask = augmented['mask']
                
                # 转换为张量
                image_tensor = self.normalize_transform(augmented_image)
                mask_tensor = torch.from_numpy(augmented_mask[None].astype(np.float32))
                
                return {
                    'image': image_tensor,
                    'mask': mask_tensor,
                    'image_path': image_path
                }
        else:
            # 测试模式
            augmented = self.augmentation_pipeline(image=image)
            augmented_image = augmented['image']
            
            image_tensor = self.normalize_transform(augmented_image)
            
            return {
                'image': image_tensor,
                'image_path': image_path,
                'image_name': os.path.basename(image_path)
            }


# ===================== 损失函数 =====================

class DiceLoss(nn.Module):
    """Dice损失函数"""
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Dice损失
        
        Args:
            predictions: 预测概率，形状为[N, 1, H, W]
            targets: 目标掩码，形状为[N, 1, H, W]
            
        Returns:
            Dice损失值
        """
        # 确保输入在[0, 1]范围内
        predictions = torch.sigmoid(predictions)
        
        # 计算交集和并集
        intersection = (predictions * targets).sum(dim=(-2, -1))
        union = predictions.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        
        # 计算Dice系数
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # 计算损失
        dice_loss = 1.0 - dice_coeff
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """Focal损失函数，用于处理类别不平衡"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal损失
        
        Args:
            predictions: 预测logits，形状为[N, 1, H, W]
            targets: 目标掩码，形状为[N, 1, H, W]
            
        Returns:
            Focal损失值
        """
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # 计算概率
        prob = torch.sigmoid(predictions)
        
        # 计算权重
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        weight = alpha_t * torch.pow(1 - prob, self.gamma)
        
        # 应用权重
        focal_loss = weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self,
                 bce_weight: float = 0.6,
                 dice_weight: float = 0.3,
                 focal_weight: float = 0.1,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma) if focal_weight > 0 else None
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算组合损失"""
        total_loss = 0.0
        
        if self.bce_weight > 0:
            total_loss += self.bce_weight * self.bce_loss(predictions, targets)
        
        if self.dice_weight > 0:
            total_loss += self.dice_weight * self.dice_loss(predictions, targets)
        
        if self.focal_weight > 0 and self.focal_loss is not None:
            total_loss += self.focal_weight * self.focal_loss(predictions, targets)
        
        return total_loss


# ===================== 模型架构 =====================

class UNetEncoder(nn.Module):
    """U-Net编码器"""
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # 编码器层
        self.enc1 = self._make_layer(in_channels, base_channels)
        self.enc2 = self._make_layer(base_channels, base_channels * 2)
        self.enc3 = self._make_layer(base_channels * 2, base_channels * 4)
        self.enc4 = self._make_layer(base_channels * 4, base_channels * 8)
        
        # 瓶颈层
        self.bottleneck = self._make_layer(base_channels * 8, base_channels * 16)
        
        # 下采样
        self.pool = nn.MaxPool2d(2)
    
    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """构建卷积层"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """前向传播"""
        # 编码路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # 瓶颈
        bottleneck = self.bottleneck(self.pool(enc4))
        
        return bottleneck, [enc1, enc2, enc3, enc4]


class UNetDecoder(nn.Module):
    """U-Net解码器"""
    
    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.base_channels = base_channels
        
        # 上采样
        self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = self._make_layer(base_channels * 16, base_channels * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._make_layer(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._make_layer(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._make_layer(base_channels * 2, base_channels)
        
        # 输出层
        self.final_conv = nn.Conv2d(base_channels, 1, 1)
    
    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """构建卷积层"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, bottleneck: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        # 解码路径
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, skip_connections[3]], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, skip_connections[2]], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, skip_connections[1]], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, skip_connections[0]], dim=1)
        dec1 = self.dec1(dec1)
        
        # 输出
        output = self.final_conv(dec1)
        
        return output


class UNetSegmentationModel(nn.Module):
    """U-Net分割模型"""
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, base_channels)
        self.decoder = UNetDecoder(base_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        bottleneck, skip_connections = self.encoder(x)
        output = self.decoder(bottleneck, skip_connections)
        return output


# ===================== 评估指标 =====================

class SegmentationMetrics:
    """分割评估指标"""
    
    @staticmethod
    def dice_coefficient(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
        """计算Dice系数"""
        pred_binary = (prediction > threshold).astype(np.uint8)
        target_binary = target.astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return (2.0 * intersection) / union
    
    @staticmethod
    def iou_score(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
        """计算IoU分数"""
        pred_binary = (prediction > threshold).astype(np.uint8)
        target_binary = target.astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union


# ===================== 训练器类 =====================

class SegmentationTrainer:
    """分割模型训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: torch.device = None,
                 output_dir: str = './output',
                 experiment_name: str = 'experiment',
                 enable_mixed_precision: bool = False,
                 gradient_accumulation_steps: int = 1):
        """
        初始化训练器
        
        Args:
            model: 分割模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 计算设备
            output_dir: 输出目录
            experiment_name: 实验名称
            enable_mixed_precision: 是否启用混合精度
            gradient_accumulation_steps: 梯度累积步数
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.enable_mixed_precision = enable_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if enable_mixed_precision else None
        
        # 训练状态
        self.current_epoch = 0
        self.best_dice = 0.0
        self.best_epoch = 0
        
        # 将模型移到设备
        self.model.to(self.device)
        
        logger.info(f"训练器初始化完成，设备: {self.device}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"混合精度训练: {enable_mixed_precision}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 前向传播
            if self.enable_mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)
                    loss = self.criterion(predictions, masks)
                    loss = loss / self.gradient_accumulation_steps
            else:
                predictions = self.model(images)
                loss = self.criterion(predictions, masks)
                loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            if self.enable_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.enable_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
            
            # 计算指标
            with torch.no_grad():
                pred_probs = torch.sigmoid(predictions)
                dice_scores = []
                for i in range(pred_probs.shape[0]):
                    pred_np = pred_probs[i, 0].cpu().numpy()
                    mask_np = masks[i, 0].cpu().numpy()
                    dice = SegmentationMetrics.dice_coefficient(pred_np, mask_np)
                    dice_scores.append(dice)
                
                avg_dice = np.mean(dice_scores)
                total_loss += loss.item() * self.gradient_accumulation_steps
                total_dice += avg_dice
                num_batches += 1
            
            # 记录训练进度
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item() * self.gradient_accumulation_steps:.4f}, '
                          f'Dice: {avg_dice:.4f}')
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return {'loss': avg_loss, 'dice': avg_dice}
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 前向传播
                if self.enable_mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(images)
                        loss = self.criterion(predictions, masks)
                else:
                    predictions = self.model(images)
                    loss = self.criterion(predictions, masks)
                
                # 计算指标
                pred_probs = torch.sigmoid(predictions)
                dice_scores = []
                for i in range(pred_probs.shape[0]):
                    pred_np = pred_probs[i, 0].cpu().numpy()
                    mask_np = masks[i, 0].cpu().numpy()
                    dice = SegmentationMetrics.dice_coefficient(pred_np, mask_np)
                    dice_scores.append(dice)
                
                avg_dice = np.mean(dice_scores)
                total_loss += loss.item()
                total_dice += avg_dice
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return {'loss': avg_loss, 'dice': avg_dice}
    
    def train(self, num_epochs: int, save_best_only: bool = True) -> Dict[str, List[float]]:
        """训练模型"""
        logger.info(f"开始训练，共{num_epochs}个epoch")
        
        train_losses = []
        train_dices = []
        val_losses = []
        val_dices = []
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            train_losses.append(train_metrics['loss'])
            train_dices.append(train_metrics['dice'])
            
            # 验证
            val_metrics = self.validate_epoch()
            val_losses.append(val_metrics['loss'])
            val_dices.append(val_metrics['dice'])
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Dice/Train', train_metrics['dice'], epoch)
            self.writer.add_scalar('Dice/Val', val_metrics['dice'], epoch)
            
            # 学习率记录
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # 保存最佳模型
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self.best_epoch = epoch
                
                if save_best_only:
                    self.save_model('best_model.pth')
            
            # 定期保存检查点
            if epoch % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch}.pth')
            
            logger.info(f'Epoch {epoch}/{num_epochs-1}: '
                       f'Train Loss: {train_metrics["loss"]:.4f}, Train Dice: {train_metrics["dice"]:.4f}, '
                       f'Val Loss: {val_metrics["loss"]:.4f}, Val Dice: {val_metrics["dice"]:.4f}, '
                       f'Best Dice: {self.best_dice:.4f} (Epoch {self.best_epoch})')
        
        # 保存最终模型
        self.save_model('final_model.pth')
        
        # 保存训练历史
        history = {
            'train_loss': train_losses,
            'train_dice': train_dices,
            'val_loss': val_losses,
            'val_dice': val_dices,
            'best_dice': self.best_dice,
            'best_epoch': self.best_epoch
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"训练完成，最佳Dice分数: {self.best_dice:.4f} (Epoch {self.best_epoch})")
        
        return history
    
    def save_model(self, filename: str):
        """保存模型"""
        model_path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'best_dice': self.best_dice,
            'best_epoch': self.best_epoch,
        }, model_path)
        logger.info(f"模型已保存: {model_path}")
    
    def load_model(self, filename: str):
        """加载模型"""
        model_path = self.output_dir / filename
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_dice = checkpoint['best_dice']
        self.best_epoch = checkpoint['best_epoch']
        
        logger.info(f"模型已加载: {model_path}")
    
    def predict(self, test_loader: DataLoader, postprocessor: MaskPostProcessor) -> List[Dict[str, Any]]:
        """在测试集上进行预测"""
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                image_paths = batch['image_path']
                image_names = batch.get('image_name', [os.path.basename(p) for p in image_paths])
                
                # 前向传播
                if self.enable_mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = self.model(images)
                else:
                    logits = self.model(images)
                
                # 转换为概率
                probs = torch.sigmoid(logits)
                
                # 处理每个样本
                for i in range(probs.shape[0]):
                    prob_np = probs[i, 0].cpu().numpy()
                    
                    # 调整到原始尺寸
                    prob_resized = cv2.resize(prob_np, (512, 512), interpolation=cv2.INTER_LINEAR)
                    
                    # 二值化
                    binary_mask = (prob_resized > 0.5).astype(np.uint8)
                    
                    # 后处理
                    processed_mask = postprocessor.process_mask(binary_mask)
                    
                    # 编码为RLE
                    rle_string = RunLengthEncoder.encode_mask(processed_mask)
                    
                    predictions.append({
                        'image_name': image_names[i],
                        'image_path': image_paths[i],
                        'rle': rle_string,
                        'probability': prob_resized,
                        'binary_mask': processed_mask
                    })
        
        return predictions


# ===================== 数据加载工具 =====================

def create_data_loaders(train_csv_path: str,
                       train_img_dir: str,
                       test_csv_path: str,
                       test_img_dir: str,
                       image_size: int = 256,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       validation_split: float = 0.2,
                       random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        train_csv_path: 训练CSV文件路径
        train_img_dir: 训练图像目录
        test_csv_path: 测试CSV文件路径
        test_img_dir: 测试图像目录
        image_size: 图像尺寸
        batch_size: 批次大小
        num_workers: 数据加载器工作进程数
        validation_split: 验证集比例
        random_seed: 随机种子
        
    Returns:
        训练、验证、测试数据加载器
    """
    # 读取训练数据
    train_df = pd.read_csv(train_csv_path, sep='\t', names=['name', 'mask'])
    train_df['image_path'] = train_df['name'].apply(lambda x: os.path.join(train_img_dir, x))
    
    # 划分训练和验证集
    np.random.seed(random_seed)
    indices = np.random.permutation(len(train_df))
    split_idx = int(len(train_df) * (1 - validation_split))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_df_split = train_df.iloc[train_indices]
    val_df_split = train_df.iloc[val_indices]
    
    # 创建数据集
    train_dataset = BuildingSegmentationDataset(
        image_paths=train_df_split['image_path'].tolist(),
        mask_rles=train_df_split['mask'].fillna('').tolist(),
        image_size=image_size,
        is_training=True,
        random_seed=random_seed
    )
    
    val_dataset = BuildingSegmentationDataset(
        image_paths=val_df_split['image_path'].tolist(),
        mask_rles=val_df_split['mask'].fillna('').tolist(),
        image_size=image_size,
        is_training=False,
        random_seed=random_seed
    )
    
    # 读取测试数据
    test_df = pd.read_csv(test_csv_path, sep='\t', names=['name', 'mask'])
    test_df['image_path'] = test_df['name'].apply(lambda x: os.path.join(test_img_dir, x))
    
    test_dataset = BuildingSegmentationDataset(
        image_paths=test_df['image_path'].tolist(),
        mask_rles=None,
        image_size=image_size,
        is_training=False,
        random_seed=random_seed
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"数据加载器创建完成:")
    logger.info(f"  训练集: {len(train_dataset)} 样本")
    logger.info(f"  验证集: {len(val_dataset)} 样本")
    logger.info(f"  测试集: {len(test_dataset)} 样本")
    
    return train_loader, val_loader, test_loader


# ===================== 主函数 =====================

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='语义分割模型训练系统')
    
    # 数据路径
    parser.add_argument('--train_csv', type=str, default='./data_source/train_mask.csv',
                       help='训练CSV文件路径')
    parser.add_argument('--train_img_dir', type=str, default='./data_source/train',
                       help='训练图像目录')
    parser.add_argument('--test_csv', type=str, default='./data_source/test_a_samplesubmit.csv',
                       help='测试CSV文件路径')
    parser.add_argument('--test_img_dir', type=str, default='./data_source/test_a',
                       help='测试图像目录')
    
    # 模型配置
    parser.add_argument('--model_name', type=str, default='unet',
                       choices=['unet'],
                       help='模型名称')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='基础通道数')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--image_size', type=int, default=256,
                       help='图像尺寸')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    
    # 损失函数配置
    parser.add_argument('--bce_weight', type=float, default=0.6,
                       help='BCE损失权重')
    parser.add_argument('--dice_weight', type=float, default=0.3,
                       help='Dice损失权重')
    parser.add_argument('--focal_weight', type=float, default=0.1,
                       help='Focal损失权重')
    
    # 数据增强配置
    parser.add_argument('--enable_denoising', action='store_true',
                       help='启用去噪')
    parser.add_argument('--mixup_probability', type=float, default=0.0,
                       help='MixUp概率')
    
    # 训练选项
    parser.add_argument('--enable_mixed_precision', action='store_true',
                       help='启用混合精度训练')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='梯度累积步数')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='./experiments',
                       help='输出目录')
    parser.add_argument('--experiment_name', type=str, default='building_segmentation',
                       help='实验名称')
    
    # 推理配置
    parser.add_argument('--prediction_threshold', type=float, default=0.5,
                       help='预测阈值')
    parser.add_argument('--postprocess_min_area', type=int, default=64,
                       help='后处理最小面积')
    parser.add_argument('--postprocess_kernel_size', type=int, default=3,
                       help='后处理核大小')
    parser.add_argument('--postprocess_iterations', type=int, default=1,
                       help='后处理迭代次数')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置随机种子
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_csv_path=args.train_csv,
        train_img_dir=args.train_img_dir,
        test_csv_path=args.test_csv,
        test_img_dir=args.test_img_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
        random_seed=args.random_seed
    )
    
    # 创建模型
    if args.model_name == 'unet':
        model = UNetSegmentationModel(base_channels=args.base_channels)
    else:
        raise ValueError(f"不支持的模型: {args.model_name}")
    
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建损失函数
    criterion = CombinedLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # 创建训练器
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=str(output_dir),
        experiment_name=args.experiment_name,
        enable_mixed_precision=args.enable_mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # 开始训练
    start_time = time.time()
    history = trainer.train(num_epochs=args.epochs)
    training_time = time.time() - start_time
    
    logger.info(f"训练完成，用时: {training_time/60:.2f} 分钟")
    logger.info(f"最佳Dice分数: {trainer.best_dice:.4f} (Epoch {trainer.best_epoch})")
    
    # 加载最佳模型进行推理
    trainer.load_model('best_model.pth')
    
    # 创建后处理器
    postprocessor = MaskPostProcessor(
        min_object_area=args.postprocess_min_area,
        morphology_kernel_size=args.postprocess_kernel_size,
        morphology_iterations=args.postprocess_iterations
    )
    
    # 在测试集上进行预测
    logger.info("开始在测试集上进行预测...")
    predictions = trainer.predict(test_loader, postprocessor)
    
    # 保存预测结果
    submission_data = []
    for pred in predictions:
        submission_data.append([pred['image_name'], pred['rle']])
    
    submission_df = pd.DataFrame(submission_data)
    submission_path = output_dir / 'submission.csv'
    submission_df.to_csv(submission_path, index=False, header=False, sep='\t')
    
    logger.info(f"预测结果已保存: {submission_path}")
    logger.info(f"预测样本数量: {len(predictions)}")
    
    # 保存训练摘要
    summary = {
        'experiment_name': args.experiment_name,
        'training_time_minutes': training_time / 60,
        'best_dice': trainer.best_dice,
        'best_epoch': trainer.best_epoch,
        'total_epochs': args.epochs,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'device': str(device),
        'config': vars(args)
    }
    
    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("训练完成！")


if __name__ == '__main__':
    main()
