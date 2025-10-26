#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数优化系统 - 语义分割模型推理参数调优
基于网格搜索和贝叶斯优化的混合策略，自动寻找最佳推理参数组合

主要功能：
- 网格搜索：系统遍历所有参数组合
- 贝叶斯优化：智能搜索最优参数空间
- 多指标评估：Dice、IoU、F1等多种指标
- 结果可视化：生成详细的参数性能分析报告
- 自动保存：最佳参数配置和性能报告

使用示例：
python hyperparameter_optimizer.py \
  --model_checkpoint ./experiments/building_segmentation/best_model.pth \
  --data_config ./configs/data_config.yaml \
  --search_strategy grid \
  --output_dir ./hyperparameter_results \
  --max_iterations 100
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm

# 贝叶斯优化相关库
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_OPTIMIZATION_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZATION_AVAILABLE = False
    logger.warning("贝叶斯优化库未安装，将使用网格搜索")

# 设置环境变量
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
warnings.filterwarnings("ignore", category=UserWarning)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hyperparameter_optimization.log')
    ]
)
logger = logging.getLogger(__name__)

# 导入主训练脚本中的组件
from semantic_segmentation_trainer import (
    BuildingSegmentationDataset,
    UNetSegmentationModel,
    RunLengthEncoder,
    MaskPostProcessor,
    SegmentationMetrics,
    create_data_loaders
)


# ===================== 评估指标计算器 =====================

class ComprehensiveMetricsCalculator:
    """综合评估指标计算器"""
    
    def __init__(self, threshold: float = 0.5):
        """
        初始化指标计算器
        
        Args:
            threshold: 二值化阈值
        """
        self.threshold = threshold
    
    def calculate_all_metrics(self, 
                            predictions: np.ndarray, 
                            targets: np.ndarray) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            predictions: 预测概率图，形状为[H, W]
            targets: 目标掩码，形状为[H, W]
            
        Returns:
            包含所有指标的字典
        """
        # 二值化
        pred_binary = (predictions > self.threshold).astype(np.uint8)
        target_binary = targets.astype(np.uint8)
        
        # 展平为一维数组用于sklearn计算
        pred_flat = pred_binary.flatten()
        target_flat = target_binary.flatten()
        
        # 计算各种指标
        metrics = {}
        
        # Dice系数
        metrics['dice'] = SegmentationMetrics.dice_coefficient(
            predictions, targets, self.threshold
        )
        
        # IoU分数
        metrics['iou'] = SegmentationMetrics.iou_score(
            predictions, targets, self.threshold
        )
        
        # 精确率、召回率、F1分数
        if target_flat.sum() > 0:  # 避免除零错误
            metrics['precision'] = precision_score(target_flat, pred_flat, zero_division=0)
            metrics['recall'] = recall_score(target_flat, pred_flat, zero_division=0)
            metrics['f1'] = f1_score(target_flat, pred_flat, zero_division=0)
        else:
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1'] = 0.0
        
        # 像素准确率
        metrics['pixel_accuracy'] = (pred_flat == target_flat).mean()
        
        # 特异性
        tn = ((pred_flat == 0) & (target_flat == 0)).sum()
        fp = ((pred_flat == 1) & (target_flat == 0)).sum()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # 平衡准确率
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        return metrics


# ===================== 参数空间定义 =====================

@dataclass
class ParameterSpace:
    """参数空间定义"""
    
    # 预测阈值
    prediction_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    
    # 后处理参数
    min_object_areas: List[int] = field(default_factory=lambda: [0, 32, 64, 96, 128, 192, 256])
    morphology_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    morphology_iterations: List[int] = field(default_factory=lambda: [1, 2, 3])
    keep_largest_options: List[bool] = field(default_factory=lambda: [False, True])
    
    # 图像尺寸（用于测试不同输入尺寸的影响）
    image_sizes: List[int] = field(default_factory=lambda: [256, 384, 512])
    
    def get_grid_search_space(self) -> List[Dict[str, Any]]:
        """获取网格搜索参数空间"""
        from itertools import product
        
        combinations = list(product(
            self.prediction_thresholds,
            self.min_object_areas,
            self.morphology_kernel_sizes,
            self.morphology_iterations,
            self.keep_largest_options,
            self.image_sizes
        ))
        
        param_combinations = []
        for combo in combinations:
            param_combinations.append({
                'prediction_threshold': combo[0],
                'min_object_area': combo[1],
                'morphology_kernel_size': combo[2],
                'morphology_iterations': combo[3],
                'keep_largest': combo[4],
                'image_size': combo[5]
            })
        
        return param_combinations
    
    def get_bayesian_search_space(self) -> List:
        """获取贝叶斯优化参数空间"""
        if not BAYESIAN_OPTIMIZATION_AVAILABLE:
            raise ImportError("贝叶斯优化库未安装")
        
        space = [
            Real(0.3, 0.7, name='prediction_threshold'),
            Integer(0, 256, name='min_object_area'),
            Integer(3, 7, name='morphology_kernel_size'),
            Integer(1, 3, name='morphology_iterations'),
            Categorical([False, True], name='keep_largest'),
            Categorical([256, 384, 512], name='image_size')
        ]
        
        return space


# ===================== 模型评估器 =====================

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self,
                 model: nn.Module,
                 val_loader: DataLoader,
                 device: torch.device,
                 enable_mixed_precision: bool = False):
        """
        初始化评估器
        
        Args:
            model: 训练好的模型
            val_loader: 验证数据加载器
            device: 计算设备
            enable_mixed_precision: 是否启用混合精度
        """
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.enable_mixed_precision = enable_mixed_precision
        
        # 将模型设置为评估模式
        self.model.eval()
        self.model.to(device)
        
        # 预计算所有验证集的预测结果（避免重复计算）
        self._precompute_predictions()
    
    def _precompute_predictions(self):
        """预计算所有验证集的预测结果"""
        logger.info("预计算验证集预测结果...")
        
        self.val_images = []
        self.val_masks = []
        self.val_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="预计算预测"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 前向传播
                if self.enable_mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = self.model(images)
                else:
                    logits = self.model(images)
                
                # 转换为概率
                probs = torch.sigmoid(logits)
                
                # 存储结果
                for i in range(images.shape[0]):
                    self.val_images.append(images[i].cpu())
                    self.val_masks.append(masks[i, 0].cpu().numpy())
                    self.val_predictions.append(probs[i, 0].cpu().numpy())
        
        logger.info(f"预计算完成，共{len(self.val_predictions)}个样本")
    
    def evaluate_parameters(self, 
                          prediction_threshold: float,
                          min_object_area: int,
                          morphology_kernel_size: int,
                          morphology_iterations: int,
                          keep_largest: bool,
                          image_size: int) -> Dict[str, float]:
        """
        评估特定参数组合的性能
        
        Args:
            prediction_threshold: 预测阈值
            min_object_area: 最小对象面积
            morphology_kernel_size: 形态学核大小
            morphology_iterations: 形态学迭代次数
            keep_largest: 是否只保留最大连通分量
            image_size: 图像尺寸
            
        Returns:
            评估指标字典
        """
        # 创建后处理器
        postprocessor = MaskPostProcessor(
            min_object_area=min_object_area,
            morphology_kernel_size=morphology_kernel_size,
            morphology_iterations=morphology_iterations,
            keep_largest_component=keep_largest
        )
        
        # 创建指标计算器
        metrics_calculator = ComprehensiveMetricsCalculator(threshold=prediction_threshold)
        
        # 计算所有样本的指标
        all_metrics = []
        
        for pred_prob, target_mask in zip(self.val_predictions, self.val_masks):
            # 调整预测结果到原始尺寸
            pred_resized = cv2.resize(pred_prob, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            # 二值化
            binary_mask = (pred_resized > prediction_threshold).astype(np.uint8)
            
            # 后处理
            processed_mask = postprocessor.process_mask(binary_mask)
            
            # 计算指标
            sample_metrics = metrics_calculator.calculate_all_metrics(
                pred_resized, target_mask
            )
            all_metrics.append(sample_metrics)
        
        # 计算平均指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics


# ===================== 优化策略 =====================

class GridSearchOptimizer:
    """网格搜索优化器"""
    
    def __init__(self, parameter_space: ParameterSpace):
        self.parameter_space = parameter_space
        self.results = []
    
    def optimize(self, evaluator: ModelEvaluator, max_iterations: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        执行网格搜索优化
        
        Args:
            evaluator: 模型评估器
            max_iterations: 最大迭代次数（None表示搜索所有组合）
            
        Returns:
            搜索结果列表
        """
        param_combinations = self.parameter_space.get_grid_search_space()
        
        if max_iterations is not None:
            param_combinations = param_combinations[:max_iterations]
        
        logger.info(f"开始网格搜索，共{len(param_combinations)}个参数组合")
        
        for i, params in enumerate(tqdm(param_combinations, desc="网格搜索")):
            try:
                # 评估参数组合
                metrics = evaluator.evaluate_parameters(**params)
                
                # 记录结果
                result = {
                    'iteration': i,
                    'parameters': params,
                    'metrics': metrics,
                    'timestamp': time.time()
                }
                self.results.append(result)
                
                # 记录最佳结果
                if i == 0 or metrics['dice'] > max(r['metrics']['dice'] for r in self.results[:-1]):
                    logger.info(f"发现更好的参数组合 (Dice: {metrics['dice']:.4f}): {params}")
                
            except Exception as e:
                logger.error(f"参数组合评估失败: {params}, 错误: {e}")
                continue
        
        logger.info(f"网格搜索完成，共评估{len(self.results)}个参数组合")
        return self.results


class BayesianOptimizer:
    """贝叶斯优化器"""
    
    def __init__(self, parameter_space: ParameterSpace):
        if not BAYESIAN_OPTIMIZATION_AVAILABLE:
            raise ImportError("贝叶斯优化库未安装")
        
        self.parameter_space = parameter_space
        self.results = []
        self.space = parameter_space.get_bayesian_search_space()
    
    def optimize(self, evaluator: ModelEvaluator, max_iterations: int = 50) -> List[Dict[str, Any]]:
        """
        执行贝叶斯优化
        
        Args:
            evaluator: 模型评估器
            max_iterations: 最大迭代次数
            
        Returns:
            搜索结果列表
        """
        logger.info(f"开始贝叶斯优化，最大迭代次数: {max_iterations}")
        
        @use_named_args(self.space)
        def objective(**params):
            """优化目标函数"""
            try:
                metrics = evaluator.evaluate_parameters(**params)
                # 返回负的Dice分数（因为gp_minimize是最小化）
                return -metrics['dice']
            except Exception as e:
                logger.error(f"参数评估失败: {params}, 错误: {e}")
                return 1.0  # 返回最差分数
        
        # 执行贝叶斯优化
        result = gp_minimize(
            func=objective,
            dimensions=self.space,
            n_calls=max_iterations,
            random_state=42,
            n_jobs=1
        )
        
        # 转换结果格式
        for i, (params, score) in enumerate(zip(result.x_iters, result.func_vals)):
            param_dict = dict(zip([dim.name for dim in self.space], params))
            metrics = {'dice': -score}  # 转换回正的Dice分数
            
            result_entry = {
                'iteration': i,
                'parameters': param_dict,
                'metrics': metrics,
                'timestamp': time.time()
            }
            self.results.append(result_entry)
        
        logger.info(f"贝叶斯优化完成，最佳Dice分数: {-result.fun:.4f}")
        return self.results


# ===================== 结果分析和可视化 =====================

class ResultsAnalyzer:
    """结果分析器"""
    
    def __init__(self, results: List[Dict[str, Any]], output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析搜索结果"""
        if not self.results:
            logger.warning("没有搜索结果可供分析")
            return {}
        
        # 转换为DataFrame便于分析
        df_data = []
        for result in self.results:
            row = result['parameters'].copy()
            row.update(result['metrics'])
            row['iteration'] = result['iteration']
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # 基本统计
        analysis = {
            'total_combinations': len(df),
            'best_dice': df['dice'].max(),
            'best_iou': df['iou'].max(),
            'best_f1': df['f1'].max(),
            'mean_dice': df['dice'].mean(),
            'std_dice': df['dice'].std()
        }
        
        # 最佳参数组合
        best_idx = df['dice'].idxmax()
        analysis['best_parameters'] = df.loc[best_idx, df.columns.difference(['iteration', 'dice', 'iou', 'f1', 'precision', 'recall', 'pixel_accuracy', 'specificity', 'balanced_accuracy'])].to_dict()
        
        # 参数重要性分析
        param_importance = {}
        for param in ['prediction_threshold', 'min_object_area', 'morphology_kernel_size', 'morphology_iterations', 'keep_largest', 'image_size']:
            if param in df.columns:
                correlation = df[param].corr(df['dice'])
                param_importance[param] = abs(correlation)
        
        analysis['parameter_importance'] = param_importance
        
        # 保存分析结果
        analysis_path = self.output_dir / 'analysis_results.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"结果分析完成，最佳Dice分数: {analysis['best_dice']:.4f}")
        return analysis
    
    def create_visualizations(self):
        """创建可视化图表"""
        if not self.results:
            return
        
        # 转换为DataFrame
        df_data = []
        for result in self.results:
            row = result['parameters'].copy()
            row.update(result['metrics'])
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        fig_size = (12, 8)
        
        # 1. Dice分数分布
        plt.figure(figsize=fig_size)
        plt.hist(df['dice'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Dice Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Dice Scores')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dice_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 参数与Dice分数的关系
        numeric_params = ['prediction_threshold', 'min_object_area', 'morphology_kernel_size', 'morphology_iterations']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, param in enumerate(numeric_params):
            if param in df.columns:
                axes[i].scatter(df[param], df['dice'], alpha=0.6)
                axes[i].set_xlabel(param)
                axes[i].set_ylabel('Dice Score')
                axes[i].set_title(f'{param} vs Dice Score')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_vs_dice.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 热力图：参数组合性能
        if len(df) > 1:
            # 选择最重要的参数进行热力图分析
            important_params = ['prediction_threshold', 'min_object_area']
            if all(param in df.columns for param in important_params):
                pivot_table = df.pivot_table(
                    values='dice', 
                    index='min_object_area', 
                    columns='prediction_threshold', 
                    aggfunc='mean'
                )
                
                plt.figure(figsize=fig_size)
                sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
                plt.title('Dice Score Heatmap: Prediction Threshold vs Min Object Area')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'dice_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 4. 迭代过程（如果是贝叶斯优化）
        if len(df) > 10:  # 假设贝叶斯优化有足够的迭代次数
            plt.figure(figsize=fig_size)
            plt.plot(df['iteration'], df['dice'], 'o-', alpha=0.7)
            plt.xlabel('Iteration')
            plt.ylabel('Dice Score')
            plt.title('Optimization Progress')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'optimization_progress.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("可视化图表已保存")
    
    def save_detailed_results(self):
        """保存详细的搜索结果"""
        # 保存为CSV
        df_data = []
        for result in self.results:
            row = result['parameters'].copy()
            row.update(result['metrics'])
            row['iteration'] = result['iteration']
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # 按Dice分数排序
        df_sorted = df.sort_values('dice', ascending=False)
        
        # 保存完整结果
        csv_path = self.output_dir / 'detailed_results.csv'
        df_sorted.to_csv(csv_path, index=False)
        
        # 保存前10个最佳结果
        top10_path = self.output_dir / 'top10_results.csv'
        df_sorted.head(10).to_csv(top10_path, index=False)
        
        logger.info(f"详细结果已保存: {csv_path}")
        logger.info(f"前10个最佳结果已保存: {top10_path}")


# ===================== 主函数 =====================

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='超参数优化系统')
    
    # 模型和数据
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--train_csv', type=str, default='./data_source/train_mask.csv',
                       help='训练CSV文件路径')
    parser.add_argument('--train_img_dir', type=str, default='./data_source/train',
                       help='训练图像目录')
    parser.add_argument('--test_csv', type=str, default='./data_source/test_a_samplesubmit.csv',
                       help='测试CSV文件路径')
    parser.add_argument('--test_img_dir', type=str, default='./data_source/test_a',
                       help='测试图像目录')
    
    # 优化配置
    parser.add_argument('--search_strategy', type=str, default='grid',
                       choices=['grid', 'bayesian'],
                       help='搜索策略')
    parser.add_argument('--max_iterations', type=int, default=100,
                       help='最大迭代次数')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    
    # 参数空间配置
    parser.add_argument('--prediction_thresholds', type=float, nargs='+',
                       default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
                       help='预测阈值列表')
    parser.add_argument('--min_object_areas', type=int, nargs='+',
                       default=[0, 32, 64, 96, 128, 192, 256],
                       help='最小对象面积列表')
    parser.add_argument('--morphology_kernel_sizes', type=int, nargs='+',
                       default=[3, 5, 7],
                       help='形态学核大小列表')
    parser.add_argument('--morphology_iterations', type=int, nargs='+',
                       default=[1, 2, 3],
                       help='形态学迭代次数列表')
    parser.add_argument('--image_sizes', type=int, nargs='+',
                       default=[256, 384, 512],
                       help='图像尺寸列表')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='./hyperparameter_results',
                       help='输出目录')
    parser.add_argument('--experiment_name', type=str, default='hyperparameter_optimization',
                       help='实验名称')
    
    # 其他配置
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--enable_mixed_precision', action='store_true',
                       help='启用混合精度')
    
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
    
    # 加载模型
    logger.info(f"加载模型: {args.model_checkpoint}")
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    
    # 创建模型（假设使用UNet）
    model = UNetSegmentationModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_csv_path=args.train_csv,
        train_img_dir=args.train_img_dir,
        test_csv_path=args.test_csv,
        test_img_dir=args.test_img_dir,
        image_size=256,  # 使用默认尺寸，实际尺寸在优化过程中动态调整
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
        random_seed=args.random_seed
    )
    
    # 创建模型评估器
    evaluator = ModelEvaluator(
        model=model,
        val_loader=val_loader,
        device=device,
        enable_mixed_precision=args.enable_mixed_precision
    )
    
    # 创建参数空间
    parameter_space = ParameterSpace(
        prediction_thresholds=args.prediction_thresholds,
        min_object_areas=args.min_object_areas,
        morphology_kernel_sizes=args.morphology_kernel_sizes,
        morphology_iterations=args.morphology_iterations,
        image_sizes=args.image_sizes
    )
    
    # 执行优化
    start_time = time.time()
    
    if args.search_strategy == 'grid':
        optimizer = GridSearchOptimizer(parameter_space)
        results = optimizer.optimize(evaluator, args.max_iterations)
    elif args.search_strategy == 'bayesian':
        if not BAYESIAN_OPTIMIZATION_AVAILABLE:
            logger.error("贝叶斯优化库未安装，请安装scikit-optimize或使用网格搜索")
            return
        optimizer = BayesianOptimizer(parameter_space)
        results = optimizer.optimize(evaluator, args.max_iterations)
    else:
        raise ValueError(f"不支持的搜索策略: {args.search_strategy}")
    
    optimization_time = time.time() - start_time
    
    logger.info(f"优化完成，用时: {optimization_time/60:.2f} 分钟")
    logger.info(f"共评估了 {len(results)} 个参数组合")
    
    # 分析结果
    analyzer = ResultsAnalyzer(results, output_dir)
    analysis = analyzer.analyze_results()
    
    # 创建可视化
    analyzer.create_visualizations()
    
    # 保存详细结果
    analyzer.save_detailed_results()
    
    # 保存优化摘要
    summary = {
        'experiment_name': args.experiment_name,
        'search_strategy': args.search_strategy,
        'optimization_time_minutes': optimization_time / 60,
        'total_combinations_evaluated': len(results),
        'best_dice': analysis.get('best_dice', 0.0),
        'best_parameters': analysis.get('best_parameters', {}),
        'config': vars(args)
    }
    
    summary_path = output_dir / 'optimization_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("超参数优化完成！")
    logger.info(f"最佳Dice分数: {analysis.get('best_dice', 0.0):.4f}")
    logger.info(f"最佳参数: {analysis.get('best_parameters', {})}")


if __name__ == '__main__':
    main()
