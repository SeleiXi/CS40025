# 24302010071 - 郭政颍

## 实现方案（基础部分）

- 数据与预处理
  - 训练集以 TSV（name\tmask）提供，mask 为 RLE。加载时对文件名做空白与首尾标点清洗，统一 Resize，归一化至 ImageNet 均值方差。
- 模型设计
  - U-Net 编码器-解码器结构，基础通道 64，输出 1 通道 logits，用于二类分割。
- 训练配置
  - AdamW 优化器，余弦退火学习率；支持混合精度与梯度累积；DataLoader 多进程并行与 pin_memory。
- 损失函数设计
  - 组合损失：BCEWithLogits + Dice（默认）。可选加入 Tversky/Focal Tversky 与 BCE 标签平滑（label smoothing）。
- 评估指标
  - 主指标 Dice（验证集），辅指标 IoU。训练阶段按批次在线统计。

## 方案报告（创新/改进部分）

- 损失函数改进
  - TverskyLoss（α=0.7, β=0.3）：强化对小目标与边界错分的约束，适配前景稀疏的建筑物场景。
  - FocalTverskyLoss（γ=0.75）：进一步聚焦困难样本与薄边界区域。
  - BCE 标签平滑（如 0.05）：降低过置信，提升泛化稳定性。
- 训练策略优化
  - 动态阈值评估（0.35→0.65）：训练早期提升召回，后期提升精确度，整体带来更稳的 Dice 收敛。
  - 混合精度与梯度累积：在显存与吞吐之间取得更优平衡。
- 数据增强与正则化
  - 几何增强（随机裁剪、仿射、弹性/透视/网格扭曲、翻转/旋转），像素增强（噪声/模糊/亮度对比/色彩偏移、CLAHE、Gamma），以及 CoarseDropout。
  - 可选 MixUp（p、alpha 可配）：缓解过拟合，提升鲁棒性。
- 推理与后处理
  - 形态学开闭 + 小连通域移除/可选保留最大连通域，有效抑制散点噪声。
  - 验证集概率图预计算 + 网格搜索阈值/形态学参数，快速得到最优提交配置。
- 工程健壮性
  - CSV 噪声文件名自动清洗；依赖自检；GPU 变量健壮化；Albumentations API 兼容修复。

## 遇到的问题与解决方案

- CSV 中个别 name 带尾逗号/引号导致找不到图：引入路径清洗与二次回退读取。
- Albumentations RandomResizedCrop 新版参数校验变更：统一使用 size=(H, W) 写法。
- Shell 里 GPU_COUNT 未绑定：统一读取为局部变量并回写导出，避免未定义错误。
- 训练脚本依赖检查误报：按包名/导入名双列表精确匹配（如 opencv-python→cv2，scikit-learn→sklearn）。
- 评测分数不够高：改进损失函数，引入 TverskyLoss（α=0.7, β=0.3）：强化对小目标与边界错分的约束，适配前景稀疏的建筑物场景 后得到极高改进。


```

## 超参数

```yaml
# configs/training_config.yaml
data:
  train_csv: "./data_source/train_mask.csv"
  train_img_dir: "./data_source/train"
  test_csv: "./data_source/test_a_samplesubmit.csv"
  test_img_dir: "./data_source/test_a"
  image_size: 256
  batch_size: 16
  num_workers: 4
  validation_split: 0.2

model:
  name: "unet"
  base_channels: 64

training:
  epochs: 100
  learning_rate: 1e-4
  weight_decay: 1e-4
  enable_mixed_precision: true
  gradient_accumulation_steps: 1

loss:
  bce_weight: 0.6
  dice_weight: 0.3
  focal_weight: 0.1

augmentation:
  enable_denoising: false
  mixup_probability: 0.0

output:
  output_dir: "./experiments"
  experiment_name: "building_segmentation"
```



## 结果记录与可复现性

- 保存内容：TensorBoard 日志、best_model.pth、training_history.json、推理超参搜索结果。
- 随机性：固定 seed，划分可复现；输出目录包含最佳 checkpoint 与指标快照。

![result_image](/img/Snipaste_2025-10-26_23-27-46.jpg)

---


## 环境配置

```bash

# 可以先配好conda环境

pip install -r requirements.txt

#### 2.2 使用启动脚本

```bash
# 单卡
bash launch_training.sh --data-dir ./data_source --output-dir ./experiments --experiment unet_v1

# 多卡
bash launch_training.sh --gpus 4 --batch-size 32 --mixed-precision

```
