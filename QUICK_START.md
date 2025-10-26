# 快速开始指南

## 🚀 一键安装

### Linux/macOS
```bash
# 下载并运行安装脚本
bash install_dependencies.sh

# 或者指定参数
bash install_dependencies.sh --env-name my_env --gpu --cuda-version 11.8
```

### Windows
```cmd
REM 下载并运行安装脚本
install_dependencies.bat

REM 或者指定参数
install_dependencies.bat --env-name my_env --gpu --cuda-version 11.8
```

## 📦 手动安装

### 1. 创建虚拟环境
```bash
# 使用conda (推荐)
conda create -n semantic_segmentation python=3.9
conda activate semantic_segmentation

# 或使用venv
python -m venv semantic_segmentation_env
source semantic_segmentation_env/bin/activate  # Linux/macOS
# semantic_segmentation_env\Scripts\activate    # Windows
```

### 2. 安装依赖
```bash
# 安装所有依赖
pip install -r requirements.txt

# 或分步安装
pip install torch torchvision torchaudio  # GPU版本
pip install numpy pandas opencv-python
pip install albumentations matplotlib seaborn
pip install scikit-learn tqdm PyYAML tensorboard
```

### 3. 验证安装
```bash
python check_environment.py
```

## 🎯 开始训练

### 基本训练
```bash
# 激活环境
conda activate semantic_segmentation

# 开始训练
bash launch_training.sh --data-dir ./data_source --output-dir ./experiments --experiment unet_v1
```

### 超参数优化
```bash
# 训练完成后进行超参数优化
bash launch_hyperparameter_optimization.sh --model ./experiments/unet_v1/best_model.pth
```

## 📋 系统要求

- **Python**: 3.8 - 3.11 (推荐 3.9)
- **内存**: 8GB+ RAM
- **GPU**: NVIDIA GPU with CUDA (可选)
- **存储**: 10GB+ 可用空间

## 🔧 常见问题

### 安装失败
```bash
# 升级pip
pip install --upgrade pip

# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### CUDA问题
```bash
# 检查CUDA版本
nvcc --version

# 安装对应版本PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 内存不足
```bash
# 安装CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📚 更多信息

- [详细技术文档](TECHNICAL_DOCUMENTATION.md)
- [依赖配置指南](DEPENDENCY_SETUP_GUIDE.md)
- [项目总结](PROJECT_SUMMARY.md)
