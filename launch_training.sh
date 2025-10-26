#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# 语义分割模型训练启动脚本
# 支持单GPU和多GPU分布式训练，自动环境配置和参数管理

set -euo pipefail

# 脚本信息
SCRIPT_NAME="语义分割训练启动器"
SCRIPT_VERSION="2.0.0"
SCRIPT_AUTHOR="AI Research Team"

# 颜色输出函数
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
${SCRIPT_NAME} v${SCRIPT_VERSION}

用法: $0 [选项]

选项:
    -h, --help              显示此帮助信息
    -v, --version           显示版本信息
    -c, --config FILE       指定配置文件路径
    -d, --data-dir DIR      指定数据目录
    -o, --output-dir DIR    指定输出目录
    -e, --experiment NAME   指定实验名称
    -g, --gpus NUM          指定GPU数量 (默认: 1)
    -b, --batch-size NUM    指定批次大小 (默认: 16)
    --epochs NUM            指定训练轮数 (默认: 100)
    --learning-rate NUM     指定学习率 (默认: 1e-4)
    --image-size NUM        指定图像尺寸 (默认: 256)
    --mixed-precision       启用混合精度训练
    --dry-run              仅显示配置，不执行训练
    --resume PATH           从检查点恢复训练

环境变量:
    CUDA_VISIBLE_DEVICES   指定可见的GPU设备
    PYTHONPATH             指定Python路径
    OMP_NUM_THREADS        指定OpenMP线程数

示例:
    # 基本训练
    $0 --data-dir ./data --output-dir ./results --experiment unet_v1

    # 多GPU训练
    $0 --gpus 4 --batch-size 32 --mixed-precision

    # 从检查点恢复
    $0 --resume ./results/unet_v1/checkpoint_epoch_50.pth

    # 使用配置文件
    $0 --config ./configs/training_config.yaml
EOF
}

# 显示版本信息
show_version() {
    echo "${SCRIPT_NAME} v${SCRIPT_VERSION}"
    echo "作者: ${SCRIPT_AUTHOR}"
    echo "Python版本: $(python --version 2>&1)"
    echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"
    echo "CUDA版本: $(nvcc --version 2>/dev/null | grep release || echo '未安装')"
}

# 检查依赖
check_dependencies() {
    print_info "检查系统依赖..."
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        print_error "Python未安装或不在PATH中"
        exit 1
    fi
    
    # 使用Python脚本检查依赖
    if [ -f "check_dependencies.py" ]; then
        local check_result=$(python check_dependencies.py 2>&1)
        if [ $? -ne 0 ]; then
            # 提取缺失的包
            local missing_packages=$(echo "$check_result" | grep "MISSING_PACKAGES:" | cut -d' ' -f2-)
            if [ -n "$missing_packages" ]; then
                print_warning "缺少以下Python包: $missing_packages"
                print_info "请运行: pip install $missing_packages"
                exit 1
            fi
        fi
    else
        # 备用检查方法
        local required_packages=("torch" "torchvision" "numpy" "pandas" "opencv-python" "albumentations" "matplotlib" "seaborn" "sklearn" "tqdm" "yaml" "tensorboard")
        local import_names=("torch" "torchvision" "numpy" "pandas" "cv2" "albumentations" "matplotlib" "seaborn" "sklearn" "tqdm" "yaml" "tensorboard")
        local missing_packages=()
        
        for i in "${!required_packages[@]}"; do
            local package="${required_packages[$i]}"
            local import_name="${import_names[$i]}"
            if ! python -c "import ${import_name}" &> /dev/null; then
                missing_packages+=("${package}")
            fi
        done
        
        if [ ${#missing_packages[@]} -ne 0 ]; then
            print_warning "缺少以下Python包: ${missing_packages[*]}"
            print_info "请运行: pip install ${missing_packages[*]}"
            exit 1
        fi
    fi
    
    # 检查CUDA（如果指定了GPU）
    if [ "${GPU_COUNT:-1}" -gt 0 ]; then
        if ! python -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
            print_warning "CUDA不可用，将使用CPU训练"
            GPU_COUNT=0
        else
            local cuda_device_count=$(python -c "import torch; print(torch.cuda.device_count())")
            if [ "${GPU_COUNT}" -gt "${cuda_device_count}" ]; then
                print_warning "请求的GPU数量(${GPU_COUNT})超过可用数量(${cuda_device_count})"
                GPU_COUNT=${cuda_device_count}
            fi
        fi
    fi
    
    print_success "依赖检查完成"
}

# 设置环境变量
setup_environment() {
    print_info "配置训练环境..."
    
    # 设置Python环境
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    
    # 设置CUDA环境
    if [ "${GPU_COUNT:-1}" -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        print_info "使用GPU训练，设备: ${CUDA_VISIBLE_DEVICES}"
    else
        export CUDA_VISIBLE_DEVICES=""
        print_info "使用CPU训练"
    fi
    
    # 设置随机种子
    export RANDOM_SEED="${RANDOM_SEED:-42}"
    
    # 创建输出目录
    if [ -n "${OUTPUT_DIR:-}" ]; then
        mkdir -p "${OUTPUT_DIR}"
        print_info "输出目录: ${OUTPUT_DIR}"
    fi
    
    print_success "环境配置完成"
}

# 解析配置文件
parse_config_file() {
    local config_file="$1"
    
    if [ ! -f "${config_file}" ]; then
        print_error "配置文件不存在: ${config_file}"
        exit 1
    fi
    
    print_info "解析配置文件: ${config_file}"
    
    # 根据文件扩展名选择解析方法
    case "${config_file}" in
        *.yaml|*.yml)
            # 使用Python解析YAML
            python -c "
import yaml
import sys
import os

try:
    with open('${config_file}', 'r') as f:
        config = yaml.safe_load(f)
    
    # 导出环境变量
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)):
            os.environ[f'CONFIG_{key.upper()}'] = str(value)
            print(f'export CONFIG_{key.upper()}={value}')
except Exception as e:
    print(f'Error parsing YAML: {e}', file=sys.stderr)
    sys.exit(1)
" | source
            ;;
        *.json)
            # 使用Python解析JSON
            python -c "
import json
import sys
import os

try:
    with open('${config_file}', 'r') as f:
        config = json.load(f)
    
    # 导出环境变量
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)):
            os.environ[f'CONFIG_{key.upper()}'] = str(value)
            print(f'export CONFIG_{key.upper()}={value}')
except Exception as e:
    print(f'Error parsing JSON: {e}', file=sys.stderr)
    sys.exit(1)
" | source
            ;;
        *)
            print_error "不支持的配置文件格式: ${config_file}"
            exit 1
            ;;
    esac
    
    print_success "配置文件解析完成"
}

# 构建训练命令
build_training_command() {
    local cmd_args=()
    
    # 基本参数
    cmd_args+=("--train_csv" "${TRAIN_CSV:-./data_source/train_mask.csv}")
    cmd_args+=("--train_img_dir" "${TRAIN_IMG_DIR:-./data_source/train}")
    cmd_args+=("--test_csv" "${TEST_CSV:-./data_source/test_a_samplesubmit.csv}")
    cmd_args+=("--test_img_dir" "${TEST_IMG_DIR:-./data_source/test_a}")
    
    # 模型参数
    cmd_args+=("--model_name" "${MODEL_NAME:-unet}")
    cmd_args+=("--base_channels" "${BASE_CHANNELS:-64}")
    
    # 训练参数
    cmd_args+=("--epochs" "${EPOCHS:-100}")
    cmd_args+=("--batch_size" "${BATCH_SIZE:-16}")
    cmd_args+=("--image_size" "${IMAGE_SIZE:-256}")
    cmd_args+=("--learning_rate" "${LEARNING_RATE:-1e-4}")
    cmd_args+=("--weight_decay" "${WEIGHT_DECAY:-1e-4}")
    cmd_args+=("--num_workers" "${NUM_WORKERS:-4}")
    
    # 损失函数参数
    cmd_args+=("--bce_weight" "${BCE_WEIGHT:-0.6}")
    cmd_args+=("--dice_weight" "${DICE_WEIGHT:-0.3}")
    cmd_args+=("--focal_weight" "${FOCAL_WEIGHT:-0.1}")
    
    # 数据增强参数
    if [ "${ENABLE_DENOISING:-false}" = "true" ]; then
        cmd_args+=("--enable_denoising")
    fi
    cmd_args+=("--mixup_probability" "${MIXUP_PROBABILITY:-0.0}")
    
    # 训练选项
    if [ "${ENABLE_MIXED_PRECISION:-false}" = "true" ]; then
        cmd_args+=("--enable_mixed_precision")
    fi
    cmd_args+=("--gradient_accumulation_steps" "${GRADIENT_ACCUMULATION_STEPS:-1}")
    cmd_args+=("--validation_split" "${VALIDATION_SPLIT:-0.2}")
    cmd_args+=("--random_seed" "${RANDOM_SEED:-42}")
    
    # 输出参数
    cmd_args+=("--output_dir" "${OUTPUT_DIR:-./experiments}")
    cmd_args+=("--experiment_name" "${EXPERIMENT_NAME:-building_segmentation}")
    
    # 推理参数
    cmd_args+=("--prediction_threshold" "${PREDICTION_THRESHOLD:-0.5}")
    cmd_args+=("--postprocess_min_area" "${POSTPROCESS_MIN_AREA:-64}")
    cmd_args+=("--postprocess_kernel_size" "${POSTPROCESS_KERNEL_SIZE:-3}")
    cmd_args+=("--postprocess_iterations" "${POSTPROCESS_ITERATIONS:-1}")
    
    # 从配置文件读取的参数
    for var in $(env | grep '^CONFIG_' | cut -d'=' -f1); do
        local key=$(echo "${var}" | sed 's/CONFIG_//' | tr '[:upper:]' '[:lower:]')
        local value=$(eval echo \$${var})
        cmd_args+=("--${key}" "${value}")
    done
    
    echo "${cmd_args[*]}"
}

# 执行训练
run_training() {
    local training_script="semantic_segmentation_trainer.py"
    
    if [ ! -f "${training_script}" ]; then
        print_error "训练脚本不存在: ${training_script}"
        exit 1
    fi
    
    # 构建命令参数
    local cmd_args=($(build_training_command))
    
    print_info "开始训练..."
    print_info "训练脚本: ${training_script}"
    print_info "参数: ${cmd_args[*]}"
    
    # 选择执行方式
    if [ "${GPU_COUNT:-1}" -gt 1 ]; then
        # 多GPU分布式训练
        print_info "使用${GPU_COUNT}个GPU进行分布式训练"
        exec torchrun \
            --standalone \
            --nproc_per_node="${GPU_COUNT}" \
            "${training_script}" \
            "${cmd_args[@]}"
    else
        # 单GPU或CPU训练
        exec python "${training_script}" "${cmd_args[@]}"
    fi
}

# 显示配置摘要
show_config_summary() {
    print_info "训练配置摘要:"
    echo "  实验名称: ${EXPERIMENT_NAME:-building_segmentation}"
    echo "  数据目录: ${TRAIN_IMG_DIR:-./data_source/train}"
    echo "  输出目录: ${OUTPUT_DIR:-./experiments}"
    echo "  模型类型: ${MODEL_NAME:-unet}"
    echo "  批次大小: ${BATCH_SIZE:-16}"
    echo "  图像尺寸: ${IMAGE_SIZE:-256}"
    echo "  训练轮数: ${EPOCHS:-100}"
    echo "  学习率: ${LEARNING_RATE:-1e-4}"
    echo "  GPU数量: ${GPU_COUNT:-1}"
    echo "  混合精度: ${ENABLE_MIXED_PRECISION:-false}"
    echo "  随机种子: ${RANDOM_SEED:-42}"
}

# 主函数
main() {
    # 默认参数
    local config_file=""
    local dry_run=false
    local resume_path=""
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--version)
                show_version
                exit 0
                ;;
            -c|--config)
                config_file="$2"
                shift 2
                ;;
            -d|--data-dir)
                export TRAIN_IMG_DIR="$2/train"
                export TEST_IMG_DIR="$2/test_a"
                shift 2
                ;;
            -o|--output-dir)
                export OUTPUT_DIR="$2"
                shift 2
                ;;
            -e|--experiment)
                export EXPERIMENT_NAME="$2"
                shift 2
                ;;
            -g|--gpus)
                export GPU_COUNT="$2"
                shift 2
                ;;
            -b|--batch-size)
                export BATCH_SIZE="$2"
                shift 2
                ;;
            --epochs)
                export EPOCHS="$2"
                shift 2
                ;;
            --learning-rate)
                export LEARNING_RATE="$2"
                shift 2
                ;;
            --image-size)
                export IMAGE_SIZE="$2"
                shift 2
                ;;
            --mixed-precision)
                export ENABLE_MIXED_PRECISION="true"
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --resume)
                resume_path="$2"
                shift 2
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 显示脚本信息
    print_info "${SCRIPT_NAME} v${SCRIPT_VERSION}"
    
    # 解析配置文件
    if [ -n "${config_file}" ]; then
        parse_config_file "${config_file}"
    fi
    
    # 检查依赖
    check_dependencies
    
    # 设置环境
    setup_environment
    
    # 显示配置摘要
    show_config_summary
    
    # 如果是干运行，只显示配置
    if [ "${dry_run}" = true ]; then
        print_info "干运行模式，不执行实际训练"
        exit 0
    fi
    
    # 执行训练
    run_training
}

# 错误处理
trap 'print_error "脚本执行失败，退出码: $?"' ERR

# 执行主函数
main "$@"
