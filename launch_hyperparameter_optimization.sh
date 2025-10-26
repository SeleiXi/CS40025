#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# 超参数优化启动脚本
# 自动化超参数搜索流程，支持网格搜索和贝叶斯优化

set -euo pipefail

# 脚本信息
SCRIPT_NAME="超参数优化启动器"
SCRIPT_VERSION="1.0.0"
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
    -m, --model PATH        指定模型检查点路径 (必需)
    -d, --data-dir DIR      指定数据目录
    -o, --output-dir DIR    指定输出目录
    -s, --strategy STRATEGY 指定搜索策略 (grid|bayesian, 默认: grid)
    -i, --iterations NUM    指定最大迭代次数 (默认: 100)
    --thresholds LIST       指定预测阈值列表
    --areas LIST            指定最小对象面积列表
    --kernels LIST          指定形态学核大小列表
    --iters LIST            指定形态学迭代次数列表
    --sizes LIST            指定图像尺寸列表
    --mixed-precision       启用混合精度
    --dry-run              仅显示配置，不执行优化

环境变量:
    CUDA_VISIBLE_DEVICES   指定可见的GPU设备
    PYTHONPATH             指定Python路径

示例:
    # 基本网格搜索
    $0 --model ./experiments/building_segmentation/best_model.pth

    # 贝叶斯优化
    $0 --model ./experiments/building_segmentation/best_model.pth --strategy bayesian --iterations 50

    # 自定义参数空间
    $0 --model ./best_model.pth --thresholds "0.4 0.5 0.6" --areas "64 128 256"

    # 使用配置文件
    $0 --model ./best_model.pth --config ./configs/hyperparameter_config.yaml
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
    
    # 检查必要的Python包
    local required_packages=("torch" "torchvision" "numpy" "pandas" "opencv-python" "albumentations" "matplotlib" "seaborn" "scikit-learn")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python -c "import ${package//-/_}" &> /dev/null; then
            missing_packages+=("${package}")
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        print_warning "缺少以下Python包: ${missing_packages[*]}"
        print_info "请运行: pip install ${missing_packages[*]}"
        exit 1
    fi
    
    # 检查贝叶斯优化包（如果使用贝叶斯策略）
    if [ "${SEARCH_STRATEGY:-grid}" = "bayesian" ]; then
        if ! python -c "import skopt" &> /dev/null; then
            print_warning "贝叶斯优化需要scikit-optimize包"
            print_info "请运行: pip install scikit-optimize"
            print_info "或者使用网格搜索策略: --strategy grid"
            exit 1
        fi
    fi
    
    # 检查CUDA
    if ! python -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
        print_warning "CUDA不可用，将使用CPU进行优化"
    else
        print_info "CUDA可用，将使用GPU加速"
    fi
    
    print_success "依赖检查完成"
}

# 设置环境变量
setup_environment() {
    print_info "配置优化环境..."
    
    # 设置Python环境
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    
    # 设置CUDA环境
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    
    # 设置随机种子
    export RANDOM_SEED="${RANDOM_SEED:-42}"
    
    # 创建输出目录
    if [ -n "${OUTPUT_DIR:-}" ]; then
        mkdir -p "${OUTPUT_DIR}"
        print_info "输出目录: ${OUTPUT_DIR}"
    fi
    
    print_success "环境配置完成"
}

# 验证模型文件
validate_model_file() {
    local model_path="$1"
    
    if [ ! -f "${model_path}" ]; then
        print_error "模型文件不存在: ${model_path}"
        exit 1
    fi
    
    # 检查文件大小
    local file_size=$(stat -c%s "${model_path}" 2>/dev/null || stat -f%z "${model_path}" 2>/dev/null)
    if [ "${file_size}" -lt 1024 ]; then
        print_warning "模型文件大小异常小: ${file_size} bytes"
    fi
    
    print_info "模型文件验证通过: ${model_path}"
}

# 构建优化命令
build_optimization_command() {
    local cmd_args=()
    
    # 必需参数
    cmd_args+=("--model_checkpoint" "${MODEL_CHECKPOINT}")
    
    # 数据路径
    cmd_args+=("--train_csv" "${TRAIN_CSV:-./data_source/train_mask.csv}")
    cmd_args+=("--train_img_dir" "${TRAIN_IMG_DIR:-./data_source/train}")
    cmd_args+=("--test_csv" "${TEST_CSV:-./data_source/test_a_samplesubmit.csv}")
    cmd_args+=("--test_img_dir" "${TEST_IMG_DIR:-./data_source/test_a}")
    
    # 优化配置
    cmd_args+=("--search_strategy" "${SEARCH_STRATEGY:-grid}")
    cmd_args+=("--max_iterations" "${MAX_ITERATIONS:-100}")
    cmd_args+=("--batch_size" "${BATCH_SIZE:-16}")
    cmd_args+=("--num_workers" "${NUM_WORKERS:-4}")
    cmd_args+=("--validation_split" "${VALIDATION_SPLIT:-0.2}")
    
    # 参数空间
    if [ -n "${PREDICTION_THRESHOLDS:-}" ]; then
        cmd_args+=("--prediction_thresholds" ${PREDICTION_THRESHOLDS})
    fi
    if [ -n "${MIN_OBJECT_AREAS:-}" ]; then
        cmd_args+=("--min_object_areas" ${MIN_OBJECT_AREAS})
    fi
    if [ -n "${MORPHOLOGY_KERNEL_SIZES:-}" ]; then
        cmd_args+=("--morphology_kernel_sizes" ${MORPHOLOGY_KERNEL_SIZES})
    fi
    if [ -n "${MORPHOLOGY_ITERATIONS:-}" ]; then
        cmd_args+=("--morphology_iterations" ${MORPHOLOGY_ITERATIONS})
    fi
    if [ -n "${IMAGE_SIZES:-}" ]; then
        cmd_args+=("--image_sizes" ${IMAGE_SIZES})
    fi
    
    # 输出配置
    cmd_args+=("--output_dir" "${OUTPUT_DIR:-./hyperparameter_results}")
    cmd_args+=("--experiment_name" "${EXPERIMENT_NAME:-hyperparameter_optimization}")
    
    # 其他配置
    cmd_args+=("--random_seed" "${RANDOM_SEED:-42}")
    
    if [ "${ENABLE_MIXED_PRECISION:-false}" = "true" ]; then
        cmd_args+=("--enable_mixed_precision")
    fi
    
    echo "${cmd_args[*]}"
}

# 执行优化
run_optimization() {
    local optimization_script="hyperparameter_optimizer.py"
    
    if [ ! -f "${optimization_script}" ]; then
        print_error "优化脚本不存在: ${optimization_script}"
        exit 1
    fi
    
    # 构建命令参数
    local cmd_args=($(build_optimization_command))
    
    print_info "开始超参数优化..."
    print_info "优化脚本: ${optimization_script}"
    print_info "参数: ${cmd_args[*]}"
    
    # 执行优化
    exec python "${optimization_script}" "${cmd_args[@]}"
}

# 显示配置摘要
show_config_summary() {
    print_info "优化配置摘要:"
    echo "  模型检查点: ${MODEL_CHECKPOINT}"
    echo "  搜索策略: ${SEARCH_STRATEGY:-grid}"
    echo "  最大迭代次数: ${MAX_ITERATIONS:-100}"
    echo "  输出目录: ${OUTPUT_DIR:-./hyperparameter_results}"
    echo "  实验名称: ${EXPERIMENT_NAME:-hyperparameter_optimization}"
    echo "  批次大小: ${BATCH_SIZE:-16}"
    echo "  验证集比例: ${VALIDATION_SPLIT:-0.2}"
    echo "  混合精度: ${ENABLE_MIXED_PRECISION:-false}"
    echo "  随机种子: ${RANDOM_SEED:-42}"
    
    if [ -n "${PREDICTION_THRESHOLDS:-}" ]; then
        echo "  预测阈值: ${PREDICTION_THRESHOLDS}"
    fi
    if [ -n "${MIN_OBJECT_AREAS:-}" ]; then
        echo "  最小对象面积: ${MIN_OBJECT_AREAS}"
    fi
    if [ -n "${MORPHOLOGY_KERNEL_SIZES:-}" ]; then
        echo "  形态学核大小: ${MORPHOLOGY_KERNEL_SIZES}"
    fi
    if [ -n "${MORPHOLOGY_ITERATIONS:-}" ]; then
        echo "  形态学迭代次数: ${MORPHOLOGY_ITERATIONS}"
    fi
    if [ -n "${IMAGE_SIZES:-}" ]; then
        echo "  图像尺寸: ${IMAGE_SIZES}"
    fi
}

# 主函数
main() {
    # 默认参数
    local dry_run=false
    
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
            -m|--model)
                export MODEL_CHECKPOINT="$2"
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
            -s|--strategy)
                export SEARCH_STRATEGY="$2"
                shift 2
                ;;
            -i|--iterations)
                export MAX_ITERATIONS="$2"
                shift 2
                ;;
            --thresholds)
                export PREDICTION_THRESHOLDS="$2"
                shift 2
                ;;
            --areas)
                export MIN_OBJECT_AREAS="$2"
                shift 2
                ;;
            --kernels)
                export MORPHOLOGY_KERNEL_SIZES="$2"
                shift 2
                ;;
            --iters)
                export MORPHOLOGY_ITERATIONS="$2"
                shift 2
                ;;
            --sizes)
                export IMAGE_SIZES="$2"
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
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查必需参数
    if [ -z "${MODEL_CHECKPOINT:-}" ]; then
        print_error "必须指定模型检查点路径: --model PATH"
        show_help
        exit 1
    fi
    
    # 显示脚本信息
    print_info "${SCRIPT_NAME} v${SCRIPT_VERSION}"
    
    # 验证模型文件
    validate_model_file "${MODEL_CHECKPOINT}"
    
    # 检查依赖
    check_dependencies
    
    # 设置环境
    setup_environment
    
    # 显示配置摘要
    show_config_summary
    
    # 如果是干运行，只显示配置
    if [ "${dry_run}" = true ]; then
        print_info "干运行模式，不执行实际优化"
        exit 0
    fi
    
    # 执行优化
    run_optimization
}

# 错误处理
trap 'print_error "脚本执行失败，退出码: $?"' ERR

# 执行主函数
main "$@"
