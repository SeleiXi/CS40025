#!/usr/bin/env python3
"""
依赖检查脚本
检查所有必要的Python包是否正确安装
"""

import sys
import importlib
from typing import List, Tuple

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """检查包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, 'not installed'

def main():
    """主函数"""
    # 核心依赖列表 (包名, 导入名)
    core_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('torchaudio', 'torchaudio'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('opencv-python', 'cv2'),
        ('Pillow', 'PIL'),
        ('albumentations', 'albumentations'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('scikit-learn', 'sklearn'),
        ('tqdm', 'tqdm'),
        ('PyYAML', 'yaml'),
        ('tensorboard', 'tensorboard'),
    ]
    
    missing_packages = []
    
    for package, import_name in core_packages:
        success, version = check_package(package, import_name)
        if not success:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"MISSING_PACKAGES: {' '.join(missing_packages)}")
        sys.exit(1)
    else:
        print("ALL_PACKAGES_INSTALLED")
        sys.exit(0)

if __name__ == "__main__":
    main()
