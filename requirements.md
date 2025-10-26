# 课程实践 - Project1：语义分割-地表建筑物识别

  * [cite\_start]**课程名称：** 人工智能前沿探索实践-2025年秋 [cite: 602]
  * [cite\_start]**学院：** 复旦大学计算与智能创新学院 [cite: 610]
  * [cite\_start]**教师：** 陈智能 [cite: 611]
  * [cite\_start]**日期：** 2025-9-29 [cite: 612]

-----

## 免费 GPU 服务器使用教程

本教程指导如何使用 ModelScope 平台提供的免费 GPU 资源。

### [cite\_start]1. 登录与账号关联 [cite: 615]

1.  [cite\_start]**登录 ModelScope：** `https://www.modelscope.cn/` [cite: 616]
2.  [cite\_start]**关联阿里云账号：** 关联阿里云账号以获取免费资源 [cite: 648]。
      * [cite\_start]**注意：** 授权内容需要**全部勾选**才能授权成功 [cite: 648]。

### [cite\_start]2. 选择环境并启动 [cite: 651]

选择合适的开发环境并启动：

| 环境 | 特点 | 注意事项 |
| :--- | :--- | :--- |
| **PAI-DSW** | [cite\_start]为算法开发者量身打造的云端深度学习开发环境 [cite: 655][cite\_start]。内置 JupyterLab、WebIDE 及 Terminal [cite: 655]。 | [cite\_start]可以将数据/代码放到 `/mnt/workspace/` 下进行**长期存放** [cite: 652]。 |
| **阿里云弹性加速计算 EAIS** | - | [cite\_start]除了 `.ipynb` 文件外，**不会自动存储** [cite: 652]。 |

**资源配置示例（GPU 环境 - 方式二）：**

  * [cite\_start]**配置：** 8核 32GB [cite: 669]
  * [cite\_start]**显存：** 24G [cite: 669]
  * [cite\_start]**预装库：** ModelScope Library [cite: 669]

-----

## 任务介绍

### [cite\_start]任务目标 [cite: 673]

[cite\_start]本赛题以计算机视觉为背景，要求选手使用给定的航拍图像训练模型，完成**地表建筑物识别**任务（即**语义分割**） [cite: 675, 609][cite\_start]。目的是利用 AI 从海量的航拍图像中识别地表建筑物 [cite: 674]。

  * [cite\_start]**赛题链接：** 【AI入门系列】地球观察员: 建筑物识别学习赛 [cite: 676, 677]

### [cite\_start]数据集介绍 [cite: 682]

  * [cite\_start]**数据来源：** Inria Aerial Image Labeling，并进行了拆分处理 [cite: 683]。
  * [cite\_start]**任务类型：** 参赛选手需要识别图片中的地表建筑的具体像素位置 [cite: 683]。
  * [cite\_start]**像素类别：** 像素属于 2 类（无建筑物和有建筑物） [cite: 684]。
  * **文件格式：**
      * [cite\_start]**原始图片：** `jpg` 格式 [cite: 684]。
      * [cite\_start]**标签（Mask）：** 采用 **RLE 编码**的字符串 [cite: 684, 687]。
  * **RLE 编码说明：**
      * [cite\_start]RLE（Run-Length Encoding）是一种压缩格式 [cite: 687]。
      * [cite\_start]通过记录“连续相同值的起始位置 + 长度”来表示二值掩码 [cite: 687]。
      * [cite\_start]**示例：** `69358 27` 表示从一维掩码的第 69358 个位置开始，有 27 个连续的建筑物像素 [cite: 687, 686]。

### 数据集文件列表及下载链接

[cite\_start]建议使用 `wget` 命令下载完整数据集 [cite: 692, 691]。

| FileName | Size | 含义 | Link |
| :--- | :--- | :--- | :--- |
| `test_a.zip` | 314.49MB | [cite\_start]测试集 A 榜图片 [cite: 685] | [cite\_start]`http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531872/%E5%9C%B0%E8%A1%A8%E5%BB%BA%E7%AD%91%E7%89%A9%E8%AF%86%E5%88%AB/test a.zip` [cite: 693] |
| `test_a_samplesubmit.csv` | 46.39KB | [cite\_start]测试集 A 榜提交样例 [cite: 685] | [cite\_start]`http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531872/%E5%9C%B0%E8%A1%A8%E5%BB%BA%E7%AD%91%E7%89%A9%E8%AF%86%E5%88%AB/test a samplesubmit.csv` [cite: 693] |
| `train.zip` | 3.68GB | [cite\_start]训练集图片 [cite: 685] | [cite\_start]`http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531872/%E5%9C%B0%E8%A1%A8%E5%BB%BA%E7%AD%91%E7%89%A9%E8%AF%86%E5%88%AB/train.zip` [cite: 693] |
| `train_mask.csv.zip` | 97.52MB | [cite\_start]训练集图片标注 [cite: 685] | [cite\_start]`http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531872/%E5%9C%B0%E8%A1%A8%E5%BB%BA%E7%AD%91%E7%89%A9%E8%AF%86%E5%88%AB/train mask.csv.zip` [cite: 693] |

-----

## Baseline 方案（部分代码）

### [cite\_start]2. RLE 与图片之间的转换 [cite: 697]

[cite\_start]将掩码图像编码为 RLE 格式 [cite: 709] [cite\_start]和将 RLE 格式解码为掩码图像 [cite: 733] 的 Python 函数。

**RLE 编码 (RLE Encode)**

```python
def rle_encode(im):
    # [cite_start]将二值图像展平，order='F' (Fortran order/列优先) [cite: 702]
    pixels = im.flatten(order='F')
    # [cite_start]在首尾添加 0 作为边界 [cite: 704]
    pixels = np.concatenate([[0], pixels, [0]])
    # [cite_start]找到所有值发生变化的位置 [cite: 705]
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # [cite_start]runs 中的数字用空格连接成字符串 [cite: 712, 713]
    return " ".join(str(x) for x in runs)
```

**RLE 解码 (RLE Decode)**

```python
def rle_decode(mask_rle, shape=(512, 512)):
    # [cite_start]如果是空字符串，表示图像没有建筑物，返回全 0 的掩码图 [cite: 716, 717, 718]
    if not mask_rle:
        return np.zeros(shape, dtype=np.uint8)
    
    # [cite_start]字符串按空格拆分为列表 [cite: 719]
    s = mask_rle.split() 
    # [cite_start]将字符串列表转换为整型数组，奇数项为起始位置，偶数项为长度 [cite: 721, 722]
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    
    # [cite_start]起始位置调整（基于 1 开始） [cite: 724]
    starts = starts - 1 
    # [cite_start]计算结束位置 [cite: 725, 726]
    ends = starts + lengths 
    
    # [cite_start]生成标记图像 [cite: 727]
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8) # 1D 数组
    for lo, hi in zip(starts, ends):
        [cite_start]img[lo:hi] = 1 # 将指定范围内的像素设置为 1 [cite: 729, 730]
        
    # [cite_start]重塑为 shape 对应的二维数组，order='F' [cite: 731, 732]
    return img.reshape(shape, order='F')
```

### [cite\_start]3. 构建数据集类 `TianChiDataset` [cite: 736]

```python
class TianChiDataset(D.Dataset):
    def __init__(self, paths, rles=None, transform=None, test_mode=False):
        [cite_start]self.paths = paths # 图像路径列表 [cite: 737]
        [cite_start]self.rles = rles if not test_mode else [] # RLE 标签列表 [cite: 737]
        [cite_start]self.transform = transform # 数据增强方法 [cite: 737]
        [cite_start]self.test_mode = test_mode # 是否为测试模式 [cite: 737]
        
        # [cite_start]定义 ToTensor 和标准化操作 [cite: 737]
        self.to_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize([0.625, 8.448, 0.688], [0.131, 0.177, 0.101]),
        ])
        
    def __len__(self):
        [cite_start]return len(self.paths) [cite: 738]
        
    def __getitem__(self, index):
        # [cite_start]读取图像并转换 RGB [cite: 737]
        img = cv2.imread(self.paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if not self.test_mode:
            # [cite_start]训练模式: 加载 mask 并进行增强 [cite: 737]
            mask = rle_decode(self.rles[index])
            augments = self.transform(image=img, mask=mask)
            return self.to_tensor(augments['image']), augments['mask'][None]
        else:
            # [cite_start]测试模式: 仅返回图像 [cite: 737]
            return self.to_tensor(img),
```

### [cite\_start]4. 定义模型 (FCN-ResNet50) [cite: 741]

使用 PyTorch 预训练模型，并修改输出层以适应二分类任务。

```python
def get_model():
    # [cite_start]初始化模型（FCN-ResNet50 分割模型） [cite: 743, 745]
    model = torchvision.models.segmentation.fcn_resnet50(weights=None)
    # [cite_start]修改输出层为 1 个通道（二分类） [cite: 747]
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    return model.to(DEVICE)
```

### [cite\_start]5. 定义损失函数（组合损失） [cite: 752]

[cite\_start]采用组合损失函数，加权融合 **BCE 损失**和 **Dice 损失**，兼顾像素级分类准确性和区域级分割完整性 [cite: 755]。

  * [cite\_start]**Dice Coefficient 公式：** $\frac{2*|X\cap Y|}{|X|+|Y|}$ [cite: 757]

**SoftDiceLoss 实现**

```python
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super().__init__()
        # [cite_start]平滑因子，防止分母为 0 导致的计算错误 [cite: 761]
        self.smooth = smooth
        # [cite_start]对高度(H)和宽度(W)求和 [cite: 762]
        self.dims = dims
        
    def forward(self, x, y):
        # [cite_start]True Positive (TP): 预测和真实是 1 的区域 [cite: 764]
        t_p = (x * y).sum(self.dims)
        # [cite_start]False Positive (FP): 预测是 1 真实是 0 的区域 [cite: 766]
        f_p = (x * (1 - y)).sum(self.dims)
        # [cite_start]False Negative (FN): 预测是 0 真实是 1 的区域 [cite: 768]
        f_n = ((1 - x) * y).sum(self.dims)

        # [cite_start]Dice 系数计算 [cite: 770]
        dc = (2 * t_p + self.smooth) / (2 * t_p + f_p + f_n + self.smooth)
        
        # [cite_start]返回 Dice 损失 (1 - Dice Coefficient) [cite: 775]
        return 1 - dc.mean()
```

**组合损失函数 (80% BCE + 20% Dice)**

```python
def get_loss_fn():
    # [cite_start]初始化 BCE 与 Dice 损失 [cite: 777, 778]
    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = SoftDiceLoss()
    
    # [cite_start]加权融合两种损失函数 [cite: 781]
    def loss_fn(y_pred, y_true):
        return 0.8 * bce_fn(y_pred, y_true) + 0.2 * dice_fn(y_pred.sigmoid(), y_true)
    
    return loss_fn
```

-----

## 实验要求与评分标准

### [cite\_start]实验要求 [cite: 596, 600]

1.  [cite\_start]**提交结果：** 官网提交测试集结果进行评测，实验结果**可复现**（切忌手动修改预测结果） [cite: 600]。
2.  [cite\_start]**提交内容：** 提交**完整实现代码**（不含数据）+ **方案报告**（附结果截图）至 elearning [cite: 600]。
      * [cite\_start]**命名格式：** `学号_姓名_PJ1.zip` [cite: 600]。
3.  [cite\_start]**截止日期：** 2025年10月26日 23:59 [cite: 600]。

### [cite\_start]评分标准（总分 20 分） [cite: 600]

| 评分项 | 分值 | 细则 |
| :--- | :--- | :--- |
| **1. 官网提交结果评测** | 10分 | 超过 0.70 得 2 分；超过 0.75 得 4 分；超过 0.80 得 6 分；超过 0.85 得 8 分；超过 0.90 得 10 分。 |
| **2. 方案报告（基础部分）** | 6分 | 介绍实现方案，包括运行环境说明、模型设计、损失函数设计等。 |
| **3. 方案报告（创新/改进部分）** | 6分 | 介绍涨点所做的数据增强、调参等工作；介绍实现方案的创新性，或在开源方案基础上做的修改；介绍实验中遇到的困难及对应的解决方案。 |

[cite\_start]**最后得分：** $\min(20, S1+S2+S3)$ [cite: 600]。

### [cite\_start]改进思路 [cite: 596]

1.  **模型与参数改进：** 选用更深更先进的 CNN 网络或其它模型；调整数据增强方案（可增加空间变换、像素变换、混合增强，覆盖更多真实场景变异）；调整损失函数；调整超参数等。
2.  **数据优化：** 针对数据集，进行数据预处理优化和后处理优化（分割结果修正），例如进行图像噪声去除、图像标准化等。
3.  **模型集成：** 采用模型集成方案，综合考虑不同模型的预测结果，对概率分布或最终预测结果等进行加权投票。

### [cite\_start]参考资源 [cite: 596]

  * 论坛里有一些高分实现方案可以直接参考，直接复现就能得到不错的结果。
  * **引用要求：** 若参考了论坛里的实现，要求必须在报告中引用链接，并要求在复现其结果的基础上**有一定程度的修改**（变好或变差都行）。
  * [cite\_start]**PyTorch UNet 实现：** `milesial/Pytorch-UNet` [cite: 600]
  * [cite\_start]**PyTorch SegNet 实现：** 选择图像分割任务中表现优异的、且有完整训练代码的模型，将数据处理成其要求格式，重新训练或微调模型，例如 `say4n/pytorch-segnet` [cite: 600]