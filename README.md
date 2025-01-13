
# 🌟 **深度图像修复系统** 🌟  
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/框架-PyTorch-red?logo=pytorch)](https://pytorch.org/)  

---

## 🎯 **项目概述**
**深度图像修复系统** 是一个基于深度学习的解决方案，利用 **生成对抗网络（GANs）** 来修复图像中被遮挡或损坏的部分。该项目提供从模型训练到实时图像修复的完整流程，并通过图形化界面（GUI）方便用户进行图像操作。

---

## 🧩 **功能模块**
### 🔧 **1. 模型训练**
- 使用 `train.py` 或 `train_2.py` 进行模型训练。
- 支持 **自定义超参数** 和从保存的断点继续训练。
- 使用了 **感知损失** 和 **MSSSIM 损失** 等自定义损失函数以提高训练性能。

### 🚀 **2. 超参数搜索**
- 使用 `param_search.py` 自动化超参数搜索。
- 自动生成多种参数组合，帮助找到最佳模型参数。

### 🎨 **3. 图像修复 GUI**
- 基于 **PyQt5** 的用户友好图形界面。
- 主要功能：
  - 选择图像。
  - 通过鼠标涂鸦绘制损坏区域，或加载遮罩文件。
  - 点击 **修复** 按钮生成修复后的图像。

### 🔍 **4. 注意力图生成**
- 使用 `demo.py` 基于 Grad-CAM 技术生成注意力图。
- 帮助用户理解模型修复时关注的区域。

---

## 🛠️ **安装方法**

### 1️⃣ 克隆代码仓库：
```bash
git clone https://github.com/BZPJ2000/Occlusion_Recovery.git
cd DeepImageRepair
```

### 2️⃣ 安装依赖：
运行以下命令安装所有必要依赖：
```bash
python install.py
```

### 3️⃣ 硬件支持：
- **推荐使用 GPU** 以加快训练和图像修复过程。
- 支持 Python 3.8+ 和 **PyTorch 1.9+**。

---

## ⚡ **快速开始**

### 🏋️‍♂️ **训练模型**
使用以下命令训练模型：
```bash
python train.py --dataset_path ./data --epochs 200 --batch_size 16 --lr 0.0002
```

从某个保存的断点继续训练：
```bash
python train.py --checkpoint ./checkpoints/generator_epoch_50.pth
```

---

### 🖌️ **启动图像修复界面**
运行以下命令启动基于 PyQt5 的图像修复 GUI：
```bash
python image_doodle_repair_app.py
```

---

### 🔍 **生成注意力图**
使用 Grad-CAM 生成注意力图：
```bash
python demo.py --image_path ./sample_images/input.jpg --model_path ./checkpoints/generator_110.pth
```

---

## 📁 **项目结构**
```
Occlusion_Recovery/
│
├── datasets/                          # 数据加载和增强模块
├── models/                            # GAN 模型架构
├── saved_models/                      # 保存的模型权重
├── result/                            # 修复后的图像保存路径
│
├── train.py                           # 模型训练脚本
├── param_search.py                    # 超参数搜索脚本
├── demo.py                            # 注意力图生成脚本
├── image_doodle_repair_app.py         # 基于 PyQt5 的图形界面
├── install.py                         # 依赖安装脚本
│
└── README.md                          # 项目文档
```

---


## 📬 **联系方式**
如果您有任何问题或建议，请通过以下方式联系我们：
- GitHub: [BZPJ2000](https://github.com/BZPJ2000)
- 邮箱: 1367713858@qq.com

---

## ⭐ **支持我们**
如果您喜欢本项目，请在 [GitHub](https://github.com/Occlusion_Recovery) 上为我们点个 ⭐！
