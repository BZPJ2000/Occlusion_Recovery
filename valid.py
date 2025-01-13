import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import GeneratorResNet
from datasets import ImageDataset
from metrics import PSNR

# -------------------------------------------- #
#               工具函数定义                    #
# -------------------------------------------- #

def postprocess(x):
    """
    后处理函数，将张量的像素值调整到 [0, 1] 范围。
    """
    x = (x + 1.) / 2.  # 将像素值从 [-1, 1] 转换到 [0, 1]
    x.clamp_(0, 1)  # 将像素值限制在 [0, 1] 范围内
    return x


def validate_model(generator, dataloader, device):
    """
    验证生成器模型的性能，并计算 PSNR（峰值信噪比）。

    参数:
        generator: 生成器模型。
        dataloader: 数据加载器，提供验证数据集。
        device: 使用的计算设备（CPU 或 GPU）。

    返回:
        avg_psnr: 数据集上的平均 PSNR 值。
    """
    generator.eval()  # 设置模型为评估模式
    psnr_values = []  # 存储每张图像的 PSNR 值

    with torch.no_grad():  # 禁用梯度计算，加速推理
        for imgs, masks in tqdm(dataloader, desc="验证中"):
            # 将图像和掩码移动到指定设备
            imgs = imgs.to(device)
            masks = masks.to(device)

            # 生成输入的遮挡图像
            input_images = imgs * masks

            # 模型推理：生成修复后的图像
            output = generator(input_images)

            # 合成修复图像
            comp = imgs * masks + output * (1 - masks)

            # 后处理并计算 PSNR
            comp = postprocess(comp)
            imgs = postprocess(imgs)
            psnr_value = PSNR(comp, imgs)
            psnr_values.append(psnr_value)

    # 计算平均 PSNR
    avg_psnr = np.mean(psnr_values)
    return avg_psnr


# -------------------------------------------- #
#                  主函数入口                   #
# -------------------------------------------- #

if __name__ == "__main__":
    # -------------------------------------------- #
    #            参数解析和设置默认值              #
    # -------------------------------------------- #
    parser = argparse.ArgumentParser(description="验证生成器模型性能")

    # 模型和数据路径相关参数
    parser.add_argument("--model_path", type=str, default="saved_models/generator_110.pth", help="生成器模型的路径")
    parser.add_argument("--data_root", type=str, default="data/image", help="验证数据集的图像路径")
    parser.add_argument("--mask_root", type=str, default="data/mask", help="验证数据集的掩码路径")

    # 验证过程参数
    parser.add_argument("--batch_size", type=int, default=2, help="验证时的批量大小")
    parser.add_argument("--hr_height", type=int, default=128, help="高分辨率图像的高度")
    parser.add_argument("--hr_width", type=int, default=128, help="高分辨率图像的宽度")
    parser.add_argument("--n_cpu", type=int, default=2, help="加载数据时使用的 CPU 线程数")

    opt = parser.parse_args()

    # -------------------------------------------- #
    #               设备与模型设置                 #
    # -------------------------------------------- #
    cuda = torch.cuda.is_available()  # 检查是否支持 GPU
    device = torch.device("cuda" if cuda else "cpu")  # 选择设备

    # 定义高分辨率图像的尺寸
    hr_shape = (opt.hr_height, opt.hr_width)

    # 初始化生成器模型并加载训练好的权重
    generator = GeneratorResNet(opt.hr_height).to(device)
    generator.load_state_dict(torch.load(opt.model_path, map_location=device))

    # -------------------------------------------- #
    #            数据加载器（DataLoader）          #
    # -------------------------------------------- #
    dataloader = DataLoader(
        ImageDataset(image_root=opt.data_root, mask_root=opt.mask_root, load_size=hr_shape, mode="val"),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # -------------------------------------------- #
    #                模型验证过程                  #
    # -------------------------------------------- #
    avg_psnr = validate_model(generator, dataloader, device)

    # -------------------------------------------- #
    #                输出验证结果                  #
    # -------------------------------------------- #
    print(f"验证数据集的平均 PSNR: {avg_psnr:.2f} dB")
