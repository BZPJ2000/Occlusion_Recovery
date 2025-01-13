import numpy as np
import math

import torch


def PSNR(img1, img2):
    """
    计算两张图像之间的峰值信噪比（PSNR）。

    参数:
        img1: 第一张图像 (PyTorch张量或Numpy数组)
        img2: 第二张图像 (PyTorch张量或Numpy数组)

    返回:
        psnr_value: PSNR值，单位为dB
    """
    # -------------------------------------------- #
    #              数据转换与检查                   #
    # -------------------------------------------- #
    # 如果输入是PyTorch张量，将其转换为Numpy数组
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # 确保两张图像形状一致
    if img1.shape != img2.shape:
        raise ValueError(f"输入图像的形状不一致: img1.shape={img1.shape}, img2.shape={img2.shape}")

    # 检查图像是否已归一化到 [0, 1]
    if img1.max() > 1.0 or img2.max() > 1.0:
        raise ValueError("输入图像未归一化，请确保像素值范围在 [0, 1] 之间")

    # -------------------------------------------- #
    #                  MSE计算                     #
    # -------------------------------------------- #
    mse = np.mean((img1 - img2) ** 2)  # 计算均方误差（MSE）
    if mse == 0:
        return float("inf")  # 如果MSE为0，返回无限大的PSNR值（两图完全一致）

    # -------------------------------------------- #
    #                PSNR计算公式                  #
    # -------------------------------------------- #
    PIXEL_MAX = 1.0  # 图像最大像素值（归一化后为1.0）
    psnr_value = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr_value
