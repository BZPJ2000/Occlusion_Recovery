import argparse
import csv
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torch import Tensor
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from losses import *
from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from metrics import *
import matplotlib.pyplot as plt

# -------------------------------------------- #
#                参数设置与目录创建             #
# -------------------------------------------- #

os.makedirs("out_images", exist_ok=True)  # 输出图像保存目录
os.makedirs("saved_models", exist_ok=True)  # 模型保存目录

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="开始训练的epoch (大于0将读取之前保存的模型继续训练)")
parser.add_argument("--n_epochs", type=int, default=201, help="训练的epoch数量")
parser.add_argument("--batch_size", type=int, default=2, help="每次训练的batch大小")
parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
parser.add_argument("--b1", type=float, default=0.5, help="Adam优化器的b1参数")
parser.add_argument("--b2", type=float, default=0.999, help="Adam优化器的b2参数")
parser.add_argument("--n_cpu", type=int, default=2, help="数据加载时使用的CPU线程数")
parser.add_argument("--hr_height", type=int, default=128, help="输入高分辨率图像的高度")
parser.add_argument("--hr_width", type=int, default=128, help="输入高分辨率图像的宽度")
parser.add_argument("--channels", type=int, default=3, help="输入图像的通道数 (默认为3，RGB图像)")
parser.add_argument("--sample_interval", type=int, default=200, help="保存输出图像的batch间隔")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="保存模型的epoch间隔")
parser.add_argument("--weight_decay", type=float, default=0, help="权重衰减")
parser.add_argument("--momentum", type=float, default=0.9, help="动量参数")
parser.add_argument("--optimizer", type=str, default="Adam", help="选择优化器 (Adam, AdamW, RMSprop, SGD)")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout概率")
opt = parser.parse_args()

# 检测是否可用GPU
cuda = torch.cuda.is_available()

# 创建保存结果的目录
os.makedirs("results", exist_ok=True)

# -------------------------------------------- #
#                 结果保存函数                  #
# -------------------------------------------- #

def save_results(lr, batch_size, n_epochs, b1, b2, weight_decay, momentum, optimizer, dropout, psnr, g_loss, d_loss):
    """
    保存超参数及训练结果到 CSV 文件中
    """
    file_exists = os.path.isfile('results/hyperparameter_results.csv')
    with open('results/hyperparameter_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # 写入标题行
            writer.writerow(["Learning Rate", "Batch Size", "Epochs", "b1", "b2", "Weight Decay", "Momentum", "Optimizer", "Dropout", "PSNR", "G Loss", "D Loss"])
        # 写入训练结果
        writer.writerow([lr, batch_size, n_epochs, b1, b2, weight_decay, momentum, optimizer, dropout, np.mean(psnr), np.mean(g_loss), np.mean(d_loss)])

# -------------------------------------------- #
#               模型与优化器初始化              #
# -------------------------------------------- #

# 初始化生成器与判别器
generator = GeneratorResNet(opt.hr_height)
discriminator = Discriminator(input_shape=(opt.channels, opt.hr_height, opt.hr_width))
feature_extractor = FeatureExtractor()
attention_map = AttentionMap()

if cuda:
    generator.cuda()
    discriminator.cuda()
    feature_extractor.cuda()
    attention_map.cuda()

# 根据用户选择的优化器初始化
if opt.optimizer == "Adam":
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
elif opt.optimizer == "AdamW":
    optimizer_G = torch.optim.AdamW(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
elif opt.optimizer == "RMSprop":
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)
elif opt.optimizer == "SGD":
    optimizer_G = torch.optim.SGD(generator.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

# 定义损失函数
criterion_content = torch.nn.L1Loss()  # 内容损失
msssim = MSSSIMLoss()  # 多尺度结构相似性损失

# 将特征提取器设置为评估模式
feature_extractor.eval()
attention_map.eval()

# 加载预训练模型（如果指定了起始epoch）
if opt.epoch != 0:
    device = torch.device('cuda:0' if cuda else 'cpu')
    generator.load_state_dict(torch.load("saved_models/generator2_%d.pth" % opt.epoch, map_location=device))
    discriminator.load_state_dict(torch.load("saved_models/discriminator2_%d.pth" % opt.epoch, map_location=device))

# 数据加载器
hr_shape = (opt.hr_height, opt.hr_width)
dataloader = DataLoader(
    ImageDataset(image_root="data/image", mask_root="data/mask", load_size=hr_shape, mode="train"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# -------------------------------------------- #
#               训练主循环逻辑                  #
# -------------------------------------------- #

if __name__ == '__main__':
    d_loss = []  # 判别器损失
    g_loss = []  # 生成器损失
    psnr = []  # 峰值信噪比

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, (imgs, masks) in enumerate(dataloader):
            # 模型输入
            imgs = Variable(imgs.type(Tensor))
            masks = Variable(masks.type(Tensor))
            input_images = imgs * masks

            # 标签
            valid = Variable(Tensor(np.ones((input_images.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((input_images.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  训练生成器
            # ------------------
            optimizer_G.zero_grad()
            output = generator(input_images)
            comp = imgs * masks + output * (1 - masks)

            # 损失计算
            gen_features = feature_extractor(comp)
            real_features = feature_extractor(imgs)
            loss_content = criterion_content(gen_features, real_features.detach())
            ms = msssim(comp, imgs)
            loss_G = loss_content + ms
            loss_G.backward(retain_graph=True)
            optimizer_G.step()

            g_loss.append(loss_G.cpu().item())
            psnr.append(PSNR(comp, imgs))

            # ---------------------
            #  训练判别器
            # ---------------------
            optimizer_D.zero_grad()
            loss_real = torch.mean(discriminator(imgs))
            loss_fake = torch.mean(discriminator(comp.detach()))
            loss_D = -(loss_real - loss_fake)
            loss_D.backward()
            optimizer_D.step()

            d_loss.append(loss_D.cpu().item())

            # 打印训练日志
            sys.stdout.write(
                "[Epoch:%d/%d] [Batch:%d/%d] [D loss:%f] [G loss:%f]\n"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
            )

            # 保存图像与模型
            if epoch % opt.checkpoint_interval == 0:
                torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
                torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
