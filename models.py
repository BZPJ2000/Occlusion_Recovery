import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
from UNet import *
from gradcam import *
from cswin import CSWinBlock


# -------------------------------------------- #
#               自注意力图模块                  #
# -------------------------------------------- #

class AttentionMap(nn.Module):
    """
    自注意力图模块：
    使用预训练的VGG19模型和Grad-CAM生成输入图像的注意力热力图。
    """
    def __init__(self):
        super(AttentionMap, self).__init__()
        # 加载预训练的 VGG19 模型
        self.vgg19_model = vgg19(pretrained=True)
        target_layer = self.vgg19_model.features[-2]  # VGG19 的第36层
        self.grad_cam = GradCAM(self.vgg19_model, target_layer)

    def forward(self, img):
        """
        前向传播，生成图像的注意力热力图。
        """
        output = self.vgg19_model(img)
        predicted_class = output.argmax(dim=1)  # 获取预测的类别索引
        heatmaps = []

        # 逐张生成注意力图
        for i in range(len(predicted_class)):
            attention_map = self.grad_cam.generate(img[i].unsqueeze(0))  # 单张图像生成
            heatmaps.append(attention_map)

        # 将注意力图合并为一个张量
        heatmaps = torch.cat(heatmaps, dim=0)
        return heatmaps


# -------------------------------------------- #
#               感知损失模块                   #
# -------------------------------------------- #

class FeatureExtractor(nn.Module):
    """
    感知损失模块：
    使用VGG19的前几层作为特征提取器，用于计算感知损失。
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 加载预训练的 VGG19 模型
        vgg19_model = vgg19(pretrained=True)
        # 提取前18层作为特征提取器
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        """
        前向传播，提取输入图像的特征。
        """
        return self.feature_extractor(img)


# -------------------------------------------- #
#               生成器模块（使用UNet）           #
# -------------------------------------------- #

class GeneratorResNet(nn.Module):
    """
    生成器模块：
    使用UNet模型作为生成器，用于将输入的遮挡图像恢复为清晰图像。
    """
    def __init__(self, img_size):
        super(GeneratorResNet, self).__init__()
        # 使用UNet结构作为生成器
        self.model = UNet(3, 3)

    def forward(self, x):
        """
        前向传播，生成修复后的图像。
        """
        out = self.model(x)
        return out


# -------------------------------------------- #
#                判别器模块                    #
# -------------------------------------------- #

class Discriminator(nn.Module):
    """
    判别器模块：
    用于判断输入图像是否为真实图像或生成图像。
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        # 输入图像形状
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        # 判别器卷积块
        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))  # 首块不使用归一化
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 256]):  # 四层卷积
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters
        self.model = nn.Sequential(*layers)

        # 添加自注意力机制
        self.atten = CSWinBlock(dim=256, num_heads=2, reso=8, split_size=4)  # CSWinBlock 注意力模块
        self.last_layer = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)  # 输出通道为1

    def forward(self, img):
        """
        前向传播，计算图像的判别结果。
        """
        # 卷积层
        x = self.model(img)

        # 注意力机制
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1)  # 调整维度
        x = x.view(B, H * W, C)  # 调整为多头注意力输入格式
        x = self.atten(x)  # 注意力计算
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        # 输出判别结果
        x = self.last_layer(x)
        return x


# -------------------------------------------- #
#                测试与模型总结                 #
# -------------------------------------------- #

if __name__ == '__main__':
    from torchkeras import summary  # 用于模型结构的可视化

    # 测试生成器模型
    print("生成器模型：")
    summary(GeneratorResNet(img_size=128), input_shape=(3, 128, 128))

    # 测试判别器模型
    print("判别器模型：")
    summary(Discriminator((3, 128, 128)), input_shape=(3, 128, 128))

    # 测试感知损失模块
    print("感知损失模块：")
    summary(FeatureExtractor(), input_shape=(3, 128, 128))
