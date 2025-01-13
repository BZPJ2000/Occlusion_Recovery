import torch
import torch.nn as nn
import torch.nn.functional as F
from cswin import CSWinBlock

# -------------------------------------------- #
#               辅助函数定义                    #
# -------------------------------------------- #

def Conv3x3BNReLU(in_channels, out_channels, stride, groups=1):
    """
    3x3卷积 + BN + ReLU 激活
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                  groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def Conv1x1BNReLU(in_channels, out_channels):
    """
    1x1卷积 + BN + ReLU 激活
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def Conv1x1BN(in_channels, out_channels):
    """
    1x1卷积 + BN
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels)
    )


# -------------------------------------------- #
#               基础模块定义                    #
# -------------------------------------------- #

class DoubleConv(nn.Module):
    """
    双层卷积模块: (3x3卷积 + BN + ReLU) * 2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv3x3BNReLU(in_channels, out_channels, stride=1),
            Conv3x3BNReLU(out_channels, out_channels, stride=1)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    """
    下采样模块: 最大池化 + 双层卷积
    """

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=stride)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.pool(self.double_conv(x))


class UpConv(nn.Module):
    """
    上采样模块: 上采样 + 通道裁剪/调整 + 双层卷积
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.reduce = Conv1x1BNReLU(in_channels, in_channels // 2)
        # 判断使用双线性插值还是反卷积
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):  # x1为上采样结果，x2为跳跃连接
        x1 = self.up(self.reduce(x1))  # 通道减少并上采样
        _, _, height1, width1 = x1.size()
        _, _, height2, width2 = x2.size()

        # 补齐尺寸以进行特征拼接
        diffY = height2 - height1
        diffX = width2 - width1
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # 特征拼接
        return self.conv(x)


# -------------------------------------------- #
#                  UNet模型定义                #
# -------------------------------------------- #

class UNet(nn.Module):
    """
    UNet网络结构:
    由多层下采样、上采样以及跳跃连接组成。
    """

    def __init__(self, in_channels=3, num_classes=3):
        """
        参数:
            in_channels: 输入图像的通道数（默认为3，即RGB图像）。
            num_classes: 输出通道数（默认为3）。
        """
        super(UNet, self).__init__()
        bilinear = True  # 是否使用双线性插值

        # 定义下采样路径
        self.conv = DoubleConv(in_channels, 32)
        self.down1 = DownConv(32, 64)
        self.down2 = DownConv(64, 128)
        self.down3 = DownConv(128, 256)
        self.down4 = DownConv(256, 512)

        # 中间注意力模块
        self.atten = CSWinBlock(dim=512, num_heads=2, reso=8, split_size=4)

        # 定义上采样路径
        self.up1 = UpConv(512, 256, bilinear)
        self.up2 = UpConv(256, 128, bilinear)
        self.up3 = UpConv(128, 64, bilinear)
        self.up4 = UpConv(64, 32, bilinear)

        # 最终输出层
        self.outconv = nn.Conv2d(32, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播过程:
        输入经过下采样路径、中间注意力模块、上采样路径并最终输出预测结果。
        """
        # 下采样路径
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 中间注意力模块
        B, C, H, W = x5.size()
        x5 = x5.permute(0, 2, 3, 1).view(B, H * W, C)  # 调整为多头注意力输入格式
        x5 = self.atten(x5)  # 应用CSWinBlock注意力模块
        x5 = x5.view(B, H, W, C).permute(0, 3, 1, 2)

        # 上采样路径
        xx = self.up1(x5, x4)
        xx = self.up2(xx, x3)
        xx = self.up3(xx, x2)
        xx = self.up4(xx, x1)

        # 输出层
        outputs = self.outconv(xx)
        outputs = self.sigmoid(outputs)  # 使用sigmoid激活函数
        return outputs


# -------------------------------------------- #
#                  测试模型                    #
# -------------------------------------------- #

if __name__ == '__main__':
    from torchkeras import summary  # 用于查看模型结构

    # 打印UNet的网络结构
    summary(UNet(3), input_shape=(3, 128, 128))
