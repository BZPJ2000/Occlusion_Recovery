import random
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

# -------------------------------------------- #
#               工具函数定义                   #
# -------------------------------------------- #

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
    '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF'
]


def is_image_file(filename):
    """
    检查文件是否是支持的图像格式。
    参数:
        filename: 文件名 (字符串)
    返回:
        bool: 如果是图像文件，则返回 True
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    """
    收集目录中的所有图像文件路径。
    参数:
        dir: 数据集目录路径 (字符串)
        max_dataset_size: 最大允许的图像数量 (默认为无限大)
    返回:
        images: 图像文件路径列表
    """
    images = []
    if not os.path.isdir(dir):
        raise ValueError(f'{dir} 不是一个有效的目录！')

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    images = sorted(images)
    return images[:min(max_dataset_size, len(images))]


def image_transforms(load_size):
    """
    定义图像的预处理流程。
    参数:
        load_size: 图像的目标大小 (宽, 高)
    返回:
        transforms.Compose: 图像预处理操作
    """
    return transforms.Compose([
        transforms.Resize(size=load_size, interpolation=Image.BILINEAR),  # 调整大小
        transforms.ToTensor(),  # 转为 PyTorch 张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1, 1]
    ])


def mask_transforms(load_size):
    """
    定义掩码的预处理流程。
    参数:
        load_size: 掩码的目标大小 (宽, 高)
    返回:
        transforms.Compose: 掩码预处理操作
    """
    return transforms.Compose([
        transforms.Resize(size=load_size, interpolation=Image.NEAREST),  # 最近邻插值
        transforms.ToTensor()  # 转为 PyTorch 张量
    ])


# -------------------------------------------- #
#              数据集类定义                     #
# -------------------------------------------- #

class ImageDataset(Dataset):
    """
    图像与掩码数据集类:
    用于加载图像与对应的掩码，支持训练与测试模式。
    """

    def __init__(self, image_root, mask_root, load_size=(128, 128), sigma=2.0, mode='test'):
        """
        初始化数据集。
        参数:
            image_root: 图像数据集路径
            mask_root: 掩码数据集路径
            load_size: 图像和掩码的目标大小 (宽, 高)
            sigma: 掩码处理的标准差 (暂未使用)
            mode: 模式 ('train' 或 'test')
        """
        super(ImageDataset, self).__init__()

        # 加载图像和掩码路径
        self.image_files = make_dataset(dir=image_root)
        self.mask_files = make_dataset(dir=mask_root)

        self.number_image = len(self.image_files)  # 图像数量
        self.number_mask = len(self.mask_files)  # 掩码数量

        self.sigma = sigma
        self.mode = mode
        self.load_size = load_size

        # 定义图像与掩码的预处理操作
        self.image_files_transforms = image_transforms(load_size)
        self.mask_files_transforms = mask_transforms(load_size)

    def __getitem__(self, index):
        """
        获取指定索引的数据样本。
        参数:
            index: 数据索引
        返回:
            image: 经过预处理的图像 (张量)
            mask: 经过预处理的掩码 (张量)
        """
        # 加载图像并应用预处理
        image_path = self.image_files[index % self.number_image]  # 防止索引越界
        image = Image.open(image_path)
        image = self.image_files_transforms(image.convert('RGB'))  # 确保为 RGB 图像

        # 加载掩码并应用预处理
        if self.mode == 'train':
            # 训练模式下随机选择一个掩码
            mask_path = self.mask_files[random.randint(0, self.number_mask - 1)]
        else:
            # 测试模式下按顺序选择掩码
            mask_path = self.mask_files[index % self.number_mask]

        mask = Image.open(mask_path).convert('L')  # 转为灰度图
        mask = self.mask_files_transforms(mask)

        # 二值化掩码
        threshold = 0.5
        mask = (mask >= threshold).float()  # 大于阈值的设置为1，小于的设置为0

        return image, mask

    def __len__(self):
        """
        获取数据集的大小。
        返回:
            int: 数据集中的图像数量
        """
        return self.number_image


# -------------------------------------------- #
#                  测试代码                     #
# -------------------------------------------- #

if __name__ == '__main__':
    # 测试数据集路径 (替换为实际路径)
    image_root = './data/image'
    mask_root = './data/mask'

    # 创建数据集
    dataset = ImageDataset(image_root, mask_root, load_size=(128, 128), mode='train')

    # 测试加载数据
    print(f"数据集中图像数量: {len(dataset)}")
    for i in range(5):  # 测试加载前 5 个样本
        image, mask = dataset[i]
        print(f"样本 {i} - 图像大小: {image.shape}, 掩码大小: {mask.shape}")
