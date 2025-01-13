import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# -------------------------------------------- #
#                Grad-CAM 实现                 #
# -------------------------------------------- #
class GradCAM:
    """
    Grad-CAM 实现:
    使用指定的模型和目标层生成输入图像的 Grad-CAM 注意力图。

    参数:
        model: 预训练模型 (例如 VGG19 或 ResNet 等)
        target_layer: 模型中的目标卷积层，用于提取激活值
    """
    def __init__(self, model, target_layer):
        """
        初始化 Grad-CAM 类。
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None  # 存储目标层的激活值
        self.hook_handles = []  # 存储注册的钩子
        self._register_hooks()  # 注册前向传播钩子

    def _register_hooks(self):
        """
        注册前向传播钩子，捕获目标层的激活值。
        """
        def forward_hook(module, input, output):
            self.activations = output  # 捕获目标层的输出（激活值）

        handle = self.target_layer.register_forward_hook(forward_hook)
        self.hook_handles.append(handle)

    def _remove_hooks(self):
        """
        移除所有注册的钩子，避免内存泄漏。
        """
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_image):
        """
        生成 Grad-CAM 注意力图。

        参数:
            input_image: 输入图像 (张量, B×C×H×W)
        返回:
            cam: Grad-CAM 注意力图 (numpy.ndarray, H×W)
        """
        # 确保模型处于评估模式
        self.model.eval()

        # 前向传播，计算模型输出
        output = self.model(input_image)

        # 移除钩子，防止后续问题
        self._remove_hooks()

        # 使用激活值生成注意力图
        # 对激活值进行全局平均池化，生成权重
        weights = F.adaptive_avg_pool2d(self.activations, 1)  # 权重大小: B×C×1×1

        # 计算 Grad-CAM
        cam = torch.sum(weights * self.activations, dim=1).squeeze()  # 加权求和后去掉通道维度
        cam = F.relu(cam)  # ReLU 去除负值
        cam = cam - cam.min()  # 最小值归零
        cam = cam / cam.max()  # 归一化到 [0, 1]

        return cam.detach().cpu().numpy()  # 转为 numpy 格式，便于可视化


# -------------------------------------------- #
#                图像预处理函数                 #
# -------------------------------------------- #
def preprocess_image(image_path):
    """
    对输入图像进行预处理，使其适配预训练模型。

    参数:
        image_path: 输入图像的文件路径
    返回:
        img_tensor: 预处理后的图像张量 (B×C×H×W)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为 224×224
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    img = Image.open(image_path).convert('RGB')  # 确保为 RGB 格式
    img_tensor = transform(img).unsqueeze(0)  # 添加批次维度
    return img_tensor


# -------------------------------------------- #
#             注意力图生成与可视化              #
# -------------------------------------------- #

def generate_attention_map(image_path, grad_cam):
    """
    使用 Grad-CAM 生成输入图像的注意力图。

    参数:
        image_path: 输入图像的文件路径
        grad_cam: Grad-CAM 对象
    返回:
        attention_map: Grad-CAM 注意力图 (numpy.ndarray, H×W)
    """
    input_image = preprocess_image(image_path)  # 对图像进行预处理
    attention_map = grad_cam.generate(input_image)  # 生成注意力图
    return attention_map


def show_attention_map(image_path, attention_map):
    """
    显示原始图像与叠加注意力图的结果。

    参数:
        image_path: 输入图像的文件路径
        attention_map: Grad-CAM 注意力图 (numpy.ndarray, H×W)
    """
    img = Image.open(image_path).convert('RGB')  # 打开原始图像
    img = np.array(img.resize((224, 224)))  # 调整大小以匹配注意力图

    # 将注意力图转换为热力图格式
    attention_map_resized = np.uint8(255 * attention_map)  # 缩放到 [0, 255]
    heatmap = plt.cm.jet(attention_map_resized)[:, :, :3]  # 使用 Jet 配色方案
    heatmap = np.uint8(heatmap * 255)  # 转换为 RGB 格式
    heatmap_img = Image.fromarray(heatmap).resize((img.shape[1], img.shape[0]))  # 调整热力图大小
    overlay_img = Image.blend(Image.fromarray(img), heatmap_img, alpha=0.6)  # 将热力图叠加到原图上

    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Attention Map Overlay")
    plt.imshow(overlay_img)
    plt.axis('off')
    plt.show()


# -------------------------------------------- #
#                  示例用法                     #
# -------------------------------------------- #
if __name__ == "__main__":
    # 加载预训练的 VGG19 模型
    vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

    # 选择目标层 (VGG19 最后一层卷积层)
    target_layer = vgg19.features[-2]

    # 创建 Grad-CAM 对象
    grad_cam = GradCAM(vgg19, target_layer)

    # 输入图像路径
    image_path = r'./image/000002.jpg'  # 替换为你的图像路径

    # 生成注意力图
    attention_map = generate_attention_map(image_path, grad_cam)

    # 显示原始图像与叠加的注意力图
    show_attention_map(image_path, attention_map)
