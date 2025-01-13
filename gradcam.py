import torchvision.models as models
import torch
from torch.nn import functional as F


# -------------------------------------------- #
#                Grad-CAM 实现                 #
# -------------------------------------------- #

class GradCAM:
    """
    Grad-CAM 实现:
    基于预训练模型生成输入图像的 Grad-CAM 注意力热力图。

    参数:
        model: 用于生成 Grad-CAM 的神经网络模型
        target_layer: 模型中的目标层，用于提取特征图

    方法:
        generate: 输入图像后生成 Grad-CAM 热力图
    """

    def __init__(self, model, target_layer):
        """
        初始化 Grad-CAM 类。

        参数:
            model: 预训练模型
            target_layer: 用于提取特征图的目标层 (例如模型的某一卷积层)
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None  # 用于存储目标层的激活值
        self.hook_handles = []  # 用于存储钩子
        self._register_hooks()  # 注册钩子以捕获目标层的激活值

    def _register_hooks(self):
        """
        为目标层注册前向传播钩子，用于捕获激活值。
        """

        def forward_hook(module, input, output):
            """
            前向传播钩子，用于捕获目标层的输出激活值。
            """
            self.activations = output

        # 注册前向钩子
        handle = self.target_layer.register_forward_hook(forward_hook)
        self.hook_handles.append(handle)

    def _remove_hooks(self):
        """
        移除所有注册的钩子，防止内存泄漏。
        """
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_image):
        """
        生成 Grad-CAM 热力图。

        参数:
            input_image: 输入图像 (张量, B×C×H×W)

        返回:
            cam: Grad-CAM 热力图 (张量, H×W)
        """
        # 确保模型处于评估模式
        self.model.eval()

        # 前向传播，生成模型输出
        output = self.model(input_image)

        # 移除钩子，防止后续操作出错
        self._remove_hooks()

        # 使用激活值生成注意力图
        # 对激活值进行全局平均池化以生成权重
        weights = F.adaptive_avg_pool2d(self.activations, 1)  # 权重大小: B×C×1×1
        # 计算 Grad-CAM
        cam = torch.sum(weights * self.activations, dim=1).squeeze()  # 加权求和后去除通道维度
        cam = F.relu(cam)  # ReLU 去除负值
        cam = cam - cam.min()  # 最小值归零
        cam = cam / cam.max()  # 归一化到 [0, 1]
        return cam.detach()


# -------------------------------------------- #
#                  测试示例                    #
# -------------------------------------------- #

if __name__ == "__main__":
    # 加载预训练的 ResNet 模型
    model = models.resnet50(pretrained=True)

    # 选择最后一个卷积层作为目标层
    target_layer = model.layer4[-1]

    # 初始化 Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # 创建一个随机输入图像 (1×3×224×224)
    input_image = torch.rand(1, 3, 224, 224)

    # 生成 Grad-CAM 热力图
    cam = grad_cam.generate(input_image)

    # 打印结果
    print("Grad-CAM 生成完成，热力图大小:", cam.shape)
