import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QBuffer, QIODevice
import cv2


import warnings

warnings.filterwarnings("ignore")

import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F
from PIL import Image
import io
from datasets import *
from models import *

# -------------------------------------------- #
#               图像后处理函数                  #
# -------------------------------------------- #

def postprocess(x):
    """
    对生成的张量进行后处理，将值映射到 [0, 1] 范围内。
    """
    x = (x + 1.) / 2.
    x.clamp_(0, 1)
    return x

# -------------------------------------------- #
#               修复模型类定义                  #
# -------------------------------------------- #

class CTSDG:
    """
    使用预训练的生成器模型进行图像修复。
    """
    def __init__(self):
        super(CTSDG, self).__init__()

        root = sys.path[0]  # 获取项目根目录
        self.result_root = os.path.join(root, 'result')  # 保存修复后的对比图片的路径
        os.makedirs(self.result_root, exist_ok=True)

        # 加载训练好的模型
        self.generator = GeneratorResNet(img_size=(128, 128))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.generator.load_state_dict(
            torch.load(os.path.join(root, 'saved_models', 'generator_110.pth'), map_location=self.device))
        self.generator.to(device=self.device)

        # 定义预处理
        load_size = (128, 128)
        self.image_files_transforms = image_transforms(load_size)
        self.mask_files_transforms = mask_transforms(load_size)

    def repair(self, image, cv_mask, save_path):
        # 推理
        with torch.no_grad():
            h, w = image.size
            # 原始图像
            image = self.image_files_transforms(image)

            # mask
            mask = Image.fromarray(cv_mask)
            mask = self.mask_files_transforms(mask)

            threshold = 0.5
            ones = mask >= threshold
            zeros = mask < threshold

            mask.masked_fill_(ones, 1.0)
            mask.masked_fill_(zeros, 0.0)

            input_image = image * mask

            image, mask, input_image = image.to(self.device).unsqueeze(0), mask.to(self.device).unsqueeze(
                0), input_image.to(self.device).unsqueeze(0)

            # 从模糊图像输入生成清晰图像
            output = self.generator(input_image)

            comp = image * mask + output * (1 - mask)  # 合成图

            # 返回大小
            _output_comp = F.interpolate(comp, size=(w, h), mode='bilinear', align_corners=False)
            input_image = F.interpolate(input_image, size=(w, h), mode='bilinear', align_corners=False)
            output_comp = torch.cat([input_image, _output_comp], dim=-1)
            output_comp = postprocess(output_comp)
            save_path = os.path.join(self.result_root, f'{os.path.split(save_path)[-1]}')
            save_image(output_comp, save_path)
            print('图片保存已至', save_path)

            # 后处理
            _output_comp = postprocess(_output_comp)
            return (_output_comp.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # 返回修复后的图像


# Ndarray数组转QPixmap
def cv2_to_pixmap(cv_img):
    height, width, channel = cv_img.shape
    bytes_per_line = channel * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(q_img)


# 将QPixmap转为PIL.Image
def qpixmap_to_pil(qpixmap):
    # 将QPixmap转换为QImage
    qimage = qpixmap.toImage()

    # 准备一个QBuffer来存放QImage的数据
    buffer = QBuffer()
    buffer.open(QIODevice.ReadWrite)
    qimage.save(buffer, "PNG")  # 将QImage保存到buffer，使用PNG格式

    # 将QBuffer的内容读取到一个Python的bytes对象中
    pil_image_data = io.BytesIO(buffer.data())

    # 使用Pillow从字节数据创建图像
    pil_image = Image.open(pil_image_data)

    return pil_image


class DoodleApp(QWidget):
    """
    基于PyQt5的图像涂鸦修复界面。
    """
    def __init__(self):
        super().__init__()
        # 选择的图像
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(256, 256)

        # 涂鸦
        self.doodle_label = QLabel(self)
        self.doodle_label.setFixedSize(256, 256)

        # 未修复的合成图像
        self.combined_label = QLabel(self)
        self.combined_label.setFixedSize(256, 256)

        # 修复后的图像
        self.repair_label = QLabel(self)
        self.repair_label.setFixedSize(256, 256)

        self.image = None
        self.doodle = None
        self.doodle_image = None
        self.combined_image = None
        self.repair = None
        self.Graffiti = []  # 鼠标涂鸦的轨迹
        self.last_point = QPoint()

        # 按钮
        image_select_button = QPushButton("选择原图", self)
        image_select_button.clicked.connect(self.select_image)

        doodle_select_button = QPushButton("选择涂鸦", self)
        doodle_select_button.clicked.connect(self.select_doodle_image)

        draw_button = QPushButton("鼠标涂鸦", self)
        draw_button.clicked.connect(self.draw_doodle)

        combine_button = QPushButton("显示合成效果", self)
        combine_button.clicked.connect(self.show_combined_image)

        repair_button = QPushButton("修复", self)
        repair_button.clicked.connect(self.show_repair)

        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(image_select_button)
        button_layout.addWidget(doodle_select_button)
        button_layout.addWidget(draw_button)
        button_layout.addWidget(combine_button)
        button_layout.addWidget(repair_button)

        # 图像布局
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.doodle_label)
        image_layout.addWidget(self.combined_label)
        image_layout.addWidget(self.repair_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addLayout(image_layout)

        self.setLayout(main_layout)

        # 设置窗口大小为1500x600
        self.setFixedSize(1500, 600)

        self.ctsdg = CTSDG()

    def select_image(self):
        """
        选择原始图像。
        """
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            self.file_path = file_dialog.selectedFiles()[0]
            self.image = QPixmap(self.file_path)
            self.image = self.image.scaled(256, 256, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(self.image)
            # Reset doodle and combined images
            self.doodle = None
            self.doodle_label.clear()
            self.combined_image = None
            self.combined_label.clear()
            self.repair_label.clear()
            self.Graffiti = []

    def select_doodle_image(self):
        """
        选择涂鸦图片作为遮罩。
        """
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            doodle_file_path = file_dialog.selectedFiles()[0]
            self.doodle_image = QPixmap(doodle_file_path)
            self.doodle_image = self.doodle_image.scaled(256, 256, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.doodle_label.setPixmap(self.doodle_image)

    def draw_doodle(self):
        """
        使用鼠标在图像上涂鸦。
        """
        self.Graffiti = []  # 清空涂鸦轨迹
        if self.image is not None:
            self.doodle = self.image.copy()
            self.doodle_label.setPixmap(self.doodle)
            self.doodle_label.mousePressEvent = self.mouse_press_event
            self.doodle_label.mouseMoveEvent = self.mouse_move_event

    def show_combined_image(self):
        """
        显示涂鸦遮罩和原图的合成效果。
        """
        if self.image and (self.doodle or self.doodle_image):
            if self.doodle:
                doodle_pixmap = self.doodle
            else:
                doodle_pixmap = self.doodle_image

            # 将涂鸦图片作为遮罩覆盖到原图上
            image = qpixmap_to_pil(self.image)
            doodle = qpixmap_to_pil(doodle_pixmap)

            # 将涂鸦图片转为遮罩
            doodle_np = np.array(doodle.convert('L'))  # 转换为灰度图
            mask = np.where(doodle_np > 128, 255, 0).astype(np.uint8)  # 转为二值化遮罩

            # 显示未修复的合成图像
            combined_image = cv2.bitwise_and(np.array(image), np.array(image), mask=mask)
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
            combined_pixmap = cv2_to_pixmap(combined_image)
            self.combined_label.setPixmap(combined_pixmap)

    def show_repair(self):
        """
        使用模型对图像进行修复。
        """
        if self.doodle_image is not None:
            # 使用选择的涂鸦图片进行修复
            image = qpixmap_to_pil(self.image)
            doodle = qpixmap_to_pil(self.doodle_image)

            # 将涂鸦图片转换为灰度图
            doodle_np = np.array(doodle.convert('L'))  # 转换为灰度图
            mask = np.where(doodle_np > 128, 255, 0).astype(np.uint8)  # 转换为二值化遮罩

            # 调用修复功能
            repaired_image = self.ctsdg.repair(image, mask, self.file_path)
            repaired_image = cv2.cvtColor(repaired_image, cv2.COLOR_BGR2RGB)
            repaired_pixmap = cv2_to_pixmap(repaired_image)
            self.repair_label.setPixmap(repaired_pixmap)

        elif len(self.Graffiti) > 2:
            # 使用鼠标涂鸦进行修复
            mask = np.ones((256, 256), np.uint8) * 255
            old_p = self.Graffiti[0]
            for new_p in self.Graffiti[1:]:
                thickness = 5  # 线宽
                color = 0  # 颜色
                cv2.line(mask, old_p, new_p, color, thickness)
                old_p = new_p

            image = qpixmap_to_pil(self.image)
            repair = self.ctsdg.repair(image, mask, self.file_path)
            repair = cv2.cvtColor(repair, cv2.COLOR_BGR2RGB)
            repair_pixmap = cv2_to_pixmap(repair)
            self.repair_label.setPixmap(repair_pixmap)

    def mouse_press_event(self, event):
        """
        记录鼠标按下的位置。
        """
        self.last_point = event.pos()

    def mouse_move_event(self, event):
        """
        记录鼠标移动的涂鸦轨迹并绘制。
        """
        if self.doodle:
            painter = QPainter(self.doodle)
            painter.setRenderHint(QPainter.Antialiasing)  # 设置抗锯齿
            pen = QPen()
            pen.setColor(QColor(Qt.black))
            pen.setWidth(5)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.doodle_label.setPixmap(self.doodle)
            self.Graffiti.append((self.last_point.x(), self.last_point.y()))  # 绘制轨迹

    def show_doodle_layer(self):
        if self.doodle is not None:
            self.doodle_label.setPixmap(self.doodle)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    doodle_app = DoodleApp()
    doodle_app.show()
    sys.exit(app.exec_())
