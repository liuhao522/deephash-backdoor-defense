import os
import torch
import torchvision.transforms as transforms
from torch.autograd import Function
import cv2
import numpy as np
import torch.nn as nn
from torchvision.models import resnet50

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature = None
        self.gradient = None
        self.model.eval()
        self.hook_handles = []

        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature = output

        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]

        for module in self.model.named_modules():
            if module[0] == self.target_layer:
                self.hook_handles.append(module[1].register_forward_hook(forward_hook))
                self.hook_handles.append(module[1].register_backward_hook(backward_hook))

    def forward(self, input):
        return self.model(input)

    def backward(self, output):
        self.model.zero_grad()
        output.backward(gradient=torch.ones_like(output))

    def generate(self, input, target_class):
        output = self.forward(input)
        self.backward(output)
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-10)

        cam = cam.squeeze(0).cpu().detach().numpy()
        return cam

def overlay_cam(image, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image) / 255
    cam = cam / np.max(cam)
    return (np.uint8(255 * cam), heatmap)

# 加载ResNet-50模型
model = resnet50(pretrained=True)
model = model.eval()
# 选择目标层的名称，通常是layer4或layer3
target_layer = 'layer4'
gradcam = GradCAM(model, target_layer)

# 指定输入文件夹路径
input_folder = 'D:/deephash_original/cifar10datasclass/9'

# 指定输出文件夹路径
output_folder = 'D:/deephash_original/dataset/cifar10-gradcam'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义数据转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 遍历文件夹中的图像文件
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 读取图像
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(0)

        # 生成Grad-CAM
        target_class = 9  # 替换为你的目标类别
        cam = gradcam.generate(image, target_class)

        # 叠加Grad-CAM到原始图像
        cam_image, _ = overlay_cam(image.squeeze(0).numpy().transpose((1, 2, 0)), cam)

        # 保存Grad-CAM图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cam_image)

print("Grad-CAM images saved in:", output_folder)
