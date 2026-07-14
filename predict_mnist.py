# -*- coding:utf-8 -*- 
# author:zhangning
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50
import os

# 加载预训练的ResNet-50模型
model_state_dict = torch.load('save/resnet/resnet50_mnist.pt')

model = resnet50()
num_classes = 10
model.fc = nn.Linear(2048, num_classes)
model.load_state_dict(model_state_dict)



# 图片预处理
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(3),  # 将图像转换为3通道
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# 加载图片
image_path = './mnistclass/8/1758-label-8.png'  # 替换为你自己的图片路径
image = Image.open(image_path).convert('RGB')
img = data_transforms(image).unsqueeze(0)
image_name = os.path.basename(image_path)

# 将模型设为评估模式
model.eval()

# 进行推断
with torch.no_grad():
    output = model(img)

# 获取预测结果
_, predicted_idx = torch.max(output, 1)
predicted_label = predicted_idx.item()

print("Predicted Label:", predicted_label)
