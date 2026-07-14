# -*- coding:utf-8 -*- 
# author:zhangning
# -*- coding:utf-8 -*-
# author:zhangning
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50
import os

# 加载预训练的ResNet-50模型
model_state_dict = torch.load('save/resnet/resnet50_cifar10.pt')

model = resnet50()
num_classes = 10
model.fc = nn.Linear(2048, num_classes)
model.load_state_dict(model_state_dict)



# 图片预处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
label_name = ['airplane', 'automobile', 'brid',
              'cat', 'deer', 'dog', 'frog',
              'horse', 'ship', 'truck']


# 加载图片
image_path = './adv_0.009/4-label-1.png'  # 替换为你自己的图片路径
image = Image.open(image_path)
img = transform(image)
img = img.unsqueeze(0)


# 将模型设为评估模式
model.eval()

# 进行推断
with torch.no_grad():
    output = model(img)

# 获取预测结果
_, predicted_idx = torch.max(output, 1)
predicted_label = predicted_idx.item()
print("Predicted Label:", predicted_label)
print("Predicted Label:", label_name[predicted_label])
