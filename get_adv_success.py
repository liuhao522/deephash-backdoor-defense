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
import shutil

# 加载预训练的ResNet-50模型
model_state_dict = torch.load('save/resnet/resnet50_cifar10.pt')

model = resnet50()
num_classes = 10
model.fc = nn.Linear(2048, num_classes)
model.load_state_dict(model_state_dict)



# 图片预处理
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
path = './adv_cifar10_0.3'
path1 = './adv_cifar10_0.3_attack'
files_list = os.listdir(path)
# 遍历列表中的所有文件
for file in files_list:
    image_path = f'./adv_cifar10_0.3/{file}'  # 替换为你自己的图片路径
    # print(file)
    image = Image.open(image_path).convert('RGB')
    img = data_transforms(image).unsqueeze(0)
    # image_name = os.path.basename(image_path)
    e = file[file.rfind('-'):file.rfind('.')]  # print 'A935
    name1 = e[1:]
    name1 = int(name1)
    # print(name1)
    # 将模型设为评估模式
    model.eval()
    # 进行推断
    with torch.no_grad():
        output = model(img)
    # 获取预测结果
    _, predicted_idx = torch.max(output, 1)
    predicted_label = predicted_idx.item()
    if name1 != predicted_label:
        shutil.copy(os.path.join(path, file), os.path.join(path1, file))

# print("Predicted Label:", predicted_label)
