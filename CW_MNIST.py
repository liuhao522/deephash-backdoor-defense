import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.models import resnet50


model_state_dict = torch.load('save/resnet/resnet50_mnist.pt')

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
print(device)
model = resnet50()
num_classes = 10
model.fc = nn.Linear(2048, num_classes)
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(3),  # 将图像转换为3通道
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 运行需要稍等，这里表示下载并加载数据集
normal_data = datasets.ImageFolder('./mnist', transform=data_transforms)
normal_loader = DataLoader(normal_data, batch_size=1, shuffle=False)
#
# correct = 0
# total = 0
#
# for images, labels in normal_loader:
#     images = images.to(device)
#     labels = labels.to(device)
#     outputs = model(images)
#
#     _, pre = torch.max(outputs.data, 1)
#
#     total += 1
#     correct += (pre == labels).sum()
#
# print('Accuracy of test text: %f %%' % (100 * float(correct) / total))


# CW-L2 Attack
# Based on the paper, i.e. not exact same version of the code on https://github.com/carlini/nn_robust_attacks
# (1) Binary search method for c, (2) Optimization on tanh space, (3) Choosing method best l2 adversaries is NOT IN THIS CODE.
def cw_l2_attack(i,model, images, labels, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01):
    images = images.to(device)
    labels = labels.to(device)

    # Define f-function
    def f(x):

        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0])).to(device)[labels.to(device)]

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())

        # If targeted, optimize for making the other class most likely
        if targeted:
            return torch.clamp(i - j, min=-kappa)

        # If untargeted, optimize for making the other class most likely
        else:
            return torch.clamp(j - i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10

    for step in range(max_iter):

        a = 1 / 2 * (nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c * f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter // 10) == 0:
            if cost > prev:
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost

        print('- Learning Progress : %2.2f %%        ' % ((step + 1) / max_iter * 100), end='\r')

    attack_images = 1 / 2 * (nn.Tanh()(w) + 1)
    filename = normal_data.imgs[i][0]
    # print(filename)
    filename = filename[10:]
    print(filename)
    save_path = './CWL2/'
    torchvision.utils.save_image(attack_images.squeeze().cpu(),
                                 save_path + '/' + filename)
    return attack_images


print("Attack Image & Predicted Label")

model.eval()

correct = 0
total = 0

for i, (images, labels) in enumerate(normal_loader):

    images = cw_l2_attack(i, model, images, labels, targeted=False, c=0.1)
    labels = labels.to(device)
    outputs = model(images)

    _, pre = torch.max(outputs.data, 1)

    total += 1
    correct += (pre == labels).sum()


print('Accuracy of test text: %f %%' % (100 * float(correct) / total))