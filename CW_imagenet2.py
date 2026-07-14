from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from torchvision.models import resnet50
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, transforms

import torch.nn.functional as F
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#MNIST数据集加载和处理
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
# train_data = MNIST(root="data", train=True, download=True, transform=transform)
# test_data = MNIST(root="data", train=False, download=True, transform=transform)
# 构建3通道的MNIST图像
# 运行需要稍等，这里表示下载并加载数据集
mnist_dataset = datasets.ImageFolder('./imagenetclass', transform=data_transforms)

data_loader = DataLoader(mnist_dataset, batch_size=1, shuffle=True)
#数据处理
batch_size = 1
# train_loader = DataLoader(train_data, batch_size=batch_size)
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
model_state_dict = torch.load('./save/resnet/resnet50_imagenet.pt')
use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
model = resnet50()
num_classes = 100
# 修改模型的输入通道数为1
# model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(2048, num_classes)
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()
j = 0
unloader = transforms.ToPILImage()
for i, (data, label) in enumerate(data_loader):
    if i == 1500:
        break
    # predic = torch.argmax(model(data.to(device)),dim=1).detach().cpu()
    print('org_label', label[j])
    advx = carlini_wagner_l2(model,data.to(device),100,torch.tensor([5]*batch_size,device=device),confidence=2.0,targeted=False)
    # adver_target = torch.max(model(advx),1)[1]
    # print('adv_label_1', adver_target)
    # predic = torch.argmax(model(advx.to(device)),dim=1).detach().cpu()


    # print('adv_label', predic[j])
    with torch.no_grad():
        output = model(advx.to(device))

    # 获取预测结果
    _, predicted_idx = torch.max(output, 1)
    predicted_label = predicted_idx.item()
    print('adv_label', predicted_label)
    torch.save(advx, f'./CW_imagenet2.0/{i}-{label[j]}-{predicted_label}.pt')  # 保存Tensor为pth文件
#     # tensor_data = advx.cpu().numpy()
#
#     # save_path = './CW3'
#     # filename = f'{i}-{label[j]}-{predicted_label}.png'
#     # torchvision.utils.save_image(advx.squeeze().cpu(),
#     #                                  save_path + '/' + filename)
#
# # 直接保存tensor格式图片
#     images = advx.cpu().clone()  # we clone the tensor to not do changes on it
#     images = images.squeeze(0)  # remove the fake batch dimension
#     images = unloader(images)
#     images.save(f'./CW2/{i}-{label[j]}-{predicted_label}.jpg')
#
#     pre_image = torch.squeeze(advx.clamp(0, 1).cpu())
#
#     a = transforms.ToPILImage()(pre_image)





# plt.figure(figsize=(16,8))
# id=0
# for i in range(4):
#     for j in range(4):
#         plt.subplot(4,4,id+1)
#         plt.imshow(advx[id,0].reshape(28,28),cmap="gray")
#         plt.title(f"{label[id]}->{predic[id]}",{"color":"red"})
#         plt.axis("off")
#         id += 1
#
# plt.show()
