# -*- coding:utf-8 -*- 
# author:zhangning
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import torchvision
import os
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
model_state_dict = torch.load('save/resnet/resnet_mnist.pt')
use_cuda = True
# 这里的扰动量先设定为几个值，后面可视化展示不同的扰动量影响以及成像效果
epsilons = [.3]
# 看看我们有没有配置GPU，没有就是使用cpu
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
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
mnist_dataset = datasets.ImageFolder('./mnistclass', transform=data_transforms)

data_loader = DataLoader(mnist_dataset, batch_size=1, shuffle=False)
"""
for i, data in enumerate(data_loader):
    images, labels = data

    # 打印数据集中的图片
    img = torchvision.utils.make_grid(images).numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

    break
"""


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    sign_data_grad = data_grad.sign()
    # 通过epsilon生成对抗样本
    perturbed_image = image + epsilon*sign_data_grad
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回对抗样本
    return perturbed_image


def save_gray_image(img, save_dir, type, name, Gray=False):
    fname, fext = name.split('.')
    imgPath = os.path.join(save_dir, "%s_%s.%s" % (fname, type, fext))
    # torchvision.utils.save_image(img, imgPath)
    # 改写：torchvision.utils.save_image
    grid = torchvision.utils.make_grid(img, nrow=8, padding=2, pad_value=0,
                                       normalize=False, range=None, scale_each=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    # im.show()
    if Gray:
        im.convert('L').save(imgPath)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
    else:
        im.save(imgPath)


def test(model, device, test_loader, epsilon):
    # 准确度计数器
    correct = 0
    # 对抗样本
    adv_examples = []

    # 循环所有测试集
    for i, (data, target) in enumerate(test_loader):
        # 将数据和标签发送到设备
        data, target = data.to(device), target.to(device)

        # 设置张量的requires_grad属性。重要的攻击
        data.requires_grad = True

        # 通过模型向前传递数据
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # 得到最大对数概率的索引

        # 如果最初的预测是错误的，不要再攻击了，继续下一个目标的对抗训练
        if init_pred.item() != target.item():
            continue

        # 计算损失
        loss = F.nll_loss(output, target)

        # 使所有现有的梯度归零
        model.zero_grad()

        # 计算模型的后向梯度
        loss.backward()

        # 收集datagrad
        data_grad = data.grad.data

        # 调用FGSM攻击
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # 对受扰动的图像进行重新分类
        output = model(perturbed_data)

        # 检查是否成功
        final_pred = output.max(1, keepdim=True)[1]  # 得到最大对数概率的索引
        if final_pred.item() == target.item():
            correct += 1
            # 这里都是为后面的可视化做准备
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # 这里都是为后面的可视化做准备
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        filename = mnist_dataset.imgs[i][0]
        filename = filename[15:]
        # print(filename)
        save_path = './adv_ex0.3/'
        torchvision.utils.save_image(perturbed_data.squeeze().cpu(),
                                     save_path + '/' + filename)
    # 计算最终精度
    final_acc = correct / float(len(test_loader))
    print("扰动量: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # 返回准确性和对抗性示例
    return final_acc, adv_examples


accuracies = []
examples = []

# 对每个干扰程度进行测试
for eps in epsilons:
    acc, ex = test(model, device, data_loader, eps)
    accuracies.append(acc * 100)
    examples.append(ex)

plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 110, step=10))
plt.xticks(np.arange(0, .5, step=0.05))

def to_percent(temp, position):
    return '%1.0f' % (temp) + '%'

plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.title("准确率 vs 扰动量")
plt.xlabel("扰动量")
plt.ylabel("准确率")
plt.show()


# 在每个处绘制几个对抗性样本的例子
# 在每个处绘制几个对抗性样本的例子
# cnt = 0
# plt.figure(figsize=(8,10))
# for i in range(len(epsilons)):
#     for j in range(len(examples[i])):
#         cnt += 1
#         plt.subplot(len(epsilons),len(examples[0]),cnt)
#         plt.xticks([], [])
#         plt.yticks([], [])
#         if j == 0:
#             plt.ylabel("扰动: {}".format(epsilons[i]), fontsize=14)
#         orig,adv,ex = examples[i][j]
#         plt.title("{} -> {}".format(orig, adv))
#         plt.imshow(ex[0], cmap="gray")
# plt.tight_layout()
# plt.show()