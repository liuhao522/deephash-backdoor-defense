# -*- coding:utf-8 -*-
# author:zhangning
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

model_state_dict = torch.load('save/resnet/resnet50_cifar10.pt')
use_cuda = True
# 这里的扰动量先设定为几个值，后面可视化展示不同的扰动量影响以及成像效果
epsilons = [0.009]
# 看看我们有没有配置GPU，没有就是使用cpu
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
model = resnet50()
num_classes = 10
model.fc = nn.Linear(2048, num_classes)
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
# 运行需要稍等，这里表示下载并加载数据集
dataset = datasets.ImageFolder('./cifar10class', transform=data_transforms)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
classes = dataset.classes

def unnormalize(img, mean = np.array(norm_mean), std = np.array(norm_std)):
  '''
   unnormalize the image that has been normalized with mean and std
  '''
  inverse_mean = - mean/std
  inverse_std = 1/std
  img = transforms.Normalize(mean=-mean/std, std=1/std)(img)
  return img


def normalize(img, mean = np.array(norm_mean), std = np.array(norm_std)):
  return transforms.Normalize(mean = norm_mean, std = norm_std)(img)



# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    sign_data_grad = data_grad.sign()
    # 通过epsilon生成对抗样本
    perturbed_image = image + epsilon * sign_data_grad
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    perturbed_image = transforms.Normalize(mean = norm_mean, std = norm_std)(perturbed_image)

    # 返回对抗样本
    return perturbed_image.float()


def test(model, device, test_loader, epsilon, save_dir):
    correct = 0
    adv_examples = []
    for i, (images, labels) in enumerate(test_loader):
        # 将数据和标签发送到设备
        data, target = images.to(device), labels.to(device)

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
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu()
                ori_ex = data.squeeze().detach().cpu()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex, ori_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu()
                ori_ex = data.squeeze().detach().cpu()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex, ori_ex))

        filename = dataset.imgs[i][0]
        filename = filename[17:]
        print(filename)
        torchvision.utils.save_image(perturbed_data.squeeze().cpu(),
                                     save_dir + '/' + filename)

    accuracy = 100 * correct / float(len(test_loader))
    print("扰动量: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), accuracy))
    return accuracy, adv_examples


accuracies = []
examples = []

# 对每个干扰程度进行测试
for eps in epsilons:
    acc, ex = test(model, device, data_loader, eps, save_dir='./adv_0.009')
    accuracies.append(acc)
    examples.append(ex)
#
# def get_num_correct(out, labels):  #求准确率
#     return out.argmax(dim=1).eq(labels).sum().item()

# test_accuracy = 0
# with torch.no_grad():
#   for (images,labels) in data_loader:
#       images,labels = images.to(device),labels.to(device)
#       outs = model(images)
#       test_accuracy += get_num_correct(outs,labels)
# test_accuracy /= len(data_loader)
# print(f"Accuracy of the trained resnet50 model: {(100*test_accuracy):>0.1f}%")


# # 测试原始模型的准确率
# test(model, device, data_loader, epsilon=0, save_dir='./original_images')
#
# # 测试对抗攻击后的模型准确率，并保存对抗样本图片
# test(model, device, data_loader, epsilon=0.1, save_dir='./adversarial_images')


