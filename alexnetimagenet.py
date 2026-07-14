# -*- coding:utf-8 -*- 
# author:zhangning
# -*- coding:utf-8 -*-
# author:zhangning
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision
from torchvision import datasets, transforms
from torchvision.models import alexnet


import os
import time



norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
# 运行需要稍等，这里表示下载并加载数据集
train_dataset = datasets.ImageFolder('./imagenetclass', transform=data_transforms)
test_dataset = datasets.ImageFolder('./imagenetvalclass_image', transform=data_transforms)
batch_size = 32
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建数据加载器
batch_size = 32

model = alexnet(pretrained=True)
# 将模型移动到GPU上
model = model.cuda()
# 将模型最后一层替换为一个新的全连接层，以适应MNIST数据集的类别数
num_classes = 100
model.classifier[6] = nn.Linear(4096, num_classes).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 训练模型
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def train(model, criterion, optimizer, train_loader, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100.0 * train_correct / len(train_loader.dataset)
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print(f'Epoch {epoch + 1}/{epochs}',
              f'{current_time}',
              f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')


def test(model, criterion, test_loader,):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * test_correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


save_path = './save/resnet'
train(model, criterion, optimizer, train_loader, epochs=10)
torch.save(model.state_dict(), os.path.join(save_path, "alexnet_imagenet.pt"))
test(model, criterion, test_loader)

