# -*- coding:utf-8 -*- 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import time

# 数据预处理
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 使用CIFAR-10数据集（作为GTSRB的替代）
print("正在加载CIFAR-10数据集...")
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 加载预训练的GoogLeNet模型
print("正在加载GoogLeNet模型...")
model = models.googlenet(pretrained=True)

# 将模型最后一层替换为一个新的全连接层
num_classes = 10  # CIFAR-10有10个类别
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 将模型移动到GPU上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# 训练函数
def train(model, criterion, optimizer, train_loader, test_loader, epochs):
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
        print(f'Epoch {epoch + 1}/{epochs} | {current_time} | '
              f'Training Loss: {train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}%')


# 测试函数
def test(model, criterion, test_loader):
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
    print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%')


# 创建保存目录
save_path = './save/googlenet'
os.makedirs(save_path, exist_ok=True)

# 训练模型
num_epochs = 5  # 减少epochs以便快速测试
print("开始训练...")
train(model, criterion, optimizer, train_loader, test_loader, epochs=num_epochs)

# 保存模型
model_save_path = os.path.join(save_path, "googlenet_cifar10.pth")
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存到: {model_save_path}")

# 测试模型
print("开始测试...")
test(model, criterion, test_loader)

print("训练完成！")