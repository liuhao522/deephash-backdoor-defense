# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import time
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 结果存储
results = []


def prepare_mnist():
    """准备MNIST数据集，转换为ConvNeXt兼容的格式"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(3),  # 转换为3通道
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, 10


def prepare_cifar10():
    """准备CIFAR-10数据集"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, 10


def prepare_imagenet():
    """准备ImageNet数据集（假设文件夹已正确设置）"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder('./imagenetclass', transform=transform)
    test_dataset = datasets.ImageFolder('./imagenetvalclass_image', transform=transform)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, 100  # ImageNet有1000类


def prepare_gtsrb():
    """准备GTSRB数据集"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 确保固定尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 下载并加载GTSRB数据集
    train_dataset = datasets.GTSRB(root='./data', split='train', download=True, transform=transform)
    test_dataset = datasets.GTSRB(root='./data', split='test', download=True, transform=transform)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # GTSRB有43个类别
    return train_loader, test_loader, 43


def create_model(num_classes):
    """创建并配置ConvNeXt模型"""
    # 加载预训练权重（IMAGENET1K_V1）
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    # 修改分类头
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    model = model.to(device)
    return model


def train(model, criterion, optimizer, train_loader, epochs, dataset_name):
    """训练函数"""
    print(f"\n在{dataset_name}数据集上训练...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0

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
            total += labels.size(0)

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100.0 * train_correct / len(train_loader.dataset)
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print(f'周期 {epoch + 1}/{epochs} | {current_time} | '
              f'训练损失: {train_loss:.4f} | 训练准确率: {train_accuracy:.2f}%')

    return train_accuracy, train_loss


def test(model, criterion, test_loader):
    """测试函数"""
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

    return test_accuracy, test_loss


def run_experiment(dataset_name):
    """运行完整的训练和测试流程"""
    # 准备数据集
    if dataset_name == "MNIST":
        train_loader, test_loader, num_classes = prepare_mnist()
    elif dataset_name == "CIFAR10":
        train_loader, test_loader, num_classes = prepare_cifar10()
    elif dataset_name == "ImageNet":
        train_loader, test_loader, num_classes = prepare_imagenet()
    elif dataset_name == "GTSRB":
        train_loader, test_loader, num_classes = prepare_gtsrb()
    else:
        raise ValueError("未知数据集")

    # 创建模型
    model = create_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练和测试
    train_acc, train_loss = train(model, criterion, optimizer, train_loader, 10, dataset_name)
    test_acc, test_loss = test(model, criterion, test_loader)

    # 保存结果
    results.append({
        'Dataset': dataset_name,
        'Train Accuracy (%)': train_acc,
        'Train Loss': train_loss,
        'Test Accuracy (%)': test_acc,
        'Test Loss': test_loss
    })

    # 保存模型
    save_path = './save/convnext'
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f"convnext_{dataset_name.lower()}.pt"))


def print_results():
    """打印所有结果"""
    print("\n最终结果:")
    print("{:<10} {:<18} {:<12} {:<18} {:<12}".format(
        '数据集', '训练准确率 (%)', '训练损失', '测试准确率 (%)', '测试损失'))
    for res in results:
        print("{:<10} {:<18.2f} {:<12.4f} {:<18.2f} {:<12.4f}".format(
            res['Dataset'],
            res['Train Accuracy (%)'],
            res['Train Loss'],
            res['Test Accuracy (%)'],
            res['Test Loss']))


# 运行所有数据集的实验
datasets_to_run = ["GTSRB"]  # 只运行GTSRB数据集
for dataset in datasets_to_run:
    run_experiment(dataset)

# 打印最终结果
print_results()