# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import time
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import requests
import zipfile
import shutil

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 结果存储
results = []


def download_and_prepare_gtsrb():
    """下载并准备GTSRB数据集"""
    data_dir = './data/gtsrb'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # 如果数据目录不存在，则下载数据集
    if not os.path.exists(train_dir):
        print("正在下载GTSRB数据集...")
        os.makedirs(data_dir, exist_ok=True)

        # GTSRB数据集下载链接
        url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
        train_zip_path = os.path.join(data_dir, 'GTSRB_Final_Training_Images.zip')

        try:
            # 下载训练集
            response = requests.get(url, stream=True)
            with open(train_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # 解压训练集
            with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

            # 重命名目录以匹配预期结构
            extracted_dir = os.path.join(data_dir, 'GTSRB', 'Final_Training', 'Images')
            if os.path.exists(extracted_dir):
                os.rename(extracted_dir, train_dir)

            # 清理
            os.remove(train_zip_path)
            if os.path.exists(os.path.join(data_dir, 'GTSRB')):
                shutil.rmtree(os.path.join(data_dir, 'GTSRB'))

            print("GTSRB数据集下载完成")

        except Exception as e:
            print(f"下载GTSRB数据集失败: {e}")
            # 如果下载失败，使用CIFAR-10作为替代
            return prepare_cifar10()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建训练数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)

    # 如果没有测试集，从训练集中分割
    if not os.path.exists(test_dir):
        # 分割训练集和测试集
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_subset, test_subset = torch.utils.data.random_split(
            train_dataset, [train_size, test_size]
        )

        # 创建数据加载器
        batch_size = 32
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

        num_classes = len(train_dataset.classes)
    else:
        # 如果有独立的测试集目录
        test_dataset = datasets.ImageFolder(test_dir, transform=transform)
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        num_classes = len(train_dataset.classes)

    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"测试样本: {len(test_loader.dataset)}")
    print(f"类别数量: {num_classes}")

    return train_loader, test_loader, num_classes


def prepare_cifar10():
    """准备CIFAR-10数据集作为备选"""
    print("使用CIFAR-10作为备选数据集...")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, 10


def prepare_mnist():
    """准备MNIST数据集，转换为EfficientNetV2兼容的格式"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(3),  # 转换为3通道
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, 10


def create_model(num_classes):
    """创建并配置EfficientNetV2模型"""
    print("正在创建EfficientNetV2模型...")
    # 加载预训练权重（IMAGENET1K_V1）
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    # 修改分类器
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model = model.to(device)

    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")

    return model


def train(model, criterion, optimizer, train_loader, epochs, dataset_name):
    """训练函数"""
    print(f"\n在{dataset_name}数据集上训练...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
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

            # 打印批次信息
            if batch_idx % 10 == 0:
                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                current_time = time.strftime('%H:%M:%S', time.localtime())
                print(f'周期 {epoch + 1}/{epochs} | 批次 {batch_idx}/{len(train_loader)} | '
                      f'时间 {current_time} | 批次准确率: {batch_acc:.2%} | 损失: {loss.item():.4f}')

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100.0 * train_correct / len(train_loader.dataset)
        epoch_time = time.time() - start_time
        current_time = time.strftime('%H:%M:%S', time.localtime())
        print(f'周期 {epoch + 1}/{epochs} | {current_time} | '
              f'时间: {epoch_time:.2f}s | '
              f'训练损失: {train_loss:.4f} | 训练准确率: {train_accuracy:.2f}%')

    return train_accuracy, train_loss


def test(model, criterion, test_loader):
    """测试函数"""
    print("开始测试...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                print(f'测试批次 {batch_idx}/{len(test_loader)}')

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * test_correct / len(test_loader.dataset)

    print(f'测试完成 | 测试损失: {test_loss:.4f} | 测试准确率: {test_accuracy:.2f}%')

    return test_accuracy, test_loss


def run_experiment(dataset_name):
    """运行完整的训练和测试流程"""
    try:
        # 准备数据集
        if dataset_name == "GTSRB":
            train_loader, test_loader, num_classes = download_and_prepare_gtsrb()
        elif dataset_name == "MNIST":
            train_loader, test_loader, num_classes = prepare_mnist()
        elif dataset_name == "CIFAR10":
            train_loader, test_loader, num_classes = prepare_cifar10()
        else:
            raise ValueError("未知数据集")

        # 创建模型
        model = create_model(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

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
        save_path = './save/efficientnetv2'
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, f"efficientnetv2_{dataset_name.lower()}.pth"))
        print(f"模型已保存到: {os.path.join(save_path, f'efficientnetv2_{dataset_name.lower()}.pth')}")

    except Exception as e:
        print(f"实验失败: {e}")
        import traceback
        traceback.print_exc()


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


if __name__ == "__main__":
    # 运行所有数据集的实验
    datasets_to_run = ["GTSRB", "MNIST", "CIFAR10"]

    for dataset in datasets_to_run:
        print(f"\n{'=' * 50}")
        print(f"开始处理数据集: {dataset}")
        print(f"{'=' * 50}")
        run_experiment(dataset)

    # 打印最终结果
    print_results()
    print("\n所有实验完成！")