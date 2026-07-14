# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets  # 添加datasets导入
from torch.utils.data import DataLoader
import os
import torchvision
import time
from torchvision.models import resnet50, googlenet, efficientnet_v2_s, vit_b_16


# 设备检测（增加详细输出）
def setup_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f'当前GPU: {torch.cuda.get_device_name(0)}')
        print(f'可用显存: {torch.cuda.mem_get_info()[1] // 1024 ** 2}MB')
    else:
        print('使用CPU运行')
    return device


# 数据转换（增加自适应调整）
def get_transform(dataset_name):
    base_transform = [
        transforms.Resize(224),
        transforms.ToTensor()
    ]

    if dataset_name == 'MNIST':
        base_transform.insert(1, transforms.Grayscale(num_output_channels=3))
        normalize = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    else:  # CIFAR10/ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    base_transform.append(normalize)
    return transforms.Compose(base_transform)


# 数据加载（增加存在性检查）
def get_data_loaders(dataset_name):
    transform = get_transform(dataset_name)

    try:
        if dataset_name == 'MNIST':
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'ImageNet100':
            if not os.path.exists('./imagenetclass'):
                raise FileNotFoundError("ImageNet100路径不存在")
            trainset = datasets.ImageFolder('./imagenetclass', transform=transform)
            testset = datasets.ImageFolder('./imagenetvalclass_image', transform=transform)

        train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)  # 减小batch_size
        test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
        return train_loader, test_loader

    except Exception as e:
        print(f"[错误] 加载{dataset_name}失败: {str(e)}")
        return None, None


# 模型初始化（增加预训练权重检查）
def get_model(model_name, num_classes):
    try:
        weights = 'IMAGENET1K_V1' if model_name != 'ViT' else 'IMAGENET1K_V1'

        if model_name == 'ResNet50':
            model = resnet50(weights=weights)
            model.fc = nn.Linear(2048, num_classes)
        elif model_name == 'GoogLeNet':
            model = googlenet(weights=weights)
            model.fc = nn.Linear(1024, num_classes)
        elif model_name == 'EfficientNet':
            model = efficientnet_v2_s(weights=weights)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'ViT':
            model = vit_b_16(weights=weights)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

        return model.to(device)
    except Exception as e:
        print(f"[错误] 初始化{model_name}失败: {str(e)}")
        return None


# 训练过程（增加梯度裁剪）
def train(model, criterion, optimizer, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        time_elapsed = time.time() - start_time

        print(f'Epoch [{epoch + 1}/{epochs}] | '
              f'Loss: {epoch_loss:.4f} | '
              f'Acc: {epoch_acc:.2f}% | '
              f'Time: {time_elapsed:.2f}s')

    return epoch_loss, epoch_acc


if __name__ == '__main__':
    device = setup_device()
    criterion = nn.CrossEntropyLoss()

    # 实验配置（可调整）
    config = {
        'datasets': ['MNIST', 'CIFAR10'],  # 建议先测试这两个
        'models': ['ResNet50', 'GoogLeNet'],
        'epochs': 5,  # 测试时减少轮数
        'batch_size': 64  # 减小batch_size
    }

    results = []

    for dataset in config['datasets']:
        print(f"\n{'=' * 30} {dataset} {'=' * 30}")

        train_loader, test_loader = get_data_loaders(dataset)
        if train_loader is None:
            continue

        num_classes = len(train_loader.dataset.classes)
        print(f"类别数: {num_classes}")

        for model_name in config['models']:
            print(f"\n▶ 正在训练 {model_name}...")

            model = get_model(model_name, num_classes)
            if model is None:
                continue

            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

            try:
                # 训练监控
                train_loss, train_acc = train(
                    model, criterion, optimizer,
                    train_loader, config['epochs']
                )

                # 测试评估
                test_loss, test_acc = test(model, criterion, test_loader)

                # 记录结果
                results.append({
                    '数据集': dataset,
                    '模型': model_name,
                    '训练准确率': f"{train_acc:.2f}%",
                    '测试准确率': f"{test_acc:.2f}%",
                    '训练损失': f"{train_loss:.4f}",
                    '测试损失': f"{test_loss:.4f}"
                })

                # 模型保存
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(model.state_dict(),
                           f"checkpoints/{model_name}_{dataset}.pth")

                # 显存清理
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"训练过程中出错: {str(e)}")
                continue

    # 结果展示
    print("\n\n最终结果汇总:")
    print("{:<10} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "数据集", "模型", "训练准确率", "测试准确率", "训练损失", "测试损失"))
    print("-" * 80)

    for res in results:
        print("{:<10} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
            res['数据集'], res['模型'], res['训练准确率'],
            res['测试准确率'], res['训练损失'], res['测试损失']))