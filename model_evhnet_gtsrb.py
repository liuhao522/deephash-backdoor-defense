# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import time
from transformers import ViTModel, ViTConfig

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 使用CIFAR-10作为示例（GTSRB的替代）
print("正在加载CIFAR-10数据集...")
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 结果存储
results = []


class EVHnet32(nn.Module):
    def __init__(self, num_classes=10):
        super(EVHnet32, self).__init__()
        try:
            print("正在初始化EVHnet32模型...")
            # ViT配置
            config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
            config.num_labels = num_classes

            # 加载预训练ViT
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224', config=config)

            # 冻结ViT参数
            for param in self.vit.parameters():
                param.requires_grad = False

            # 自定义分类头
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
            print("模型初始化成功")
        except Exception as e:
            print(f"模型初始化错误: {e}")
            raise

    def forward(self, x):
        outputs = self.vit(x)
        cls_output = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_output)


def create_model(num_classes):
    """创建并配置EVHnet32模型"""
    try:
        model = EVHnet32(num_classes=num_classes).to(device)

        # 打印模型参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数: {total_params / 1e6:.2f}M")
        print(f"可训练参数: {trainable_params / 1e6:.2f}M")

        return model
    except Exception as e:
        print(f"创建模型错误: {e}")
        raise


def train(model, criterion, optimizer, train_loader, test_loader, epochs):
    """训练函数"""
    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计指标
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 打印批次信息
            if batch_idx % 10 == 0:
                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                current_time = time.strftime('%H:%M:%S', time.localtime())
                print(f'Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | '
                      f'Time {current_time} | Batch Acc: {batch_acc:.2%} | Loss: {loss.item():.4f}')

        # 计算epoch指标
        epoch_loss = train_loss / len(train_loader.dataset)
        epoch_acc = train_correct / len(train_loader.dataset)
        epoch_time = time.time() - start_time

        current_time = time.strftime('%H:%M:%S', time.localtime())
        print(f'Epoch {epoch + 1}/{epochs} | {current_time} | '
              f'Time: {epoch_time:.2f}s | '
              f'Loss: {epoch_loss:.4f} | '
              f'Accuracy: {epoch_acc:.2%}')

    return epoch_acc * 100, epoch_loss


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
                print(f'Testing batch {batch_idx}/{len(test_loader)}')

    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2%}')

    return test_acc * 100, test_loss


def run_experiment():
    """运行完整的训练和测试流程"""
    try:
        print("开始EVHnet32实验...")

        # 创建模型
        num_classes = 10  # CIFAR-10有10个类别
        model = create_model(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

        # 训练和测试
        train_acc, train_loss = train(model, criterion, optimizer, train_loader, test_loader, epochs=5)
        test_acc, test_loss = test(model, criterion, test_loader)

        # 保存结果
        results.append({
            'Dataset': "CIFAR-10",
            'Train Accuracy (%)': train_acc,
            'Train Loss': train_loss,
            'Test Accuracy (%)': test_acc,
            'Test Loss': test_loss
        })

        # 保存模型
        save_path = './save/evhnet32'
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, "evhnet32_cifar10.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model.vit.config,
            'classifier': model.classifier.state_dict()
        }, model_path)
        print(f"模型已保存到: {model_path}")

    except Exception as e:
        print(f"实验失败: {e}")
        raise


def print_results():
    """打印所有结果"""
    print("\n最终结果:")
    print("{:<12} {:<18} {:<12} {:<18} {:<12}".format(
        'Dataset', 'Train Acc (%)', 'Train Loss', 'Test Acc (%)', 'Test Loss'))
    for res in results:
        print("{:<12} {:<18.2f} {:<12.4f} {:<18.2f} {:<12.4f}".format(
            res['Dataset'],
            res['Train Accuracy (%)'],
            res['Train Loss'],
            res['Test Accuracy (%)'],
            res['Test Loss']))


if __name__ == "__main__":
    try:
        # 运行实验
        run_experiment()

        # 打印结果
        print_results()
        print("训练完成！")
    except Exception as e:
        print(f"主程序执行失败: {e}")
        exit(1)