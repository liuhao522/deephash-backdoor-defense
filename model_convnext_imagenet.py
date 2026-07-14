# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import time
import traceback
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 结果存储
results = []


def prepare_imagenet():
    """准备ImageNet100数据集"""
    try:
        print("Preparing ImageNet100 dataset...")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 检查数据集路径是否存在
        train_path = './imagenetclass'
        test_path = './imagenetvalclass_image'
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training dataset path not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test dataset path not found: {test_path}")

        train_dataset = datasets.ImageFolder(train_path, transform=transform)
        test_dataset = datasets.ImageFolder(test_path, transform=transform)

        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")
        print(f"Number of classes: {len(train_dataset.classes)}")

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, test_loader, len(train_dataset.classes)
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        traceback.print_exc()
        raise


def create_model(num_classes):
    """创建并配置ConvNeXt模型"""
    try:
        print("Creating ConvNeXt model...")
        # 加载预训练权重（IMAGENET1K_V1）
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # 修改分类头
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

        model = model.to(device)
        print("Model created successfully.")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        traceback.print_exc()
        raise


def train(model, criterion, optimizer, train_loader, epochs, dataset_name):
    """训练函数"""
    print(f"\nTraining on {dataset_name} dataset...")
    try:
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

                # 打印批次进度
                if batch_idx % 10 == 0:
                    current_time = time.strftime('%H:%M:%S', time.localtime())
                    batch_acc = 100.0 * (predicted == labels).sum().item() / labels.size(0)
                    print(f'Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | '
                          f'Time {current_time} | Batch Acc: {batch_acc:.2f}%')

            train_loss /= len(train_loader.dataset)
            train_accuracy = 100.0 * train_correct / len(train_loader.dataset)
            epoch_time = time.time() - start_time
            current_time = time.strftime('%H:%M:%S', time.localtime())
            print(f'Epoch {epoch + 1}/{epochs} | {current_time} | Time: {epoch_time:.2f}s | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%')

        return train_accuracy, train_loss
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        raise


def test(model, criterion, test_loader):
    """测试函数"""
    print("\nTesting model...")
    try:
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

                # 打印测试进度
                if batch_idx % 10 == 0:
                    print(f'Testing batch {batch_idx}/{len(test_loader)}')

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100.0 * test_correct / len(test_loader.dataset)
        print(f'Test completed | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%')

        return test_accuracy, test_loss
    except Exception as e:
        print(f"Error during testing: {e}")
        traceback.print_exc()
        raise


def run_experiment():
    """运行完整的训练和测试流程"""
    try:
        print("Starting experiment with ConvNeXt...")

        # 准备数据集
        train_loader, test_loader, num_classes = prepare_imagenet()

        # 创建模型
        model = create_model(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # 训练和测试
        train_acc, train_loss = train(model, criterion, optimizer, train_loader, 10, "ImageNet100")
        test_acc, test_loss = test(model, criterion, test_loader)

        # 保存结果
        results.append({
            'Dataset': "ImageNet100",
            'Train Accuracy (%)': train_acc,
            'Train Loss': train_loss,
            'Test Accuracy (%)': test_acc,
            'Test Loss': test_loss
        })

        # 保存模型
        save_path = './save/convnext'
        os.makedirs(save_path, exist_ok=True)
        model_save_path = os.path.join(save_path, "convnext_imagenet100.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    except Exception as e:
        print(f"Experiment failed: {e}")
        traceback.print_exc()
        raise


def print_results():
    """打印所有结果"""
    print("\nFinal results:")
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

        # 打印最终结果
        print_results()
    except Exception as e:
        print(f"Main execution failed: {e}")
        traceback.print_exc()
        exit(1)