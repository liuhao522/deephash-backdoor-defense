# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import time
import traceback
from transformers import ViTModel, ViTConfig

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

        # 检查数据集路径
        train_path = './imagenetclass'
        test_path = './imagenetvalclass_image'
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training dataset path not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test dataset path not found: {test_path}")

        train_dataset = datasets.ImageFolder(train_path, transform=transform)
        test_dataset = datasets.ImageFolder(test_path, transform=transform)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Number of classes: {len(train_dataset.classes)}")

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, test_loader, len(train_dataset.classes)
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        traceback.print_exc()
        raise


class EVHnet32(nn.Module):
    def __init__(self, num_classes=100):
        super(EVHnet32, self).__init__()
        try:
            print("Initializing EVHnet32 model...")
            # ViT配置
            config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
            config.num_labels = num_classes

            # 加载预训练ViT
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224', config=config)

            # 冻结ViT参数(可选)
            for param in self.vit.parameters():
                param.requires_grad = False

            # 自定义分类头
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
            print("Model initialized successfully.")
        except Exception as e:
            print(f"Error initializing model: {e}")
            traceback.print_exc()
            raise

    def forward(self, x):
        try:
            outputs = self.vit(x)
            cls_output = outputs.last_hidden_state[:, 0]
            return self.classifier(cls_output)
        except Exception as e:
            print(f"Forward pass error: {e}")
            traceback.print_exc()
            raise


def create_model(num_classes):
    """创建并配置EVHnet32模型"""
    try:
        model = EVHnet32(num_classes=num_classes).to(device)

        # 打印模型参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

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

            print(f'Epoch {epoch + 1}/{epochs} completed | '
                  f'Time: {epoch_time:.2f}s | '
                  f'Loss: {epoch_loss:.4f} | '
                  f'Accuracy: {epoch_acc:.2%}')

        return epoch_acc * 100, epoch_loss
    except Exception as e:
        print(f"Training error: {e}")
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

                if batch_idx % 10 == 0:
                    print(f'Testing batch {batch_idx}/{len(test_loader)}')

        test_loss /= len(test_loader.dataset)
        test_acc = test_correct / len(test_loader.dataset)

        print(f'Test completed | Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}')

        return test_acc * 100, test_loss
    except Exception as e:
        print(f"Testing error: {e}")
        traceback.print_exc()
        raise


def run_experiment():
    """运行完整的训练和测试流程"""
    try:
        print("Starting EVHnet32 experiment...")

        # 准备数据
        train_loader, test_loader, num_classes = prepare_imagenet()

        # 创建模型
        model = create_model(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

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
        save_path = './save/evhnet32'
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, "evhnet32_imagenet100.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model.vit.config,
            'classifier': model.classifier.state_dict()
        }, model_path)
        print(f"Model saved to {model_path}")

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

        # 打印结果
        print_results()
    except Exception as e:
        print(f"Main execution failed: {e}")
        traceback.print_exc()
        exit(1)