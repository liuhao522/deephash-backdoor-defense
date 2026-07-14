import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import sys
import logging
import time
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 文件路径配置
train_excel_path = r'D:/deephash_original/data/CIFAR10/train1.xlsx'
test_excel_path = r'D:/deephash_original/data/CIFAR10/train2.xlsx'
images_folder = r'D:/deephash_original/dataset/cifar10/images_blended2/'
test_images_folder = r'D:/deephash_original/dataset/cifar10/images_blended_test/'

# 模型保存路径
attack_model_path = 'D:/deephash_original/attack_model_resnet50.pth'
abl_defended_model_path = 'D:/deephash_original/abl_defended_model.pth'

# 训练参数
BATCH_SIZE = 32  # 进一步减小batch size
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 10

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CIFAR10Dataset(Dataset):
    def __init__(self, df, root_dir, transform=None, is_test=False):
        self.df = df.iloc[1:].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            img_name = str(self.df.iloc[idx, 0])
            img_path = os.path.join(self.root_dir, img_name)

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图片文件不存在: {img_path}")

            image = Image.open(img_path).convert('RGB')

            # 提取真实标签和机器标签
            true_label = self.extract_true_label(img_name)
            machine_label = int(self.df.iloc[idx, 1])

            # 如果无法提取真实标签，则使用Excel中的标签作为真实标签
            if true_label == -1:
                true_label = machine_label

            # 计算中毒指示器
            poison_indicator = 1 if true_label != machine_label else 0

            if self.transform:
                image = self.transform(image)

            return image, machine_label, true_label, poison_indicator, idx, img_name

        except Exception as e:
            print(f"处理文件 {img_name} 出错: {e}")
            return None, None, None, None, None, None

    def extract_true_label(self, filename):
        basename = os.path.splitext(filename)[0]
        if '-label-' in basename:
            parts = basename.split('-label-')
            if len(parts) == 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return -1
        return -1


def create_resnet50_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, NUM_CLASSES)
    return model.to(device)


class FloodingLoss(nn.Module):
    """Flooding Loss - 改进版本"""

    def __init__(self, flood_level=0.2):
        super(FloodingLoss, self).__init__()
        self.flood_level = flood_level

    def forward(self, loss):
        return (loss - self.flood_level).abs() + self.flood_level


def compute_robust_loss_values(model, data_loader, flooding_level=0.1):
    """使用Flooding方法计算鲁棒的损失值"""
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none')
    flooding_criterion = FloodingLoss(flooding_level)

    losses = []
    indices = []
    all_features = []

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for batch_idx, batch in enumerate(tqdm(data_loader, desc="计算鲁棒损失")):
        valid_batch = [item for item in zip(*batch) if item[0] is not None]
        if not valid_batch:
            continue

        images, machine_labels, _, _, batch_indices, _ = zip(*valid_batch)
        images = torch.stack(images).to(device)
        machine_labels = torch.tensor(machine_labels).to(device)

        # 使用Flooding方法进行前向传播
        outputs = model(images)
        batch_losses = criterion(outputs, machine_labels)

        # 应用Flooding
        flooded_loss = flooding_criterion(batch_losses.mean())

        # 反向传播
        optimizer.zero_grad()
        flooded_loss.backward()
        optimizer.step()

        # 记录损失
        losses.extend(batch_losses.detach().cpu().numpy())
        indices.extend(batch_indices)

        # 提取特征用于聚类分析
        with torch.no_grad():
            features = model(images)
            all_features.extend(features.cpu().numpy())

    return np.array(losses), np.array(indices), np.array(all_features)


def analyze_loss_distribution(losses, method='dynamic'):
    """分析损失分布并确定阈值"""
    if method == 'dynamic':
        # 动态阈值：基于异常检测
        Q1 = np.percentile(losses, 25)
        Q3 = np.percentile(losses, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 选择损失较低且不是异常值的样本作为疑似中毒样本
        suspected_mask = (losses <= Q1) & (losses >= lower_bound)
        return suspected_mask, Q1

    elif method == 'percentile':
        # 百分位数方法：选择损失最低的10%
        threshold = np.percentile(losses, 10)
        suspected_mask = losses <= threshold
        return suspected_mask, threshold

    else:
        # 混合方法
        threshold1 = np.percentile(losses, 15)
        threshold2 = np.percentile(losses, 5)
        suspected_mask = (losses <= threshold1) & (losses >= threshold2)
        return suspected_mask, threshold1


def advanced_sample_isolation(model, data_loader, isolation_ratio=0.1):
    """高级样本隔离策略"""
    print("执行高级样本隔离策略...")

    # 方法1: 标准损失计算
    standard_losses, indices, _ = compute_robust_loss_values(model, data_loader, flooding_level=0.05)

    # 方法2: 使用多种策略分析
    methods = ['dynamic', 'percentile', 'mixed']
    all_suspected_masks = []

    for method in methods:
        suspected_mask, threshold = analyze_loss_distribution(standard_losses, method)
        all_suspected_masks.append(suspected_mask)
        print(f"方法 '{method}': 阈值={threshold:.4f}, 疑似样本数={np.sum(suspected_mask)}")

    # 综合所有方法的结果
    final_suspected_mask = np.any(all_suspected_masks, axis=0)

    # 确保隔离样本数量合理
    max_isolate = int(len(standard_losses) * isolation_ratio)
    if np.sum(final_suspected_mask) > max_isolate:
        # 选择损失最低的样本
        sorted_indices = np.argsort(standard_losses)
        final_suspected_mask = np.zeros_like(final_suspected_mask)
        final_suspected_mask[sorted_indices[:max_isolate]] = True

    suspected_indices = indices[final_suspected_mask]
    clean_indices = indices[~final_suspected_mask]

    print(f"最终隔离了 {len(suspected_indices)} 个疑似中毒样本")
    print(f"剩余 {len(clean_indices)} 个干净样本")

    return suspected_indices, clean_indices, standard_losses


def evaluate_model_comprehensive(model, test_loader):
    """综合评估模型性能"""
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_machine_labels = []
    all_poison_indicators = []
    all_confidences = []

    with torch.no_grad():
        for batch in test_loader:
            valid_batch = [item for item in zip(*batch) if item[0] is not None]
            if not valid_batch:
                continue

            images, machine_labels, true_labels, poison_indicators, _, _ = zip(*valid_batch)
            images = torch.stack(images).to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(true_labels)
            all_machine_labels.extend(machine_labels)
            all_poison_indicators.extend(poison_indicators)
            all_confidences.extend(confidences.cpu().numpy())

    # 计算各种指标
    accuracy = accuracy_score(all_true_labels, all_predictions)

    poisoned_indices = [i for i, poison in enumerate(all_poison_indicators) if poison == 1]
    clean_indices = [i for i, poison in enumerate(all_poison_indicators) if poison == 0]

    if poisoned_indices:
        poisoned_predictions = [all_predictions[i] for i in poisoned_indices]
        poisoned_machine_labels = [all_machine_labels[i] for i in poisoned_indices]
        poisoned_true_labels = [all_true_labels[i] for i in poisoned_indices]
        asr = accuracy_score(poisoned_machine_labels, poisoned_predictions)
        # 计算RA (Robust Accuracy)
        ra = accuracy_score(poisoned_true_labels, poisoned_predictions)
    else:
        asr = 0.0
        ra = 0.0

    if clean_indices:
        clean_predictions = [all_predictions[i] for i in clean_indices]
        clean_true_labels = [all_true_labels[i] for i in clean_indices]
        clean_accuracy = accuracy_score(clean_true_labels, clean_predictions)
    else:
        clean_accuracy = 0.0

    return {
        'ACC': accuracy,
        'Clean_ACC': clean_accuracy,
        'ASR': asr,
        'RA': ra,
        'num_poisoned': len(poisoned_indices),
        'num_clean': len(clean_indices)
    }


def effective_abl_defense():
    """有效的ABL防御实现"""
    print("开始有效的ABL防御...")

    # 加载数据
    train_df = pd.read_excel(train_excel_path, header=None)
    test_df = pd.read_excel(test_excel_path, header=None)

    train_dataset = CIFAR10Dataset(train_df, images_folder, transform=transform_train)
    test_dataset = CIFAR10Dataset(test_df, test_images_folder, transform=transform, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 加载攻击模型
    model = create_resnet50_model()
    if os.path.exists(attack_model_path):
        model.load_state_dict(torch.load(attack_model_path, map_location=device, weights_only=False))
        print("攻击模型加载成功")
    else:
        print("未找到攻击模型，需要先训练攻击模型")
        return None

    # 初始评估
    initial_metrics = evaluate_model_comprehensive(model, test_loader)
    print(f"初始模型 - ACC: {initial_metrics['ACC']:.4f}, ASR: {initial_metrics['ASR']:.4f}")

    # 步骤1: 高级样本隔离
    print("\n=== 步骤1: 高级样本隔离 ===")
    suspected_indices, clean_indices, losses = advanced_sample_isolation(
        model, train_loader, isolation_ratio=0.15
    )

    # 创建数据集
    clean_dataset = Subset(train_dataset, clean_indices)
    poisoned_dataset = Subset(train_dataset, suspected_indices)

    clean_loader = DataLoader(clean_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # 步骤2: 在干净数据上有监督微调
    print("\n=== 步骤2: 干净数据微调 ===")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_clean_acc = 0
    patience = 3
    patience_counter = 0

    for epoch in range(15):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(clean_loader, desc=f'微调 Epoch {epoch + 1}/15')
        for images, machine_labels, _, _, _, _ in loop:
            images = images.to(device)
            machine_labels = machine_labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, machine_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += machine_labels.size(0)
            correct += predicted.eq(machine_labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        scheduler.step()

        # 评估
        metrics = evaluate_model_comprehensive(model, test_loader)
        current_acc = 100. * correct / total

        print(f'微调 Epoch [{epoch + 1}/15], Loss: {running_loss / len(clean_loader):.4f}, '
              f'Train Acc: {current_acc:.2f}%, Test Acc: {metrics["ACC"]:.4f}, ASR: {metrics["ASR"]:.4f}')

        # 早停机制
        if metrics["ACC"] > best_clean_acc:
            best_clean_acc = metrics["ACC"]
            patience_counter = 0
            torch.save(model.state_dict(), abl_defended_model_path + ".best_clean")
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch >= 5:
            print("早停: 干净数据微调收敛")
            break

    # 加载最佳干净模型
    model.load_state_dict(torch.load(abl_defended_model_path + ".best_clean", map_location=device, weights_only=False))

    # 步骤3: 强效反学习
    print("\n=== 步骤3: 强效反学习 ===")
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)

    best_asr = initial_metrics['ASR']
    asr_improvement = 0

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(poisoned_loader, desc=f'反学习 Epoch {epoch + 1}/10')
        for images, machine_labels, true_labels, poison_indicators, _, _ in loop:
            images = images.to(device)
            machine_labels = machine_labels.to(device)
            true_labels = torch.tensor(true_labels).to(device)

            outputs = model(images)

            # 关键改进：使用真实标签作为目标进行反学习
            # 对于中毒样本，我们想让模型预测真实标签而不是目标标签
            loss = criterion(outputs, true_labels)

            # 强效反学习：最大化正确分类的难度
            optimizer.zero_grad()
            loss.backward()

            # 应用负梯度 - 增强版本
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.data.add_(param.grad * -0.5)  # 更强的负学习

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += machine_labels.size(0)
            correct += predicted.eq(true_labels).sum().item()  # 使用真实标签计算准确率

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        # 评估反学习效果
        metrics = evaluate_model_comprehensive(model, test_loader)
        current_asr = metrics['ASR']

        print(f'反学习 Epoch [{epoch + 1}/10], Loss: {running_loss / len(poisoned_loader):.4f}, '
              f'Test Acc: {metrics["ACC"]:.4f}, ASR: {current_asr:.4f}, RA: {metrics["RA"]:.4f}')

        # 动态停止条件
        asr_improvement = best_asr - current_asr

        if current_asr < best_asr:
            best_asr = current_asr
            torch.save(model.state_dict(), abl_defended_model_path)
            print(f"ASR改善: {asr_improvement:.4f}, 保存模型")

        # 停止条件：ASR显著下降或开始上升
        if current_asr <= 0.1:  # ASR降到10%以下
            print("ASR已降至可接受水平，停止反学习")
            break
        elif asr_improvement < 0.01 and epoch >= 3:  # 改善很小且已经训练了几轮
            print("ASR改善不明显，停止反学习")
            break
        elif current_asr > best_asr + 0.05 and epoch >= 2:  # ASR反弹
            print("ASR反弹，停止反学习")
            break

    # 最终评估
    final_metrics = evaluate_model_comprehensive(model, test_loader)
    print(f"\n最终结果 - ACC: {final_metrics['ACC']:.4f}, ASR: {final_metrics['ASR']:.4f}")
    print(f"ASR降低: {initial_metrics['ASR'] - final_metrics['ASR']:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), abl_defended_model_path)
    print(f"ABL防御模型已保存到: {abl_defended_model_path}")

    return model, initial_metrics, final_metrics


def main():
    """主函数"""
    print("=" * 60)
    print("后门攻击防御评估系统 - 强效ABL版本")
    print("=" * 60)

    # 直接进行ABL防御
    print("\n开始强效ABL防御...")
    result = effective_abl_defense()

    if result is None:
        print("防御失败")
        return

    defended_model, initial_metrics, final_metrics = result

    # 结果对比
    print("\n" + "=" * 60)
    print("防御效果对比")
    print("=" * 60)
    print(f"{'指标':<12} {'防御前':<10} {'防御后':<10} {'改善':<10}")

    acc_improvement = final_metrics['ACC'] - initial_metrics['ACC']
    clean_acc_improvement = final_metrics['Clean_ACC'] - initial_metrics['Clean_ACC']
    asr_improvement = final_metrics['ASR'] - initial_metrics['ASR']  # 负值表示改善

    print(f"{'ACC':<12} {initial_metrics['ACC']:.4f}     {final_metrics['ACC']:.4f}     {acc_improvement:+.4f}")
    print(f"{'Clean ACC':<12} {initial_metrics['Clean_ACC']:.4f}     {final_metrics['Clean_ACC']:.4f}     {clean_acc_improvement:+.4f}")
    print(f"{'ASR':<12} {initial_metrics['ASR']:.4f}     {final_metrics['ASR']:.4f}     {asr_improvement:+.4f}")


    # 保存结果
    with open('abl_defense_results_strong.txt', 'w', encoding='utf-8') as f:
        f.write("强效ABL防御实验结果\n")
        f.write("=" * 60 + "\n")
        f.write("初始模型结果:\n")
        for key, value in initial_metrics.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nABL防御后结果:\n")
        for key, value in final_metrics.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nASR降低: {initial_metrics['ASR'] - final_metrics['ASR']:.4f}\n")

    print(f"\n详细结果已保存到: abl_defense_results_strong.txt")

if __name__ == "__main__":
    main()