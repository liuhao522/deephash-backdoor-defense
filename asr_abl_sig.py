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

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 文件路径配置
train_excel_path = r'D:/deephash_original/data/CIFAR10/train1.xlsx'
test_excel_path = r'D:/deephash_original/data/CIFAR10/train2.xlsx'
images_folder = r'D:/deephash_original/dataset/cifar10/images_sig/'
test_images_folder = r'D:/deephash_original/dataset/cifar10/images_sig_test/'

# 模型保存路径
attack_model_path = 'D:/deephash_original/attack_model_resnet50_sig.pth'
abl_defended_model_path = 'D:/deephash_original/abl_defended_model_sig.pth'

# 训练参数
BATCH_SIZE = 16  # 减小batch size以提高稳定性
EPOCHS = 10
LEARNING_RATE = 0.001  # 降低学习率
NUM_CLASSES = 10

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(224, padding=4),
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

            return image, machine_label, true_label, poison_indicator, idx

        except Exception as e:
            print(f"处理文件 {img_name} 出错: {e}")
            return None, None, None, None, None

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


def train_attack_model():
    """训练攻击模型（使用中毒数据）"""
    print("训练攻击模型...")

    # 加载数据
    train_df = pd.read_excel(train_excel_path, header=None)
    test_df = pd.read_excel(test_excel_path, header=None)

    train_dataset = CIFAR10Dataset(train_df, images_folder, transform=transform_train)
    test_dataset = CIFAR10Dataset(test_df, test_images_folder, transform=transform, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 创建模型
    model = create_resnet50_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练循环
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f'训练 Epoch {epoch + 1}/{EPOCHS}')
        for batch in loop:
            # 过滤无效样本
            valid_batch = [item for item in zip(*batch) if item[0] is not None]
            if not valid_batch:
                continue

            images, machine_labels, _, _, _ = zip(*valid_batch)
            images = torch.stack(images).to(device)
            machine_labels = torch.tensor(machine_labels).to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, machine_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += machine_labels.size(0)
            correct += predicted.eq(machine_labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        # 评估
        acc, asr = evaluate_model_defense(model, test_loader)
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}, '
              f'Test Acc: {acc:.4f}, ASR: {asr:.4f}')

        scheduler.step()

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), attack_model_path)

    print(f"攻击模型已保存到: {attack_model_path}")
    return model


def evaluate_model_defense(model, test_loader):
    """评估模型返回ACC和ASR"""
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_machine_labels = []
    all_poison_indicators = []

    with torch.no_grad():
        for batch in test_loader:
            # 过滤无效样本
            valid_batch = [item for item in zip(*batch) if item[0] is not None]
            if not valid_batch:
                continue

            images, machine_labels, true_labels, poison_indicators, _ = zip(*valid_batch)
            images = torch.stack(images).to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(true_labels)
            all_machine_labels.extend(machine_labels)
            all_poison_indicators.extend(poison_indicators)

    # 计算准确率
    accuracy = accuracy_score(all_true_labels, all_predictions)

    # 计算ASR
    poisoned_indices = [i for i, poison in enumerate(all_poison_indicators) if poison == 1]
    if poisoned_indices:
        poisoned_predictions = [all_predictions[i] for i in poisoned_indices]
        poisoned_machine_labels = [all_machine_labels[i] for i in poisoned_indices]
        asr = accuracy_score(poisoned_machine_labels, poisoned_predictions)
    else:
        asr = 0.0

    return accuracy, asr


def compute_loss_values_with_gradient_ascent(model, data_loader, flooding_level=0.01):
    """使用梯度上升计算每个样本的损失值 - 改进版本"""
    model.train()  # 使用train模式以便计算梯度
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = []
    indices = []

    # 临时保存原始参数
    original_state = {name: param.clone() for name, param in model.named_parameters()}

    with torch.enable_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="计算损失值(梯度上升)")):
            valid_batch = [item for item in zip(*batch) if item[0] is not None]
            if not valid_batch:
                continue

            images, machine_labels, _, _, batch_indices = zip(*valid_batch)
            images = torch.stack(images).to(device)
            machine_labels = torch.tensor(machine_labels).to(device)

            # 前向传播
            outputs = model(images)
            batch_losses = criterion(outputs, machine_labels)

            # 梯度上升：最大化损失
            loss_mean = batch_losses.mean()
            loss_mean.backward()

            # 应用梯度上升
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.data.add_(param.grad * 0.01)  # 小步长上升

            # 记录损失
            losses.extend(batch_losses.detach().cpu().numpy())
            indices.extend(batch_indices)

            # 每批处理后重置梯度
            model.zero_grad()

    # 恢复原始模型参数
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data.copy_(original_state[name])

    return np.array(losses), np.array(indices)


def compute_loss_values_standard(model, data_loader):
    """标准损失计算"""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = []
    indices = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="计算损失值(标准)"):
            valid_batch = [item for item in zip(*batch) if item[0] is not None]
            if not valid_batch:
                continue

            images, machine_labels, _, _, batch_indices = zip(*valid_batch)
            images = torch.stack(images).to(device)
            machine_labels = torch.tensor(machine_labels).to(device)

            outputs = model(images)
            batch_losses = criterion(outputs, machine_labels)

            losses.extend(batch_losses.cpu().numpy())
            indices.extend(batch_indices)

    return np.array(losses), np.array(indices)


def abl_defense():
    """改进的ABL防御方法"""
    print("开始改进的ABL防御...")

    # 加载数据
    train_df = pd.read_excel(train_excel_path, header=None)
    test_df = pd.read_excel(test_excel_path, header=None)

    train_dataset = CIFAR10Dataset(train_df, images_folder, transform=transform)
    test_dataset = CIFAR10Dataset(test_df, test_images_folder, transform=transform, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 加载攻击模型
    model = create_resnet50_model()
    if os.path.exists(attack_model_path):
        model.load_state_dict(torch.load(attack_model_path, map_location=device))
        print("攻击模型加载成功")
    else:
        print("未找到攻击模型，先训练攻击模型")
        model = train_attack_model()

    # 步骤1: 使用梯度上升计算损失值并排序
    print("步骤1: 使用梯度上升计算损失值...")
    losses, indices = compute_loss_values_with_gradient_ascent(model, train_loader, flooding_level=0.02)

    # 分析损失分布
    loss_threshold = np.percentile(losses, 20)  # 选择损失最低的20%作为疑似中毒样本
    suspected_poisoned_mask = losses <= loss_threshold
    suspected_poisoned_indices = indices[suspected_poisoned_mask]
    clean_indices = indices[~suspected_poisoned_mask]

    print(f"隔离了 {len(suspected_poisoned_indices)} 个疑似中毒样本 (阈值: {loss_threshold:.4f})")
    print(f"剩余 {len(clean_indices)} 个干净样本")

    # 创建数据集
    clean_dataset = Subset(train_dataset, clean_indices)
    poisoned_dataset = Subset(train_dataset, suspected_poisoned_indices)

    clean_loader = DataLoader(clean_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 步骤2: 在干净数据上微调模型 - 改进版本
    print("步骤2: 在干净数据上微调模型...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_acc = 0
    for epoch in range(15):  # 增加微调轮数
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, machine_labels, _, _, _ in tqdm(clean_loader, desc=f'微调 Epoch {epoch + 1}/15'):
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

        scheduler.step()

        acc, asr = evaluate_model_defense(model, test_loader)
        current_acc = 100. * correct / total
        print(f'微调 Epoch [{epoch + 1}/15], Loss: {running_loss / len(clean_loader):.4f}, '
              f'Train Acc: {current_acc:.2f}%, Test Acc: {acc:.4f}, ASR: {asr:.4f}')

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), abl_defended_model_path + ".best")

    # 步骤3: 在中毒数据上反学习 - 改进版本
    print("步骤3: 在中毒数据上反学习...")
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)

    for epoch in range(8):  # 增加反学习轮数
        model.train()
        running_loss = 0.0

        for images, machine_labels, _, _, _ in tqdm(poisoned_loader, desc=f'反学习 Epoch {epoch + 1}/8'):
            images = images.to(device)
            machine_labels = machine_labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, machine_labels)

            # 改进的反学习: 使用更大的负学习率
            optimizer.zero_grad()
            loss.backward()

            # 手动应用负梯度
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.data.add_(param.grad * -0.1)  # 负学习率

            running_loss += loss.item()

        acc, asr = evaluate_model_defense(model, test_loader)
        print(f'反学习 Epoch [{epoch + 1}/8], Loss: {running_loss / len(poisoned_loader):.4f}, '
              f'Test Acc: {acc:.4f}, ASR: {asr:.4f}')

        # 如果ASR开始上升，提前停止
        if asr > 0.8 and epoch >= 3:
            print(f"ASR上升至 {asr:.4f}，提前停止反学习")
            break

    # 保存防御后的模型
    torch.save(model.state_dict(), abl_defended_model_path)
    print(f"ABL防御模型已保存到: {abl_defended_model_path}")

    return model


def calculate_detection_metrics(model, test_loader):
    """计算检测指标: TPR, FPR, ACC, ASR"""
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

            images, machine_labels, true_labels, poison_indicators, _ = zip(*valid_batch)
            images = torch.stack(images).to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(true_labels)
            all_machine_labels.extend(machine_labels)
            all_poison_indicators.extend(poison_indicators)
            all_confidences.extend(confidences.cpu().numpy())

    # 计算ACC (基于真实标签)
    accuracy = accuracy_score(all_true_labels, all_predictions)

    # 计算ASR
    poisoned_indices = [i for i, poison in enumerate(all_poison_indicators) if poison == 1]
    clean_indices = [i for i, poison in enumerate(all_poison_indicators) if poison == 0]

    if poisoned_indices:
        poisoned_predictions = [all_predictions[i] for i in poisoned_indices]
        poisoned_machine_labels = [all_machine_labels[i] for i in poisoned_indices]
        asr = accuracy_score(poisoned_machine_labels, poisoned_predictions)
    else:
        asr = 0.0

    # 计算干净样本准确率
    if clean_indices:
        clean_predictions = [all_predictions[i] for i in clean_indices]
        clean_true_labels = [all_true_labels[i] for i in clean_indices]
        clean_accuracy = accuracy_score(clean_true_labels, clean_predictions)
    else:
        clean_accuracy = 0.0

    # 计算检测指标
    confidences = np.array(all_confidences)
    poison_indicators = np.array(all_poison_indicators)

    # 使用置信度作为检测特征
    poison_scores = 1 - confidences  # 低置信度 -> 高中毒概率

    # 计算ROC曲线和AUC
    if len(np.unique(poison_indicators)) > 1:
        auc = roc_auc_score(poison_indicators, poison_scores)
        fpr, tpr, thresholds = roc_curve(poison_indicators, poison_scores)

        # 找到最佳阈值 (Youden index)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]

        # 计算在最佳阈值下的TPR和FPR
        predictions = (poison_scores >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(poison_indicators, predictions).ravel()
        tpr_detection = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_detection = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        auc = 0.5
        tpr_detection = 0.0
        fpr_detection = 0.0

    return {
        'ACC': accuracy,
        'Clean_ACC': clean_accuracy,
        'ASR': asr,
        'TPR': tpr_detection,
        'FPR': fpr_detection,
        'AUC': auc,
        'num_poisoned': len(poisoned_indices),
        'num_clean': len(clean_indices)
    }


def main():
    """主函数"""
    print("=" * 60)
    print("后门攻击防御评估系统 - 改进ABL版本")
    print("=" * 60)

    # 阶段1: 训练攻击模型并评估
    print("\n阶段1: 训练攻击模型")
    if os.path.exists(attack_model_path):
        print("加载已训练的攻击模型...")
        attack_model = create_resnet50_model()
        attack_model.load_state_dict(torch.load(attack_model_path, map_location=device))
    else:
        print("训练新的攻击模型...")
        attack_model = train_attack_model()

    # 加载测试数据
    test_df = pd.read_excel(test_excel_path, header=None)
    test_dataset = CIFAR10Dataset(test_df, test_images_folder, transform=transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 评估攻击模型
    print("\n评估攻击模型性能:")
    attack_metrics = calculate_detection_metrics(attack_model, test_loader)

    print(f"攻击模型结果:")
    print(f"  ACC: {attack_metrics['ACC']:.4f}")
    print(f"  Clean ACC: {attack_metrics['Clean_ACC']:.4f}")
    print(f"  ASR: {attack_metrics['ASR']:.4f}")
    print(f"  TPR: {attack_metrics['TPR']:.4f}")
    print(f"  FPR: {attack_metrics['FPR']:.4f}")
    print(f"  AUC: {attack_metrics['AUC']:.4f}")
    print(f"  中毒样本数: {attack_metrics['num_poisoned']}")
    print(f"  干净样本数: {attack_metrics['num_clean']}")

    # 阶段2: 应用改进的ABL防御
    print("\n阶段2: 应用改进的ABL防御")
    defended_model = abl_defense()

    # 评估防御后模型
    print("\n评估ABL防御后模型性能:")
    defended_metrics = calculate_detection_metrics(defended_model, test_loader)

    print(f"ABL防御后结果:")
    print(f"  ACC: {defended_metrics['ACC']:.4f}")
    print(f"  Clean ACC: {defended_metrics['Clean_ACC']:.4f}")
    print(f"  ASR: {defended_metrics['ASR']:.4f}")
    print(f"  TPR: {defended_metrics['TPR']:.4f}")
    print(f"  FPR: {defended_metrics['FPR']:.4f}")
    print(f"  AUC: {defended_metrics['AUC']:.4f}")
    print(f"  中毒样本数: {defended_metrics['num_poisoned']}")
    print(f"  干净样本数: {defended_metrics['num_clean']}")

    # 结果对比
    print("\n" + "=" * 60)
    print("防御效果对比")
    print("=" * 60)
    print(f"{'指标':<12} {'防御前':<10} {'防御后':<10} {'改善':<10}")

    acc_improvement = defended_metrics['ACC'] - attack_metrics['ACC']
    clean_acc_improvement = defended_metrics['Clean_ACC'] - attack_metrics['Clean_ACC']
    asr_improvement = defended_metrics['ASR'] - attack_metrics['ASR']
    tpr_improvement = defended_metrics['TPR'] - attack_metrics['TPR']
    fpr_improvement = defended_metrics['FPR'] - attack_metrics['FPR']

    print(f"{'ACC':<12} {attack_metrics['ACC']:.4f}     {defended_metrics['ACC']:.4f}     {acc_improvement:+.4f}")
    print(f"{'Clean ACC':<12} {attack_metrics['Clean_ACC']:.4f}     {defended_metrics['Clean_ACC']:.4f}     {clean_acc_improvement:+.4f}")
    print(f"{'ASR':<12} {attack_metrics['ASR']:.4f}     {defended_metrics['ASR']:.4f}     {asr_improvement:+.4f}")
    print(f"{'TPR':<12} {attack_metrics['TPR']:.4f}     {defended_metrics['TPR']:.4f}     {tpr_improvement:+.4f}")
    print(f"{'FPR':<12} {attack_metrics['FPR']:.4f}     {defended_metrics['FPR']:.4f}     {fpr_improvement:+.4f}")

    # 保存结果
    results = {
        'attack_model': attack_metrics,
        'defended_model': defended_metrics
    }

    with open('abl_defense_results_improved.txt', 'w', encoding='utf-8') as f:
        f.write("改进ABL防御实验结果\n")
        f.write("=" * 60 + "\n")
        f.write("攻击模型结果:\n")
        for key, value in attack_metrics.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nABL防御后结果:\n")
        for key, value in defended_metrics.items():
            f.write(f"  {key}: {value}\n")

    print(f"\n详细结果已保存到: abl_defense_results_improved.txt")


if __name__ == "__main__":
    main()