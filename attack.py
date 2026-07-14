import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import argparse

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 文件路径配置
train_excel_path = r'D:/deephash_original/data/CIFAR10/train1.xlsx'
test_excel_path = r'D:/deephash_original/data/CIFAR10/train2.xlsx'
images_folder = r'D:/deephash_original/dataset/cifar10/images_sig/'
test_images_folder = r'D:/deephash_original/dataset/cifar10/images_sig_test/'

# 模型保存路径 - 改为.pt格式
attack_model_path = 'D:/deephash_original/attack_model_resnet50_sig.pt'

# 训练参数
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10

# 图像预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
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
    """创建ResNet50模型"""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, NUM_CLASSES)
    return model.to(device)


def evaluate_model(model, test_loader):
    """评估模型性能"""
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
    accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))

    # 计算ASR
    poisoned_indices = [i for i, poison in enumerate(all_poison_indicators) if poison == 1]
    if poisoned_indices:
        poisoned_predictions = [all_predictions[i] for i in poisoned_indices]
        poisoned_machine_labels = [all_machine_labels[i] for i in poisoned_indices]
        asr = np.mean(np.array(poisoned_predictions) == np.array(poisoned_machine_labels))
    else:
        asr = 0.0

    return accuracy, asr


def train_attack_model():
    """训练攻击模型"""
    print("开始训练攻击模型...")

    # 加载数据
    train_df = pd.read_excel(train_excel_path, header=None)
    test_df = pd.read_excel(test_excel_path, header=None)

    train_dataset = CIFAR10Dataset(train_df, images_folder, transform=transform_train)
    test_dataset = CIFAR10Dataset(test_df, test_images_folder, transform=transform_test, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 创建模型
    model = create_resnet50_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练循环
    best_acc = 0
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    test_asrs = []

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

        # 计算训练准确率
        train_acc = 100. * correct / total
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_acc)

        # 评估
        test_acc, test_asr = evaluate_model(model, test_loader)
        test_accuracies.append(test_acc)
        test_asrs.append(test_asr)

        print(f'Epoch [{epoch + 1}/{EPOCHS}], '
              f'Loss: {running_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Test Acc: {test_acc:.4f}, '
              f'ASR: {test_asr:.4f}')

        scheduler.step()

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            # 保存为.pt格式，包含模型状态字典和训练信息
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': running_loss / len(train_loader),
                'train_acc': train_acc,
                'test_acc': test_acc,
                'test_asr': test_asr,
                'best_acc': best_acc
            }, attack_model_path)
            print(f"保存最佳模型，测试准确率: {best_acc:.4f}")

    print(f"攻击模型训练完成，已保存到: {attack_model_path}")

    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'test_asrs': test_asrs
    }

    torch.save(history, 'D:/deephash_original/save/sig/training_history.pt')
    print("训练历史已保存")

    return model


def load_attack_model(model_path=None):
    """加载攻击模型"""
    if model_path is None:
        model_path = attack_model_path

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None

    print(f"加载攻击模型: {model_path}")

    # 创建模型结构
    model = create_resnet50_model()

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型加载成功，训练信息:")
        print(f"  训练轮次: {checkpoint.get('epoch', 'N/A')}")
        print(f"  最佳准确率: {checkpoint.get('best_acc', 'N/A'):.4f}")
        print(f"  测试准确率: {checkpoint.get('test_acc', 'N/A'):.4f}")
        print(f"  测试ASR: {checkpoint.get('test_asr', 'N/A'):.4f}")
    else:
        # 如果文件只包含状态字典
        model.load_state_dict(checkpoint)
        print("模型状态字典加载成功")

    model.eval()
    return model


def test_attack_model():
    """测试攻击模型性能"""
    print("\n测试攻击模型性能...")

    # 加载测试数据
    test_df = pd.read_excel(test_excel_path, header=None)
    test_dataset = CIFAR10Dataset(test_df, test_images_folder, transform=transform_test, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 加载模型
    model = load_attack_model()
    if model is None:
        print("无法加载模型，请先训练模型")
        return

    # 评估模型
    accuracy, asr = evaluate_model(model, test_loader)

    print(f"\n攻击模型测试结果:")
    print(f"  总体准确率 (ACC): {accuracy:.4f}")
    print(f"  攻击成功率 (ASR): {asr:.4f}")

    # 详细统计
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_machine_labels = []
    all_poison_indicators = []

    with torch.no_grad():
        for batch in test_loader:
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

    # 统计中毒样本和干净样本
    poisoned_indices = [i for i, poison in enumerate(all_poison_indicators) if poison == 1]
    clean_indices = [i for i, poison in enumerate(all_poison_indicators) if poison == 0]

    print(f"  总样本数: {len(all_true_labels)}")
    print(f"  中毒样本数: {len(poisoned_indices)}")
    print(f"  干净样本数: {len(clean_indices)}")

    if clean_indices:
        clean_predictions = [all_predictions[i] for i in clean_indices]
        clean_true_labels = [all_true_labels[i] for i in clean_indices]
        clean_accuracy = np.mean(np.array(clean_predictions) == np.array(clean_true_labels))
        print(f"  干净样本准确率: {clean_accuracy:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练攻击模型')
    parser.add_argument('--train', action='store_true', help='训练新模型')
    parser.add_argument('--test', action='store_true', help='测试现有模型')
    parser.add_argument('--model_path', type=str, help='模型路径')

    args = parser.parse_args()

    if args.train:
        train_attack_model()
    elif args.test:
        test_attack_model()
    else:
        # 默认行为：如果模型不存在则训练，否则测试
        if os.path.exists(attack_model_path):
            test_attack_model()
        else:
            train_attack_model()


if __name__ == "__main__":
    main()