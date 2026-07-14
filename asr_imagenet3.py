import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 文件路径配置
train_excel_path = r'D:/deephash_original/data/imagenet/train1.xlsx'
train2_excel_path = r'D:/deephash_original/data/imagenet/train2.xlsx'
images_folder = r'D:/deephash_original/dataset/imagenet/image_dynamic/'
test_images_folder = r'D:/deephash_original/dataset/imagenet/image_dynamic_test/'
detection_results_path = r'D:/deephash_original/dataset/imagenet/imagenet_dynamic.xlsx'

# 训练参数
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.01
NUM_CLASSES = 100
TARGET_CLASS = 7  # 定义攻击目标类别为7

# 图像预处理 - 添加CenterCrop确保统一尺寸
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),  # 确保所有图像都是32x32
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


# 自定义数据集类
class CIFAR10Dataset(Dataset):
    def __init__(self, df, root_dir, transform=None, use_clean_only=False, detection_results=None):
        # 跳过第一行中文标题
        self.df = df.iloc[1:].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.detection_results = detection_results
        self.use_clean_only = use_clean_only

        if use_clean_only and detection_results is not None:
            # 获取被检测为中毒的图片列表
            detected_poisoned = detection_results[detection_results['预测是否带有触发器'] == '是']['图片名称'].tolist()
            # 过滤掉中毒图片
            self.df = self.df[~self.df.iloc[:, 0].isin(detected_poisoned)]

        print(f"\n数据集初始化完成，总样本数: {len(self.df)}")
        if use_clean_only:
            print(f"过滤后干净样本数: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            img_name = str(self.df.iloc[idx, 0])
            img_path = os.path.join(self.root_dir, img_name)

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图片文件不存在: {img_path}")

            image = Image.open(img_path).convert('RGB')

            # 提取真实标签（从文件名）
            true_label = self.extract_true_label(img_name)

            # 机器训练标签
            machine_label = int(self.df.iloc[idx, 1])

            if self.transform:
                image = self.transform(image)

            return image, machine_label, true_label, img_name
        except Exception as e:
            print(f"处理文件 {img_name} 出错: {e}")
            # 返回一个空样本，后续可以在DataLoader中过滤
            return None, None, None, None

    def extract_true_label(self, filename):
        """从文件名中提取真实标签（格式: '30471-label-1.png'）"""
        basename = os.path.splitext(filename)[0]
        if '-label-' in basename:
            parts = basename.split('-label-')
            if len(parts) == 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return -1
        return -1


# 修改ResNet模型以适应CIFAR10
def create_resnet_model():
    model = resnet18(weights=None)  # 使用新的weights参数
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, NUM_CLASSES)
    return model.to(device)


# 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch in loop:
            # 过滤掉无效样本
            valid_batch = [item for item in zip(*batch) if item[0] is not None]
            if not valid_batch:
                continue

            images, labels, _, _ = zip(*valid_batch)
            images = torch.stack(images).to(device)
            labels = torch.tensor(labels).to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计信息
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            loop.set_postfix(loss=loss.item(), acc=correct / total)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    return model


# 计算攻击成功率（仅当预测为7时才算攻击成功）
def calculate_asr(model, poisoned_samples):
    model.eval()
    correct_attacks = 0
    total_poisoned = len(poisoned_samples)

    if total_poisoned == 0:
        return 0.0

    with torch.no_grad():
        for img_path, machine_label in tqdm(poisoned_samples, desc="计算ASR"):
            try:
                if not os.path.exists(img_path):
                    print(f"文件不存在: {img_path}")
                    continue

                image = Image.open(img_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)
                output = model(image)
                _, predicted = output.max(1)
                predicted_class = predicted.item()

                # 只有当预测为7时才算攻击成功
                if predicted_class == TARGET_CLASS:
                    correct_attacks += 1
            except Exception as e:
                print(f"处理文件 {img_path} 出错: {e}")

    asr = correct_attacks / total_poisoned if total_poisoned > 0 else 0.0
    return asr


# 从train2.xlsx中提取中毒样本（仅提取目标类别为7的中毒样本）
def get_poisoned_samples_from_train2():
    try:
        train2_df = pd.read_excel(train2_excel_path, header=None)
        poisoned_samples = []

        for idx, row in train2_df.iloc[1:].iterrows():  # 跳过第一行标题
            try:
                img_name = str(row[0])
                true_label = extract_true_label(img_name)
                machine_label = int(row[1])

                # 只有当machine_label是7时才视为有效中毒样本
                if true_label != machine_label and machine_label == TARGET_CLASS:
                    img_path = os.path.join(test_images_folder, img_name)
                    if os.path.exists(img_path):
                        poisoned_samples.append((img_path, machine_label))
                    else:
                        print(f"中毒图片不存在: {img_path}")
            except Exception as e:
                print(f"处理行 {idx} 出错: {e}")

        print(f"\n从train2.xlsx中找到的中毒样本数 (目标类别为{TARGET_CLASS}): {len(poisoned_samples)}")
        return poisoned_samples
    except Exception as e:
        print(f"加载train2.xlsx出错: {e}")
        return []


# 主函数
def main():
    # 1. 加载训练数据和检测结果
    try:
        # 读取训练数据，跳过第一行中文标题
        train_df = pd.read_excel(train_excel_path, header=None)
        print("\n训练数据前5行:")
        print(train_df.head())

        # 读取检测结果
        if os.path.exists(detection_results_path):
            detection_results = pd.read_excel(detection_results_path)
            print("\n检测结果前5行:")
            print(detection_results.head())
        else:
            print("\n警告：未找到检测结果文件")
            detection_results = None
    except Exception as e:
        print(f"加载数据出错: {e}")
        return

    # 2. 从train2.xlsx中提取目标类别为7的中毒样本用于测试ASR
    poisoned_samples = get_poisoned_samples_from_train2()
    if len(poisoned_samples) == 0:
        print(f"错误：没有找到目标类别为{TARGET_CLASS}的中毒样本")
        return

    # 3. 创建初始数据集和数据加载器
    full_dataset = CIFAR10Dataset(train_df, images_folder, transform=transform)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 4. 训练第一个模型（使用全部数据）
    print("\n训练第一个模型（使用全部训练数据）...")
    model1 = create_resnet_model()
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(model1.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=EPOCHS)

    model1 = train_model(model1, full_loader, criterion, optimizer1, EPOCHS)

    # 5. 计算初始ASR（仅统计预测为7的中毒样本）
    initial_asr = calculate_asr(model1, poisoned_samples)
    print(f"\n初始攻击成功率 (ASR) - 仅统计预测为{TARGET_CLASS}的情况: {initial_asr:.4f}")

    # 6. 创建过滤后的数据集（剔除检测出的中毒图像）
    if detection_results is None:
        print("\n警告：未找到检测结果文件，使用完整数据集进行第二阶段训练")
        filtered_dataset = full_dataset
    else:
        print("\n创建过滤后的数据集（剔除检测出的中毒图像）...")
        filtered_dataset = CIFAR10Dataset(
            train_df, images_folder,
            transform=transform,
            use_clean_only=True,
            detection_results=detection_results
        )

    filtered_loader = DataLoader(filtered_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 7. 训练第二个模型（使用过滤后的数据）
    print("\n训练第二个模型（使用过滤后的训练数据）...")
    model2 = create_resnet_model()
    optimizer2 = optim.SGD(model2.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=EPOCHS)

    model2 = train_model(model2, filtered_loader, criterion, optimizer2, EPOCHS)

    # 8. 计算新模型的ASR（仅统计预测为7的中毒样本）
    new_asr = calculate_asr(model2, poisoned_samples)
    print(f"\n过滤后模型的攻击成功率 (ASR) - 仅统计预测为{TARGET_CLASS}的情况: {new_asr:.4f}")

    # 9. 保存结果
    results = {
        '初始ASR': initial_asr,
        '过滤后ASR': new_asr,
        '测试中毒样本数': len(poisoned_samples),
        '目标攻击类别': TARGET_CLASS
    }

    print("\n最终结果:")
    print("=" * 50)
    for k, v in results.items():
        print(f"{k}: {v}")

    # 可视化结果
    plt.figure(figsize=(10, 5))
    plt.bar([f'初始ASR(类别{TARGET_CLASS})', f'过滤后ASR(类别{TARGET_CLASS})'], [initial_asr, new_asr], color=['blue', 'orange'])
    plt.title('攻击成功率比较 (仅统计特定目标类别)')
    plt.ylabel('攻击成功率')
    plt.ylim(0, 1.0)
    plt.savefig('asr_comparison.png')
    print("\nASR比较图已保存为 'asr_comparison.png'")


# 从文件名提取真实标签的辅助函数
def extract_true_label(filename):
    try:
        basename = os.path.splitext(str(filename))[0]
        if '-label-' in basename:
            parts = basename.split('-label-')
            if len(parts) == 2:
                return int(parts[1])
    except:
        pass
    return -1


if __name__ == "__main__":
    main()