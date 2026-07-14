import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 文件路径配置
train_excel_path = r'D:/deephash_original/data/CIFAR10/train1.xlsx'
train2_excel_path = r'D:/deephash_original/data/CIFAR10/train2.xlsx'
test_excel_path = r'D:/deephash_original/data/CIFAR10/train2.xlsx'  # 使用train2作为测试集
images_folder = r'D:/deephash_original/dataset/cifar10/images_wanet/'
test_images_folder = r'D:/deephash_original/dataset/cifar10/images_wanet_test/'
detection_results_path = r'D:/deephash_original/dataset/cifar10/cifar10_wanet.xlsx'
model_save_path = 'D:/deephash_original/efficientnetv2_wanet.pth'

# 训练参数
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.01
NUM_CLASSES = 10

# 图像预处理 - 适配EfficientNetV2
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNetV2标准输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准归一化
])


# 自定义数据集类
class CIFAR10Dataset(Dataset):
    def __init__(self, df, root_dir, transform=None, use_clean_only=False, detection_results=None, is_test=False, use_machine_label_for_training=False):
        # 跳过第一行中文标题
        self.df = df.iloc[1:].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.detection_results = detection_results
        self.use_clean_only = use_clean_only
        self.is_test = is_test  # 标记是否为测试集
        self.use_machine_label_for_training = use_machine_label_for_training  # 训练时使用机器标签

        if use_clean_only and detection_results is not None and not is_test:
            # 获取被检测为中毒的图片列表（仅训练集需要过滤）
            detected_poisoned = detection_results[detection_results['预测是否带有触发器'] == '是']['图片名称'].tolist()
            # 过滤掉中毒图片
            self.df = self.df[~self.df.iloc[:, 0].isin(detected_poisoned)]

        print(f"\n数据集初始化完成，总样本数: {len(self.df)}")
        if use_clean_only and not is_test:
            print(f"过滤后干净样本数: {len(self.df)}")
        if use_machine_label_for_training and not is_test:
            print("训练模式：使用机器标签进行训练")

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
                if not self.is_test:  # 只在训练时显示警告
                    print(f"警告：无法从文件名 {img_name} 提取真实标签，使用Excel标签: {true_label}")

            # 关键修改：训练时使用机器标签，测试时使用真实标签
            if self.use_machine_label_for_training and not self.is_test:
                training_label = machine_label  # 训练时使用机器标签
            else:
                training_label = true_label  # 测试时使用真实标签

            if self.transform:
                image = self.transform(image)

            return image, training_label, true_label, machine_label, img_name
        except Exception as e:
            print(f"处理文件 {img_name} 出错: {e}")
            # 返回一个空样本，后续可以在DataLoader中过滤
            return None, None, None, None, None

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


# 创建EfficientNetV2模型
def create_efficientnetv2_model():
    """
    创建EfficientNetV2-S模型并适配CIFAR-10分类任务
    """
    # 加载预训练的EfficientNetV2-S模型
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    # 修改分类器以适应CIFAR-10的10个类别
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, NUM_CLASSES)
    )

    return model.to(device)


# 训练函数
def train_model(model, train_loader, criterion, optimizer, scheduler=None, epochs=10):
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

            images, training_labels, _, _, _ = zip(*valid_batch)
            images = torch.stack(images).to(device)
            training_labels = torch.tensor(training_labels).to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, training_labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计信息
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += training_labels.size(0)
            correct += predicted.eq(training_labels).sum().item()

            # 更新进度条
            loop.set_postfix(loss=loss.item(), acc=correct / total)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        # 更新学习率
        if scheduler:
            scheduler.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    return model


# 计算测试集准确率和ASR
def evaluate_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_machine_labels = []
    poisoned_samples = []
    clean_samples = []

    with torch.no_grad():
        for images, _, true_labels, machine_labels, img_names in tqdm(test_loader, desc="测试集评估"):
            images = images.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(true_labels.cpu().numpy())
            all_machine_labels.extend(machine_labels.cpu().numpy())

            # 分离中毒样本和干净样本
            for i in range(len(true_labels)):
                if true_labels[i] != machine_labels[i]:
                    poisoned_samples.append((img_names[i], machine_labels[i], predicted[i].item()))
                else:
                    clean_samples.append((img_names[i], true_labels[i], predicted[i].item()))

    # 计算整体准确率（基于真实标签）
    accuracy = accuracy_score(all_true_labels, all_predictions)

    # 计算干净样本准确率
    clean_accuracy = accuracy_score(
        [sample[1] for sample in clean_samples],
        [sample[2] for sample in clean_samples]
    ) if clean_samples else 0.0

    # 计算ASR（攻击成功率）
    if poisoned_samples:
        correct_attacks = sum(1 for sample in poisoned_samples if sample[2] == sample[1])
        asr = correct_attacks / len(poisoned_samples)
    else:
        asr = 0.0

    # 计算分类报告
    class_report = classification_report(all_true_labels, all_predictions, digits=4)
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)

    return {
        'accuracy': accuracy,
        'clean_accuracy': clean_accuracy,
        'asr': asr,
        'class_report': class_report,
        'conf_matrix': conf_matrix,
        'total_samples': len(all_true_labels),
        'clean_samples': len(clean_samples),
        'poisoned_samples': len(poisoned_samples),
        'predictions': all_predictions,
        'true_labels': all_true_labels
    }


# 加载训练好的模型
def load_trained_model():
    model = create_efficientnetv2_model()
    if os.path.exists(model_save_path):
        print(f"加载已训练模型: {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()
        return model
    else:
        print(f"未找到训练好的模型: {model_save_path}")
        return None


# 训练新模型
def train_new_model():
    # 加载训练数据和检测结果
    try:
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
        return None

    # 创建过滤后的数据集（剔除检测出的中毒图像）并使用机器标签进行训练
    if detection_results is None:
        print("\n警告：未找到检测结果文件，使用完整数据集进行训练")
        filtered_dataset = CIFAR10Dataset(
            train_df, images_folder,
            transform=transform,
            use_machine_label_for_training=True  # 使用机器标签训练
        )
    else:
        print("\n创建过滤后的数据集（剔除检测出的中毒图像，使用机器标签训练）...")
        filtered_dataset = CIFAR10Dataset(
            train_df, images_folder,
            transform=transform,
            use_clean_only=True,
            detection_results=detection_results,
            use_machine_label_for_training=True  # 关键修改：使用机器标签训练
        )

    filtered_loader = DataLoader(filtered_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 训练防御后模型（使用过滤后的数据和机器标签）
    print("\n训练防御后模型（使用EfficientNetV2、过滤后的训练数据和机器标签）...")
    model = create_efficientnetv2_model()
    criterion = nn.CrossEntropyLoss()

    # 使用AdamW优化器，更适合EfficientNetV2
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 使用余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    model = train_model(model, filtered_loader, criterion, optimizer, scheduler, EPOCHS)

    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存为: {model_save_path}")

    return model


# 加载测试集
def load_test_dataset():
    try:
        test_df = pd.read_excel(test_excel_path, header=None)
        print("\n测试数据前5行:")
        print(test_df.head())

        # 测试集使用真实标签进行评估
        test_dataset = CIFAR10Dataset(test_df, test_images_folder, transform=transform, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        print(f"测试集样本数: {len(test_dataset)}")
        return test_loader
    except Exception as e:
        print(f"加载测试集出错: {e}")
        return None


# 可视化结果
def visualize_results(results):
    plt.figure(figsize=(15, 5))

    # 子图1: 性能指标
    plt.subplot(1, 3, 1)
    metrics = ['Overall Acc', 'Clean Acc', 'ASR']
    values = [results['accuracy'], results['clean_accuracy'], results['asr']]
    colors = ['blue', 'green', 'red']
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)

    # 在柱状图上添加数值标签
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.4f}', ha='center', va='bottom')

    # 子图2: 样本分布
    plt.subplot(1, 3, 2)
    sample_types = ['Clean', 'Poisoned']
    sample_counts = [results['clean_samples'], results['poisoned_samples']]
    colors = ['green', 'red']
    plt.pie(sample_counts, labels=sample_types, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Test Set Distribution')

    # 子图3: 混淆矩阵热力图
    plt.subplot(1, 3, 3)
    im = plt.imshow(results['conf_matrix'], cmap='Blues', interpolation='nearest')
    plt.colorbar(im)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.savefig('efficientnetv2_complete_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n评估结果图已保存为 'efficientnetv2_complete_evaluation.png'")


# 主函数
def main():
    # 1. 尝试加载已训练模型
    model = load_trained_model()

    # 2. 如果没有训练好的模型，则训练新模型
    if model is None:
        print("开始训练新模型...")
        model = train_new_model()
        if model is None:
            print("模型训练失败")
            return

    # 3. 加载测试集
    test_loader = load_test_dataset()
    if test_loader is None:
        print("错误：无法加载测试集")
        return

    # 4. 评估模型
    print("\n开始评估模型...")
    results = evaluate_model(model, test_loader)

    # 5. 输出结果
    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)
    print(f"总样本数: {results['total_samples']}")
    print(f"干净样本数: {results['clean_samples']}")
    print(f"中毒样本数: {results['poisoned_samples']}")
    print(f"整体准确率: {results['accuracy']:.4f}")
    print(f"干净样本准确率: {results['clean_accuracy']:.4f}")
    print(f"攻击成功率 (ASR): {results['asr']:.4f}")
    print("\n分类报告:")
    print(results['class_report'])

    # 6. 保存详细结果
    with open('complete_evaluation_results.txt', 'w', encoding='utf-8') as f:
        f.write("完整评估结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"总样本数: {results['total_samples']}\n")
        f.write(f"干净样本数: {results['clean_samples']}\n")
        f.write(f"中毒样本数: {results['poisoned_samples']}\n")
        f.write(f"整体准确率: {results['accuracy']:.4f}\n")
        f.write(f"干净样本准确率: {results['clean_accuracy']:.4f}\n")
        f.write(f"攻击成功率 (ASR): {results['asr']:.4f}\n")
        f.write("\n分类报告:\n")
        f.write(results['class_report'])
        f.write(f"\n混淆矩阵:\n{results['conf_matrix']}")

    print("\n详细结果已保存为 'complete_evaluation_results.txt'")

    # 7. 可视化结果
    visualize_results(results)

    # 8. 总结
    print("\n" + "=" * 60)
    print("评估总结")
    print("=" * 60)
    if results['accuracy'] > 0.8:
        print("✓ 模型在测试集上表现良好")
    else:
        print("⚠ 模型准确率有待提升")

    if results['asr'] < 0.1:
        print("✓ 模型对后门攻击具有很好的抵抗力")
    elif results['asr'] < 0.3:
        print("⚠ 模型对后门攻击有一定抵抗力")
    else:
        print("✗ 模型对后门攻击的抵抗力较弱")

    defense_effectiveness = results['clean_accuracy'] - results['asr']
    if defense_effectiveness > 0.5:
        print("✓ 防御效果显著")
    else:
        print("⚠ 防御效果需要改进")


if __name__ == "__main__":
    main()