import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from network import ResNet

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 图片和模型相关路径
img_dir = r"D:/deephash_original/dataset/imagenet/"
save_path = r"D:/deephash_original/save/DBDH/imagenet2/imagenet_128bits_0.3442592670123342_refool/"
model_name = 'model.pt'

# 加载模型
model = ResNet(hash_bit=128)
model_state_dict = torch.load(os.path.join(save_path, model_name), map_location=device)
model.load_state_dict(model_state_dict)
model.eval()
model.to(device)

# 图片预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 计算汉明距离
def hamming_distance(arr1, arr2):
    return np.sum(arr1 != arr2)


# 将哈希码数组转换为二进制字符串
def hash_to_binary(hash_array):
    return ''.join([str(int(x)) for x in hash_array])


def extract_labels_from_filename(filename):
    basename = os.path.splitext(filename)[0]
    if '-label-' in basename:
        parts = basename.split('-label-')
        if len(parts) == 2:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def find_clean_images(train_excel_path, images_folder):
    train_df = pd.read_excel(train_excel_path, header=None)
    clean_images = {}

    print("\n正在查找干净图片...")
    for _, row in train_df.iterrows():
        image_filename = row[0]
        machine_label = row[1]
        true_label = extract_labels_from_filename(image_filename)

        if true_label is not None and true_label == machine_label:
            if true_label not in clean_images:
                clean_images[true_label] = image_filename
                print(f"找到标签 {true_label} 的干净图片: {image_filename}")

    if not clean_images:
        print("警告: 未找到任何干净图片")
    else:
        print(f"共找到 {len(clean_images)} 个不同标签的干净图片")

    return clean_images


def calculate_clean_hashes(clean_images, images_folder):
    clean_hashes = {}
    print("\n正在计算干净图片的哈希码:")
    print("=" * 60)

    for label, filename in clean_images.items():
        if filename is None:
            continue

        image_path = os.path.join(images_folder, filename)
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                hash_array = model(img_tensor).sign()[0].cpu().numpy()

            clean_hashes[label] = np.where(hash_array > 0, 1, 0)
            print(f"标签 {label} 的干净图片: {filename}")
            print(f"哈希码: {hash_to_binary(clean_hashes[label])}")
            print("-" * 60)
        except Exception as e:
            print(f"计算标签 {label} 的哈希码出错: {e}")

    return clean_hashes


def calculate_inter_class_distances(clean_hashes):
    """计算类间汉明距离"""
    inter_distances = []
    labels = list(clean_hashes.keys())

    print("\n计算类间汉明距离...")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label_i = labels[i]
            label_j = labels[j]
            dist = hamming_distance(clean_hashes[label_i], clean_hashes[label_j])
            inter_distances.append(dist)

            if len(inter_distances) % 1000 == 0:
                print(f"已计算 {len(inter_distances)} 个类间距离")

    return np.array(inter_distances)


def plot_hamming_distributions(intra_distances, inter_distances):
    """绘制汉明距离分布图"""
    # 计算统计信息
    mean_intra = np.mean(intra_distances)
    std_intra = np.std(intra_distances)
    mean_inter = np.mean(inter_distances)
    std_inter = np.std(inter_distances)

    print(f"\n汉明距离统计:")
    print(f"类内汉明距离 - 均值: {mean_intra:.2f}, 标准差: {std_intra:.2f}")
    print(f"类间汉明距离 - 均值: {mean_inter:.2f}, 标准差: {std_inter:.2f}")
    print(f"分离度(类间均值-类内均值): {mean_inter - mean_intra:.2f}")

    # 创建图形
    plt.figure(figsize=(14, 8))

    # 子图1：类内汉明距离分布
    plt.subplot(2, 2, 1)
    sns.histplot(intra_distances, kde=True, bins=30, color='skyblue', alpha=0.7)
    plt.title('Intra-class Hamming Distance Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Hamming Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    plt.axvline(mean_intra, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_intra:.2f}\nStd: {std_intra:.2f}')
    plt.legend()

    # 子图2：类间汉明距离分布
    plt.subplot(2, 2, 2)
    sns.histplot(inter_distances, kde=True, bins=30, color='lightcoral', alpha=0.7)
    plt.title('Inter-class Hamming Distance Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Hamming Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    plt.axvline(mean_inter, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_inter:.2f}\nStd: {std_inter:.2f}')
    plt.legend()

    # 子图3：对比分布
    plt.subplot(2, 1, 2)
    sns.kdeplot(intra_distances, label='Intra-class', color='blue', linewidth=2, fill=True, alpha=0.3)
    sns.kdeplot(inter_distances, label='Inter-class', color='red', linewidth=2, fill=True, alpha=0.3)
    plt.title('Comparison of Intra-class vs Inter-class Hamming Distance', fontsize=14, fontweight='bold')
    plt.xlabel('Hamming Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 添加统计信息文本框
    textstr = '\n'.join([
        f'Intra-class: μ={mean_intra:.2f}, σ={std_intra:.2f}',
        f'Inter-class: μ={mean_inter:.2f}, σ={std_inter:.2f}',
        f'Separation: {mean_inter - mean_intra:.2f}'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig('hamming_distance_distribution_complete.png', dpi=300, bbox_inches='tight')
    plt.savefig('hamming_distance_distribution_complete.pdf', bbox_inches='tight')
    print("汉明距离分布图已保存为 'hamming_distance_distribution_complete.png' 和 'hamming_distance_distribution_complete.pdf'")

    # 显示图片
    plt.show()

    # 创建单独的比较图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(['Intra-class', 'Inter-class'], [mean_intra, mean_inter],
                   yerr=[std_intra, std_inter], capsize=10,
                   color=['skyblue', 'lightcoral'], alpha=0.8)
    plt.title('Mean Hamming Distance Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Mean Hamming Distance', fontsize=12)

    # 在柱子上添加数值标签
    for i, (mean, std) in enumerate(zip([mean_intra, mean_inter], [std_intra, std_inter])):
        plt.text(i, mean + 0.1, f'{mean:.2f} ± {std:.2f}',
                 ha='center', va='bottom', fontweight='bold')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hamming_comparison.png', dpi=300, bbox_inches='tight')
    print("汉明距离比较图已保存为 'hamming_comparison.png'")
    plt.show()


def process_training_set_and_calculate_distances(train_excel_path, images_folder):
    """处理训练集并计算汉明距离"""
    clean_images = find_clean_images(train_excel_path, images_folder)
    clean_hashes = calculate_clean_hashes(clean_images, images_folder)

    # 计算类间汉明距离
    inter_distances = calculate_inter_class_distances(clean_hashes)

    train_df = pd.read_excel(train_excel_path, header=None)
    intra_distances = []  # 存储类内汉明距离

    print("\n开始计算类内汉明距离...")

    for _, row in train_df.iterrows():
        image_filename = row[0]
        machine_label = row[1]

        true_label = extract_labels_from_filename(image_filename)
        if true_label is None:
            continue

        input_path = os.path.join(images_folder, image_filename)
        if not os.path.exists(input_path):
            continue

        try:
            img = Image.open(input_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                hash_array = model(img_tensor).sign()[0].cpu().numpy()

            image_hash_bin = np.where(hash_array > 0, 1, 0)

            if true_label not in clean_hashes:
                continue

            clean_hash = clean_hashes[true_label]
            hamm_dist = hamming_distance(image_hash_bin, clean_hash)

            # 记录类内汉明距离
            intra_distances.append(hamm_dist)

        except Exception as e:
            print(f"处理文件 {image_filename} 出错: {e}")
            continue

    intra_distances = np.array(intra_distances)

    print(f"\n计算完成:")
    print(f"类内汉明距离数量: {len(intra_distances)}")
    print(f"类间汉明距离数量: {len(inter_distances)}")

    return intra_distances, inter_distances


# 主函数
def main():
    # 设置输入文件路径
    train_excel_path = r'D:/deephash_original/data/imagenet/train1.xlsx'
    images_folder = r'D:/deephash_original/dataset/imagenet/image_refool/'

    # 处理训练集并计算汉明距离
    intra_distances, inter_distances = process_training_set_and_calculate_distances(
        train_excel_path, images_folder)

    # 绘制汉明距离分布图
    plot_hamming_distributions(intra_distances, inter_distances)


if __name__ == "__main__":
    main()