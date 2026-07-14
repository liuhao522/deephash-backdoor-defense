import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from network import ResNet

# 设置中文字体兼容性（移除英文字体设置）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用更通用的无衬线字体
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 图片和模型相关路径
img_dir = r"D:/deephash_original/dataset/cifar10/"
save_path = r"D:/deephash_original/save/DBDH/CIFAR10/CIFAR10_16bits_0.7999549951348336/"
model_name = 'model.pt'
# 加载模型
model = ResNet(hash_bit=16)
model_state_dict = torch.load(os.path.join(save_path, model_name), map_location=device, weights_only=False)
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


# 提取文件名中的标签
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


# 查找所有干净图片
def find_all_clean_images(train_excel_path, images_folder):
    train_df = pd.read_excel(train_excel_path, header=None)
    clean_images = {}

    print("\nFinding all clean images...")
    for _, row in train_df.iterrows():
        image_filename = row[0]
        machine_label = row[1]
        true_label = extract_labels_from_filename(image_filename)

        if true_label is not None and true_label == machine_label:
            if true_label not in clean_images:
                clean_images[true_label] = []
            clean_images[true_label].append(image_filename)

    if not clean_images:
        print("Warning: No clean images found")
    else:
        total_clean = sum(len(imgs) for imgs in clean_images.values())
        print(f"Found {total_clean} clean images, distributed across {len(clean_images)} different labels")

        # 打印每个类别的图片数量
        for label, imgs in clean_images.items():
            print(f"Label {label}: {len(imgs)} images")

    return clean_images


# 计算所有干净图片的哈希码
def calculate_all_clean_hashes(clean_images, images_folder):
    clean_hashes = {}
    print("\nCalculating hash codes for all clean images...")

    for label, filenames in clean_images.items():
        if label not in clean_hashes:
            clean_hashes[label] = []

        for filename in filenames:
            image_path = os.path.join(images_folder, filename)
            try:
                img = Image.open(image_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    hash_array = model(img_tensor).sign()[0].cpu().numpy()

                binary_hash = np.where(hash_array > 0, 1, 0)
                clean_hashes[label].append(binary_hash)

            except Exception as e:
                print(f"Error calculating hash for image {filename}: {e}")

        print(f"Label {label}: Calculated {len(clean_hashes[label])} hash codes")

    return clean_hashes


# 计算每个类别的基准哈希码（通过多数投票）
def calculate_benchmark_hashes(clean_hashes):
    benchmark_hashes = {}

    print("\nCalculating benchmark hash codes for each class...")
    for label, hash_list in clean_hashes.items():
        if not hash_list:
            continue

        # 将哈希码列表转换为二维数组
        hash_array = np.array(hash_list)

        # 使用多数投票法确定每个比特位的值
        benchmark_hash = np.round(np.mean(hash_array, axis=0)).astype(int)
        benchmark_hashes[label] = benchmark_hash

        print(f"Label {label}: Benchmark hash calculated, shape {benchmark_hash.shape}")

    return benchmark_hashes


# 计算每个类别每个比特位的一致性比例
def calculate_bit_consistency(clean_hashes, benchmark_hashes):
    consistency_matrix = {}

    print("\nCalculating bit-level consistency for each class...")
    for label, hash_list in clean_hashes.items():
        if label not in benchmark_hashes or not hash_list:
            continue

        benchmark = benchmark_hashes[label]
        hash_array = np.array(hash_list)

        # 计算每个比特位与基准哈希码一致的比例
        consistency = np.mean(hash_array == benchmark, axis=0)
        consistency_matrix[label] = consistency

        print(f"Label {label}: Consistency calculation completed, average consistency: {np.mean(consistency):.4f}")

    return consistency_matrix


# 绘制比特一致性热图
def plot_bit_consistency_heatmap(consistency_matrix, benchmark_hashes):
    if not consistency_matrix:
        print("No data available for heatmap")
        return

    # 获取所有标签并排序
    labels = sorted(consistency_matrix.keys())
    num_bits = len(benchmark_hashes[labels[0]])

    print(f"\nPlotting bit consistency heatmap: {len(labels)} classes, {num_bits} bits")

    # 创建一致性矩阵
    data = np.zeros((len(labels), num_bits))
    for i, label in enumerate(labels):
        data[i, :] = consistency_matrix[label]

    # 创建图形 - 使用16:9比例
    plt.figure(figsize=(16, 9))

    # 绘制热图 - 使用square=False使方格水平方向变长
    ax = sns.heatmap(data,
                     cmap="YlGnBu",
                     vmin=0.5,
                     vmax=1.0,
                     square=False,  # 设置为False使方格不是正方形
                     cbar_kws={'label': 'Consistency Ratio'},
                     xticklabels=False,  # 关闭默认的x轴标签
                     yticklabels=True,
                     linewidths=0.5,  # 单元格之间的线宽
                     linecolor='lightgray')  # 单元格之间的线颜色

    # 设置标题和轴标签
    plt.title('Bit-Level Consistency Heatmap of Hash Codes', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('', fontsize=12)  # 清空x轴标签
    plt.ylabel('Class Label', fontsize=12)

    # 调整坐标轴
    ax.set_yticklabels(labels, rotation=0)

    # 设置x轴刻度位置
    x_ticks = np.arange(num_bits) + 0.5  # 将刻度设置在每列的中间
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])  # 清空默认的刻度标签

    # 添加颜色条标签
    cbar = ax.collections[0].colorbar
    cbar.set_label('Consistency Ratio', fontsize=12)

    # ========== 只在热图底部添加每一位的数字标签 (0-15) ==========
    # 移除了上方的数字标签，只保留底部标签
    for bit_idx in range(num_bits):
        # 在热图底部添加文本，位置为每列的中心
        plt.text(bit_idx + 0.5, len(labels) + 0.15, str(bit_idx),  # 使用+0.15让数字更贴合
                 ha='center', va='top', fontsize=10, fontweight='bold', color='black')

    # 添加平均一致性文本
    avg_consistency = np.mean(data)
    plt.figtext(0.15, 0.02, f'Average Consistency: {avg_consistency:.4f}',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig('bit_consistency_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('bit_consistency_heatmap.pdf', bbox_inches='tight')
    print("Bit consistency heatmap saved as 'bit_consistency_heatmap.png' and 'bit_consistency_heatmap.pdf'")

    # 显示图片
    plt.show()

    return avg_consistency


# 主函数
def main():
    # 设置输入文件路径
    train_excel_path = r'D:/deephash_original/data/CIFAR10/train1.xlsx'
    images_folder = r'D:/deephash_original/dataset/cifar10/images_refool/'

    # 查找所有干净图片
    clean_images = find_all_clean_images(train_excel_path, images_folder)

    # 计算所有干净图片的哈希码
    clean_hashes = calculate_all_clean_hashes(clean_images, images_folder)

    # 计算基准哈希码
    benchmark_hashes = calculate_benchmark_hashes(clean_hashes)

    # 计算比特一致性
    consistency_matrix = calculate_bit_consistency(clean_hashes, benchmark_hashes)

    # 绘制比特一致性热图
    avg_consistency = plot_bit_consistency_heatmap(consistency_matrix, benchmark_hashes)

    print(f"\nOverall average bit consistency: {avg_consistency:.4f}")


if __name__ == "__main__":
    main()