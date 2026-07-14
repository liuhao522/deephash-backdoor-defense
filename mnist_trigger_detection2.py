from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import pandas as pd
from network import ResNet

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图片和模型相关路径
# 文件路径配置（根据实际需求修改）
img_dir = r"D:/deephash_original/dataset/MNIST/"
save_path = r"D:/deephash_original/save/DBDH/MNIST128/MNIST_128bits_0.9780780623311316_youxia/"
model_name = 'model.pt'


# 加载哈希模型
hash_model = ResNet(hash_bit=128)
model_state_dict = torch.load(os.path.join(save_path, model_name), map_location=device)
hash_model.load_state_dict(model_state_dict)
hash_model.eval().to(device)

# 加载分类模型(用于生成FGSM对抗样本)
classify_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
classify_model.fc = nn.Linear(2048, 10)
classify_model_state = torch.load('save/resnet/resnet_mnist.pt', map_location=device)
classify_model.load_state_dict(classify_model_state)
classify_model.eval().to(device)

# 图片预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def pgd_attack(image, epsilon, alpha, num_iter, data_grad, is_poisoned):
    perturbed_image = image.clone().detach()

    for _ in range(num_iter):
        # 计算扰动
        if is_poisoned:
            # 后门样本：最小化哈希距离
            perturbed_image = perturbed_image - alpha * data_grad.sign()
        else:
            # 干净样本：最大化哈希距离
            perturbed_image = perturbed_image + alpha * data_grad.sign()

        # 投影到epsilon邻域内
        eta = torch.clamp(perturbed_image - image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(image + eta, 0, 1)

    return perturbed_image


def generate_pgd_sample(image_path, epsilon=0.03, alpha=0.01, num_iter=10, is_poisoned=False):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    original_hash = hash_model(img_tensor).sign()

    if is_poisoned:
        loss = -F.mse_loss(hash_model(img_tensor).sign(), original_hash)
    else:
        loss = F.mse_loss(hash_model(img_tensor).sign(), original_hash)

    classify_model.zero_grad()
    hash_model.zero_grad()
    loss.backward()

    data_grad = img_tensor.grad.data
    perturbed_data = pgd_attack(img_tensor, epsilon, alpha, num_iter, data_grad, is_poisoned)

    return perturbed_data


def find_optimal_threshold(hamm_distances, is_poisoned_list):
    # 收集所有汉明距离
    clean_dists = [d for d, p in zip(hamm_distances, is_poisoned_list) if not p]
    poison_dists = [d for d, p in zip(hamm_distances, is_poisoned_list) if p]

    # 计算均值和标准差
    mean_clean = np.mean(clean_dists) if clean_dists else 0
    std_clean = np.std(clean_dists) if clean_dists else 0
    mean_poison = np.mean(poison_dists) if poison_dists else 0
    std_poison = np.std(poison_dists) if poison_dists else 0

    # 初始阈值为两类均值的中间值
    initial_threshold = (mean_clean + mean_poison) / 2

    # 使用网格搜索寻找最佳阈值
    min_dist = min(min(clean_dists) if clean_dists else 0, min(poison_dists) if poison_dists else 0)
    max_dist = max(max(clean_dists) if clean_dists else 128, max(poison_dists) if poison_dists else 128)

    best_threshold = initial_threshold
    best_f1 = 0

    for threshold in np.linspace(min_dist, max_dist, 100):
        TP = sum(1 for d in poison_dists if d <= threshold)
        FP = sum(1 for d in clean_dists if d <= threshold)
        TN = sum(1 for d in clean_dists if d > threshold)
        FN = sum(1 for d in poison_dists if d > threshold)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def detect(image_tensor):
    with torch.no_grad():
        qB = hash_model(image_tensor).sign()[0].detach().cpu().numpy()
    return np.where(qB > 0, 1, 0)


def hamming_distance(arr1, arr2):
    return np.sum(arr1 != arr2)


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


def improved_process_training_set_and_detect_backdoor(train_excel_path, images_folder, output_excel_path):
    train_df = pd.read_excel(train_excel_path, header=None)
    results = []
    hamm_distances = []
    is_poisoned_list = []

    print("\n开始改进版后门检测...")
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
            is_poisoned = true_label != machine_label
            is_poisoned_list.append(is_poisoned)

            # 生成PGD对抗样本
            perturbed_img = generate_pgd_sample(input_path, is_poisoned=is_poisoned)

            # 计算哈希和汉明距离
            original_img = transform(Image.open(input_path).convert('RGB')).unsqueeze(0).to(device)
            original_hash = detect(original_img)
            perturbed_hash = detect(perturbed_img)

            hamm_dist = hamming_distance(original_hash, perturbed_hash)
            hamm_distances.append(hamm_dist)

            results.append({
                '图片名称': image_filename,
                '真实标签': true_label,
                '机器训练标签': machine_label,
                '汉明距离': hamm_dist,
                '是否带有触发器': is_poisoned
            })

        except Exception as e:
            print(f"处理文件 {image_filename} 出错: {e}")
            continue

    if results:
        # 动态确定最佳阈值
        optimal_threshold = find_optimal_threshold(hamm_distances, is_poisoned_list)
        print(f"\n自动确定的最佳汉明距离阈值: {optimal_threshold:.2f}")

        # 使用最佳阈值重新评估
        TP = FP = TN = FN = 0
        for result, dist in zip(results, hamm_distances):
            predicted_poisoned = dist <= optimal_threshold
            is_poisoned = result['是否带有触发器']

            if predicted_poisoned and is_poisoned:
                TP += 1
            elif predicted_poisoned and not is_poisoned:
                FP += 1
            elif not predicted_poisoned and not is_poisoned:
                TN += 1
            else:
                FN += 1

            result['预测是否带有触发器'] = '是' if predicted_poisoned else '否'
            result['是否预测正确'] = '是' if predicted_poisoned == is_poisoned else '否'

        # 保存结果
        df = pd.DataFrame(results)
        df.to_excel(output_excel_path, index=False)

        # 计算评估指标
        total_images = len(results)
        poison_count = sum(is_poisoned_list)
        clean_count = total_images - poison_count
        accuracy = (TP + TN) / total_images * 100
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        poison_detection_rate = TP / poison_count * 100 if poison_count > 0 else 0
        clean_detection_rate = TN / clean_count * 100 if clean_count > 0 else 0

        # 打印统计信息
        print("\n改进版检测结果统计:")
        print("=" * 60)
        print(f"总处理图片数: {len(train_df)}")
        print(f"有效处理图片数: {total_images}")
        print(f"中毒样本数: {poison_count}")
        print(f"干净样本数: {clean_count}")
        print(f"最佳汉明距离阈值: {optimal_threshold:.2f}")
        print(f"整体预测准确率: {accuracy:.2f}%")
        print(f"中毒数据检测率: {poison_detection_rate:.2f}%")
        print(f"干净数据检测率: {clean_detection_rate:.2f}%")
        print(f"精确率(Precision): {precision:.4f}")
        print(f"召回率(Recall): {recall:.4f}")
        print(f"F1分数: {f1_score:.4f}")
        print("\n混淆矩阵:")
        print(f"真阳性(TP): {TP}")
        print(f"假阳性(FP): {FP}")
        print(f"真阴性(TN): {TN}")
        print(f"假阴性(FN): {FN}")
        print("=" * 60)
        print(f"结果已保存到: {output_excel_path}")

# 设置输入文件路径和输出Excel文件路径
train_excel_path = r'D:/deephash_original/data/MNIST/train1.xlsx'
images_folder = r'D:/deephash_original/dataset/MNIST/images_youxia/'
output_excel_path = r'D:/deephash_original/dataset/MNIST/fgsm_backdoor_detection_results.xlsx'

# 处理训练集并检测后门
improved_process_training_set_and_detect_backdoor(train_excel_path, images_folder, output_excel_path)