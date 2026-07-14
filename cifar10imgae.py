# -*- coding:utf-8 -*- 
# author:zhangning
import os
import pickle
import numpy as np
from PIL import Image
import cv2


# 加载CIFAR-10数据集
def load_cifar10_data(batch_file):
    with open(batch_file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


# 将CIFAR-10数据集转换为图像并保存
def convert_cifar10_to_images(data_dir, save_dir):
    # 创建保存图像的文件夹
    os.makedirs(save_dir, exist_ok=True)

    # 加载训练集和测试集数据
    for batch in range(1, 7):
        batch_file = os.path.join(data_dir, f"data_batch_{batch}")
        batch_data = load_cifar10_data(batch_file)

        images = batch_data[b"data"]
        labels = batch_data[b"labels"]

        num_images = len(images)
        for i in range(num_images):
            image = np.reshape(images[i], (3, 32, 32))  # CIFAR-10图像的维度为(3, 32, 32)
            image = np.transpose(image, (1, 2, 0))  # 转置为(32, 32, 3)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            label = labels[i]
            image_name = f"{i + (batch - 1) * 10000}-label-{label}.png"  # 图像文件名格式为：序号_标签.png
            save_path = os.path.join(save_dir, image_name)

            image = Image.fromarray(image.astype('uint8'))
            image.save(save_path, quality=100)


# 指定数据集和保存图像的文件夹路径
data_dir = f"D:/deephash_original/data\cifar\cifar-10-batches-py"
save_dir = "./dataset/cifar10/images"

# 调用函数将CIFAR-10数据集转换为图像并保存
convert_cifar10_to_images(data_dir, save_dir)
