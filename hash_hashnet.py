# -*- coding:utf-8 -*-
# author:zhangning
from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
from efficientnet_pytorch import EfficientNet

# 设置多进程共享策略
torch.multiprocessing.set_sharing_strategy('file_system')


def get_config():
    """获取配置参数"""
    # 基础路径设置（修改为您的实际路径）
    base_data_path = './data/imagenet/'
    base_save_path = './save/HashNet/imagenet/'

    # 确保保存路径存在
    os.makedirs(base_save_path, exist_ok=True)

    config = {
        "alpha": 0.1,  # 量化损失权重
        "p": 2,  # 量化范数类型 (1或2)
        "optimizer": {
            "type": optim.RMSprop,
            "epoch_lr_decrease": 30,  # 学习率衰减周期(原50改为30)
            "optim_params": {
                "lr": 1e-4,  # 学习率(从1e-5提高到1e-4)
                "weight_decay": 10 ** -5
            }
        },
        "info": "[HashNet]",
        "resize_size": 256,  # 图像调整大小
        "crop_size": 224,  # 图像裁剪尺寸
        "batch_size": 16,  # 批大小(根据GPU内存调整)
        "net": "EfficientNetV2",  # 使用的网络
        "n_class": 100,  # 类别数
        "dataset": "imagenet",  # 数据集名称
        "epoch": 50,  # 训练轮数
        "test_map": 5,  # 每多少轮测试一次
        "save_path": base_save_path,  # 模型保存路径
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),  # 自动检测设备
        "bit_list": [16, 32, 48, 64, 128],  # 哈希位宽列表
        "topK": -1,
        "data_path": './dataset/imagenet/',  # 数据集路径
        "data": {
            "train_set": {
                "list_path": os.path.join(base_data_path, 'train.txt'),
                "batch_size": 16
            },
            "database": {
                "list_path": os.path.join(base_data_path, 'database.txt'),
                "batch_size": 16
            },
            "test": {
                "list_path": os.path.join(base_data_path, 'test.txt'),
                "batch_size": 16
            }
        }
    }
    config = config_dataset(config)
    return config


class EfficientNetV2_Hash(torch.nn.Module):
    """改进的EfficientNetV2哈希网络"""

    def __init__(self, bit):
        super(EfficientNetV2_Hash, self).__init__()
        # 加载预训练的EfficientNetV2
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        # 替换最后的全连接层为哈希层
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = torch.nn.Linear(in_features, bit)

        # 改进的初始化方式
        torch.nn.init.xavier_uniform_(self.efficientnet._fc.weight)
        self.efficientnet._fc.bias.data.zero_()

    def forward(self, x):
        x = self.efficientnet(x)
        return torch.tanh(x)  # 使用tanh将输出限制在[-1,1]范围内


class DPSHLoss(torch.nn.Module):
    """改进的深度成对监督哈希损失函数"""

    def __init__(self, config, bit):
        super(DPSHLoss, self).__init__()
        # 初始化存储矩阵
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])
        self.config = config

    def forward(self, u, y, ind, config):
        # 限制哈希码范围
        u = u.clamp(min=-1, max=1)

        # 更新存储矩阵
        self.U[ind, :] = u.detach()  # 使用detach()避免梯度计算
        self.Y[ind, :] = y.float()

        # 计算相似度矩阵
        s = (y @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5

        # 改进的似然损失计算
        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product
        likelihood_loss = likelihood_loss.mean()

        # 量化损失计算
        if config["p"] == 1:
            quantization_loss = config["alpha"] * u.mean(dim=1).abs().mean()
        else:
            quantization_loss = config["alpha"] * u.mean(dim=1).pow(2).mean()

        return likelihood_loss + quantization_loss


def train_val(config, bit):
    """训练和验证函数"""
    device = config["device"]

    # 打印GPU内存信息
    if torch.cuda.is_available():
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")

    # 获取数据加载器
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train

    # 初始化网络
    if config["net"] == "EfficientNetV2":
        net = EfficientNetV2_Hash(bit).to(device)
    else:
        net = config["net"](bit).to(device)

    # 初始化优化器
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    # 初始化损失函数
    criterion = DPSHLoss(config, bit)

    Best_mAP = 0
    results = {}

    try:
        for epoch in range(config["epoch"]):
            # 动态调整学习率
            lr = config["optimizer"]["optim_params"]["lr"] * (
                        0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print("%s[%2d/%2d][%s] bit:%d, dataset:%s, 训练中..." % (
                config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

            net.train()
            train_loss = 0

            for image, label, ind in train_loader:
                image = image.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                u = net(image)

                loss = criterion(u, label.float(), ind, config)
                train_loss += loss.item()

                loss.backward()
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss = train_loss / len(train_loader)
            print("\b\b\b\b\b\b\b 损失:%.3f" % (train_loss))

            # 定期测试模型性能
            if (epoch + 1) % config["test_map"] == 0:
                current_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
                if current_mAP > Best_mAP:
                    Best_mAP = current_mAP
                    # 保存最佳模型
                    torch.save(net.state_dict(), os.path.join(config["save_path"], f"best_model_{bit}bit.pth"))

    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        # 保存当前模型以防崩溃
        torch.save(net.state_dict(), os.path.join(config["save_path"], f"emergency_save_{bit}bit.pth"))
        raise e

    results[bit] = Best_mAP
    return results


if __name__ == "__main__":
    # 获取配置
    config = get_config()
    print("配置参数:", config)

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU训练，速度会很慢!")

    final_results = {}
    for bit in config["bit_list"]:
        print(f"\n开始训练 {bit}-bit 模型...")
        results = train_val(config, bit)
        final_results.update(results)
        print(f"\n{bit}-bit 模型结果: mAP = {results[bit]:.4f}")

    # 打印最终结果
    print("\n最终结果:")
    for bit, mAP in final_results.items():
        print(f"{bit}-bit 模型 mAP: {mAP:.4f}")