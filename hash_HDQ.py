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
import torch.nn.functional as F
import math

# 设置多进程共享策略
torch.multiprocessing.set_sharing_strategy('file_system')


class PoincareBall:
    """Basic Poincaré ball implementation to replace geoopt"""

    def __init__(self, c=1.0):
        self.c = c
        self.eps = 1e-5
        self.radius = 1 / math.sqrt(c)

    def expmap0(self, u):
        """Exponential map at origin"""
        sqrt_c = math.sqrt(self.c)
        u_norm = torch.clamp(torch.norm(u, dim=-1, keepdim=True), min=self.eps)
        gamma_u = torch.tanh(sqrt_c * u_norm) / u_norm
        return gamma_u * u

    def logmap0(self, x):
        """Logarithmic map at origin"""
        sqrt_c = math.sqrt(self.c)
        x_norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=self.eps)
        return torch.atanh(sqrt_c * x_norm) * x / (sqrt_c * x_norm)

    def dist(self, x, y):
        """Poincaré distance"""
        sqrt_c = math.sqrt(self.c)
        x_norm = torch.clamp(torch.norm(x, dim=-1), min=self.eps)
        y_norm = torch.clamp(torch.norm(y, dim=-1), min=self.eps)
        xy_norm = torch.clamp(torch.norm(x - y, dim=-1), min=self.eps)

        num = 2 * xy_norm ** 2
        denom = (1 - self.c * x_norm ** 2) * (1 - self.c * y_norm ** 2)
        return torch.acosh(1 + num / denom) / sqrt_c

    def norm(self, x):
        """Norm in the Poincaré ball"""
        return torch.norm(x, dim=-1)


def get_config():
    """获取配置参数"""
    base_data_path = './data/imagenet/'
    base_save_path = './save/HDQ/imagenet/'  # 修改保存路径

    os.makedirs(base_save_path, exist_ok=True)

    config = {
        "gamma": 0.1,  # 双曲量化损失权重
        "zeta": 0.5,  # 双曲相似性损失权重
        "c": 1.0,  # 双曲空间曲率
        "optimizer": {
            "type": optim.RAdam,  # 使用Rectified Adam优化器
            "epoch_lr_decrease": 15,  # 学习率衰减周期
            "optim_params": {
                "lr": 5e-5,  # HDQ推荐的学习率
                "weight_decay": 10 ** -5
            }
        },
        "info": "[HDQ]",  # 修改信息标识
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": "EfficientNetV2",
        "n_class": 100,
        "dataset": "imagenet",
        "epoch": 50,  # HDQ需要更少的epoch
        "test_map": 5,
        "save_path": base_save_path,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "bit_list": [16, 32, 48, 64, 128],
        "topK": -1,
        "data_path": './dataset/imagenet/',
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


class HyperbolicProjection(torch.nn.Module):
    """双曲空间投影层"""

    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x):
        return self.manifold.expmap0(x)


class EfficientNetV2_HDQ(torch.nn.Module):
    """HDQ专用的EfficientNetV2网络结构"""

    def __init__(self, bit, c=1.0):
        super(EfficientNetV2_HDQ, self).__init__()
        self.bit = bit
        self.manifold = PoincareBall(c=c)  # 庞加莱球模型

        # 加载预训练的EfficientNetV2
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        # 替换最后的全连接层
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, bit),
            HyperbolicProjection(self.manifold))  # 投影到双曲空间

        # 双曲空间特定的初始化
        torch.nn.init.xavier_uniform_(self.efficientnet._fc[0].weight)
        torch.nn.init.xavier_uniform_(self.efficientnet._fc[2].weight)
        self.efficientnet._fc[0].bias.data.zero_()
        self.efficientnet._fc[2].bias.data.zero_()

    def forward(self, x):
        x = self.efficientnet(x)
        return x


class HDQLoss(torch.nn.Module):
    """HDQ损失函数"""

    def __init__(self, config, bit):
        super(HDQLoss, self).__init__()
        self.config = config
        self.bit = bit
        self.manifold = PoincareBall(c=config["c"])
        self.centers = self.init_centers(config["n_class"], bit).to(config["device"])

    def init_centers(self, n_class, bit):
        """在双曲空间中初始化中心点"""
        # 在切空间生成随机点
        centers = torch.randn(n_class, bit) * 0.01
        # 投影到双曲空间
        centers = self.manifold.expmap0(centers)
        return centers

    def hyperbolic_distance(self, u, v):
        """计算双曲距离"""
        return self.manifold.dist(u, v)

    def forward(self, u, y, ind):
        # 计算目标中心 (基于标签)
        target_centers = self.manifold.expmap0(
            torch.matmul(y.float(), self.manifold.logmap0(self.centers)))

        # 双曲相似性损失
        sim_loss = torch.mean(self.hyperbolic_distance(u, target_centers) ** 2)

        # 双曲量化损失 (到边界距离)
        norm_u = self.manifold.norm(u)
        quant_loss = torch.mean((norm_u - 1.0) ** 2)

        # 总损失
        total_loss = self.config["zeta"] * sim_loss + self.config["gamma"] * quant_loss

        return total_loss


def validate_hdq(config, best_mAP, test_loader, database_loader, net, bit, epoch, num_dataset):
    """HDQ专用的验证函数"""
    net.eval()
    device = config["device"]

    # 生成数据库哈希码 (在双曲空间中)
    database_hash = torch.zeros(num_dataset, bit).to(device)
    database_labels = torch.zeros(num_dataset, config["n_class"]).to(device)

    with torch.no_grad():
        for images, labels, indices in database_loader:
            images = images.to(device)
            labels = labels.to(device)  # 确保标签也在相同设备上
            outputs = net(images)
            database_hash[indices, :] = outputs
            database_labels[indices, :] = labels.float()

    # 将双曲点映射到切空间用于计算汉明距离
    database_hash = net.manifold.logmap0(database_hash).sign()

    # 生成测试集哈希码
    test_hash = []
    test_labels = []
    for images, labels, _ in test_loader:
        images = images.to(device)
        labels = labels.to(device)  # 确保标签也在相同设备上
        outputs = net(images)
        test_hash.append(net.manifold.logmap0(outputs).sign().cpu())
        test_labels.append(labels.cpu())

    test_hash = torch.cat(test_hash, 0)
    test_labels = torch.cat(test_labels, 0)

    # 计算mAP
    mAP = compute_mAP(test_hash, database_hash.cpu(), test_labels, database_labels.cpu())

    print(f"测试mAP@{bit}bit: {mAP:.4f}")
    if mAP > best_mAP:
        best_mAP = mAP
    return best_mAP


def train_val(config, bit):
    """HDQ训练和验证函数"""
    device = config["device"]

    # 获取数据加载器
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train

    # 初始化网络
    net = EfficientNetV2_HDQ(bit, c=config["c"]).to(device)

    # 初始化优化器
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    # 初始化损失函数
    criterion = HDQLoss(config, bit)

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
                label = label.to(device)  # 确保标签也在相同设备上

                optimizer.zero_grad()
                u = net(image)

                loss = criterion(u, label, ind)
                train_loss += loss.item()

                loss.backward()
                # 双曲空间需要特殊的梯度裁剪
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
                optimizer.step()

            train_loss = train_loss / len(train_loader)
            print("\b\b\b\b\b\b\b 损失:%.3f" % (train_loss))

            # 定期测试模型性能
            if (epoch + 1) % config["test_map"] == 0:
                current_mAP = validate_hdq(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
                if current_mAP > Best_mAP:
                    Best_mAP = current_mAP
                    torch.save(net.state_dict(), os.path.join(config["save_path"], f"best_model_{bit}bit.pth"))

    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        torch.save(net.state_dict(), os.path.join(config["save_path"], f"emergency_save_{bit}bit.pth"))
        raise e

    results[bit] = Best_mAP
    return results


if __name__ == "__main__":
    config = get_config()
    print("配置参数:", config)

    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU训练，速度会很慢!")

    final_results = {}
    for bit in config["bit_list"]:
        print(f"\n开始训练 {bit}-bit 模型...")
        results = train_val(config, bit)
        final_results.update(results)
        print(f"\n{bit}-bit 模型结果: mAP = {results[bit]:.4f}")

    print("\n最终结果:")
    for bit, mAP in final_results.items():
        print(f"{bit}-bit 模型 mAP: {mAP:.4f}")