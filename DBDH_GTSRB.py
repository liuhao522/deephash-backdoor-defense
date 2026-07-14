from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# DBDH(Neurocomputing2020)
# 论文 [Deep balanced discrete hashing for image retrieval](https://www.sciencedirect.com/science/article/abs/pii/S0925231220306032)
# [DBDH] epoch:150, bit:48, dataset:cifar10-1, MAP:0.792, Best MAP: 0.793
# [DBDH] epoch:80, bit:48, dataset:nuswide_21, MAP:0.833, Best MAP: 0.834
# [DBDH] epoch:150 bit:32 dataset:CIFAR10 MAP:0.8685966591429419 Best MAP: 0.8764188350089713
# [DBDH] epoch:150 bit:16 dataset:CIFAR10 MAP:0.8703498494713863 Best MAP: 0.8724828981090088
# [DBDH] epoch:150 bit:64 dataset:CIFAR10 MAP:0.8688282014152111 Best MAP: 0.8828527489426025

def get_config():
    config = {
        "alpha": 0.1,  # 量化损失的权重
        "p": 2,  # 使用L2正则化
        "optimizer": {
            "type": optim.RMSprop,
            "epoch_lr_decrease": 50,  # 每50个epoch降低学习率
            "optim_params": {
                "lr": 1e-5,  # 学习率
                "weight_decay": 10 ** -5  # 权重衰减
            }
        },
        "info": "[DBDH]",  # 模型信息
        "resize_size": 256,  # 图像调整大小
        "crop_size": 224,  # 图像裁剪大小
        "batch_size": 16,  # 批量大小
        "net": ResNet,  # 使用ResNet网络
        "n_class": 43,  # GTSRB有43个类别
        "dataset": "GTSRB",  # 使用GTSRB数据集
        "epoch": 50,  # 训练epoch数
        "test_map": 5,  # 每5个epoch测试一次
        "save_path": "save/DBDH/GTSRB",  # 模型保存路径
        "device": torch.device("cuda:0"),  # 使用GPU
        "bit_list": [128],  # 哈希码长度
    }

    # 配置数据集相关参数
    config = config_dataset(config)

    # 打印关键路径信息用于调试
    print("\n配置验证:")
    print(f"数据集路径: {config['data_path']}")
    print(f"训练列表路径: {config['data']['train_set']['list_path']}")
    print(f"测试列表路径: {config['data']['test']['list_path']}")
    print(f"数据库列表路径: {config['data']['database']['list_path']}")

    # 检查关键路径是否存在
    required_paths = [
        config['data_path'],
        config['data']['train_set']['list_path'],
        config['data']['test']['list_path'],
        config['data']['database']['list_path']
    ]

    for path in required_paths:
        if not os.path.exists(path):
            print(f"警告: 路径不存在 - {path}")

    return config


class DPSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DPSHLoss, self).__init__()
        # 初始化存储哈希码和标签的矩阵
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        # 限制哈希码在[-1,1]范围内
        u = u.clamp(min=-1, max=1)

        # 更新存储的哈希码和标签
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        # 计算相似度矩阵
        s = (y @ self.Y.t() > 0).float()

        # 计算内积
        inner_product = u @ self.U.t() * 0.5

        # 计算似然损失
        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product
        likelihood_loss = likelihood_loss.mean()

        # 计算量化损失
        if config["p"] == 1:
            quantization_loss = config["alpha"] * u.mean(dim=1).abs().mean()
        else:
            quantization_loss = config["alpha"] * u.mean(dim=1).pow(2).mean()

        # 总损失
        return likelihood_loss + quantization_loss


def train_val(config, bit):
    device = config["device"]

    # 获取数据加载器
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train

    # 初始化网络
    net = config["net"](bit).to(device)

    # 初始化优化器
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    # 初始化损失函数
    criterion = DPSHLoss(config, bit)

    Best_mAP = 0  # 记录最佳mAP

    for epoch in range(config["epoch"]):
        # 调整学习率
        lr = config["optimizer"]["optim_params"]["lr"] * (0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] 位数:%d, 数据集:%s, 训练中..." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        train_loss = 0

        # 训练过程
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b 损失:%.3f" % (train_loss))

        # 定期测试
        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)


if __name__ == "__main__":
    config = get_config()
    print("\n配置信息:")
    print(config)

    for bit in config["bit_list"]:
        print(f"\n开始训练 {bit} 位哈希码...")
        train_val(config, bit)