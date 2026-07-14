from utils.tools import *
from network import *
from scipy.linalg import hadamard

import os
import torch
import torch.optim as optim
import time
import numpy as np
import random

torch.multiprocessing.set_sharing_strategy('file_system')


# DCSQ (Deep Central Similarity Quantization)
# 结合DBDH的平衡离散哈希和CSQ的中心相似性量化

def get_config():
    config = {
        "alpha": 0.1,  # 量化损失权重
        "lambda": 0.0001,  # 中心相似性损失权重 (来自CSQ)
        "beta": 0.01,  # 平衡约束权重 (来自DBDH)
        "p": 2,  # 量化范数类型 (1或2)

        "optimizer": {
            "type": optim.RMSprop,
            "epoch_lr_decrease": 50,
            "optim_params": {
                "lr": 1e-5,
                "weight_decay": 10 ** -5
            }
        },

        "info": "[DCSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": ResNet,
        "n_class": 10,
        "dataset": "GTSRB",
        "epoch": 50,
        "test_map": 5,
        "save_path": "save/DCSQ/MNIST128",
        "device": torch.device("cuda:0"),
        "bit_list": [64],
    }
    config = config_dataset(config)
    return config


class DCSQLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DCSQLoss, self).__init__()
        self.bit = bit
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}

        # 来自CSQ的哈希中心生成
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])

        # 来自DBDH的存储变量
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

        self.criterion = torch.nn.BCELoss().to(config["device"])

    def forward(self, u, y, ind, config):
        # 存储当前batch的特征和标签 (来自DBDH)
        u = u.tanh()  # 使用tanh激活确保输出在[-1,1]之间
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        # 1. 中心相似性损失 (来自CSQ)
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        # 2. 相似性保持损失 (改进自DBDH)
        s = (y @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5

        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product
        likelihood_loss = likelihood_loss.mean()

        # 3. 量化损失 (结合两者优点)
        if config["p"] == 1:
            quant_loss = config["alpha"] * u.mean(dim=1).abs().mean()  # L1量化
        else:
            quant_loss = config["alpha"] * (u.abs() - 1).pow(2).mean()  # L2量化 (改进自CSQ)

        # 4. 平衡约束 (来自DBDH)
        balance_loss = config["beta"] * (u.mean(dim=0).pow(2).mean())

        # 修正后的总损失计算，确保正确的括号和行连接
        total_loss = (config["lambda"] * center_loss +
                      likelihood_loss +
                      quant_loss +
                      balance_loss)

        return total_loss

    # 来自CSQ的哈希中心生成方法
    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # 来自CSQ的Hadamard矩阵生成方法
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones

                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    criterion = DCSQLoss(config, bit)
    Best_mAP = 0

    for epoch in range(config["epoch"]):
        # 学习率衰减 (来自DBDH)
        lr = config["optimizer"]["optim_params"]["lr"] * (0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
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
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)

            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mAP': Best_mAP,
            }, os.path.join(config["save_path"], f"model_{bit}bit_{epoch}.pth"))

    # 保存最终模型
    torch.save({
        'epoch': config["epoch"],
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_mAP': Best_mAP,
    }, os.path.join(config["save_path"], "model_final.pth"))


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)