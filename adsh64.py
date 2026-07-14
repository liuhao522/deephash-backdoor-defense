from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

torch.multiprocessing.set_sharing_strategy('file_system')

# ADSH (Asymmetric Deep Supervised Hashing)
# 论文《Asymmetric Deep Supervised Hashing for Image Retrieval》
# 核心思想：查询样本和数据库样本使用不同的哈希函数，减少量化误差

def get_config():
    config = {
        "gamma": 1.0,    # 增大相似性损失的权重
        "eta": 0.01,     # 减小量化损失的权重
        "optimizer": {
            "type": optim.Adam,  # 改用Adam优化器
            "optim_params": {
                "lr": 1e-4,      # 增大学习率
                "weight_decay": 1e-5,
                "betas": (0.9, 0.999)
            }
        },
        "info": "[ADSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": ResNet,
        "dataset": "GTSRB",
        "epoch": 50,
        "test_map": 5,  # 更频繁的验证
        "device": torch.device("cuda:0"),
        "bit_list": [64],  # 尝试不同比特数
        "topK": 5000,  # 设置topK值
    }
    config = config_dataset(config)
    return config

class ADSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(ADSHLoss, self).__init__()
        self.bit = bit
        self.gamma = config["gamma"]
        self.eta = config["eta"]
        self.n_class = config["n_class"]
        self.device = config["device"]

        # 分类层（共享）
        self.classifier = torch.nn.Linear(bit, self.n_class).to(self.device)
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)

    def forward(self, u, y, ind, config):
        # 添加数值稳定处理
        u = torch.clamp(u, -10, 10)  # 限制哈希码范围防止数值溢出

        # 分类损失
        cls_output = self.classifier(u)
        if config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}:
            cls_loss = self.criterion_cls(cls_output, y.argmax(axis=1))
        else:
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                cls_output, y.float(), reduction='mean')

        # 量化损失（使用平滑L1损失替代MSE）
        quant_loss = torch.mean(torch.abs(torch.abs(u) - 1))

        # 相似性保持损失（添加温度系数和数值稳定处理）
        S = (y @ y.t() > 0).float()  # 相似性矩阵
        theta = u @ u.t() / self.bit  # 归一化内积
        # 数值稳定处理
        theta = torch.clamp(theta, -10, 10)
        sim_loss = -torch.mean(S * theta - torch.log(1 + torch.exp(theta)))

        # 总损失（添加损失项权重平衡）
        total_loss = 0.5 * cls_loss + self.gamma * sim_loss + self.eta * quant_loss
        return total_loss, cls_loss, sim_loss, quant_loss


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    criterion = ADSHLoss(config, bit)

    Best_mAP = 0
    writer = SummaryWriter()

    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, 训练中..." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        train_loss = 0
        total_cls_loss = 0
        total_sim_loss = 0
        total_quant_loss = 0
        total_grad_norm = 0

        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)  # 生成哈希码
            loss, cls_loss, sim_loss, quant_loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_sim_loss += sim_loss.item()
            total_quant_loss += quant_loss.item()

            loss.backward()

            # 计算梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            total_grad_norm += grad_norm.item()

            optimizer.step()

        train_loss = train_loss / len(train_loader)
        total_cls_loss = total_cls_loss / len(train_loader)
        total_sim_loss = total_sim_loss / len(train_loader)
        total_quant_loss = total_quant_loss / len(train_loader)
        total_grad_norm = total_grad_norm / len(train_loader)

        print("\b\b\b\b\b\b\b 损失:%.3f" % (train_loss))

        # 记录训练监控指标
        writer.add_scalar(f'Train/Loss_{bit}', train_loss, epoch)
        writer.add_scalar(f'Train/Classification_Loss_{bit}', total_cls_loss, epoch)
        writer.add_scalar(f'Train/Similarity_Loss_{bit}', total_sim_loss, epoch)
        writer.add_scalar(f'Train/Quantization_Loss_{bit}', total_quant_loss, epoch)
        writer.add_scalar(f'Train/Gradient_Norm_{bit}', total_grad_norm, epoch)

        # 添加学习率衰减
        if (epoch + 1) % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/alexnet/ADSH_{config['dataset']}_{bit}.json"
        train_val(config, bit)