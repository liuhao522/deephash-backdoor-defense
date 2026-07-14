from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy('file_system')


# TransHash (Transformer-based Deep Hashing)
# 使用Transformer架构进行深度哈希学习

def get_config():
    config = {
        "alpha": 0.1,  # 量化损失的权重
        "beta": 0.1,  # 分类损失的权重
        "gamma": 0.05,  # 相似性损失的权重
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-4, "weight_decay": 10 ** -5}},
        "info": "[TransHash]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": TransHashNet,  # 使用Transformer网络
        "dataset": "GTSRB",
        "epoch": 50,
        "test_map": 10,
        "device": torch.device("cuda:0"),
        "bit_list": [32],
        # Transformer特定参数
        "num_heads": 8,
        "num_layers": 6,
        "d_model": 512,
        "dropout": 0.1,
    }
    config = config_dataset(config)
    return config


class TransHashNet(nn.Module):
    def __init__(self, bit, num_heads=8, num_layers=6, d_model=512, dropout=0.1):
        super(TransHashNet, self).__init__()
        self.bit = bit
        self.d_model = d_model

        # 特征提取backbone (可以是ResNet或其他)
        self.backbone = ResNet(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # 哈希码生成层
        self.hash_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, bit),
            nn.Tanh()  # 输出范围在[-1, 1]
        )

    def forward(self, x):
        # 特征提取
        features = self.backbone(x)  # [batch_size, d_model]

        # 添加序列维度用于Transformer
        features = features.unsqueeze(1)  # [batch_size, 1, d_model]

        # 位置编码
        features = self.pos_encoding(features)

        # Transformer编码
        encoded = self.transformer(features)  # [batch_size, 1, d_model]

        # 移除序列维度
        encoded = encoded.squeeze(1)  # [batch_size, d_model]

        # 生成哈希码
        hash_code = self.hash_layer(encoded)

        return hash_code


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TransHashLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(TransHashLoss, self).__init__()
        self.bit = bit
        self.alpha = config["alpha"]  # 量化损失权重
        self.beta = config["beta"]  # 分类损失权重
        self.gamma = config["gamma"]  # 相似性损失权重
        self.n_class = config["n_class"]
        self.device = config["device"]

        # 分类层
        self.classifier = torch.nn.Linear(bit, self.n_class).to(self.device)
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)

    def forward(self, u, y, ind, config):
        batch_size = u.size(0)

        # 1. 分类损失
        cls_output = self.classifier(u)
        if config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}:
            cls_loss = self.criterion_cls(cls_output, y.argmax(axis=1))
        else:
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(cls_output, y.float())

        # 2. 量化损失 - 强制哈希码接近-1或1
        quant_loss = torch.mean((torch.abs(u) - 1) ** 2)

        # 3. 相似性损失 - 基于标签相似性的哈希码相似性
        # 计算标签相似性矩阵
        if config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}:
            # 单标签数据集
            y_labels = y.argmax(axis=1)
            S = (y_labels.unsqueeze(0) == y_labels.unsqueeze(1)).float()
        else:
            # 多标签数据集
            S = (torch.mm(y, y.t()) > 0).float()

        # 计算哈希码相似性
        inner_product = torch.mm(u, u.t()) / self.bit
        similarity_loss = torch.mean((S * inner_product - S) ** 2)

        # 总损失
        total_loss = (self.beta * cls_loss +
                      self.alpha * quant_loss +
                      self.gamma * similarity_loss)

        return total_loss


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train

    # 创建TransHash网络
    net = config["net"](
        bit=bit,
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        dropout=config["dropout"]
    ).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    criterion = TransHashLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, 训练中..." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)  # 生成哈希码

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()

        # 更新学习率
        scheduler.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b 损失:%.3f, LR:%.6f" % (train_loss, scheduler.get_last_lr()[0]))

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/transhash/TransHash_{config['dataset']}_{bit}.json"
        train_val(config, bit)
