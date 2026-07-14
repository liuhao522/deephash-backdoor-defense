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


# ARcHash (Adaptive Robust Cross-modal Hashing, adapted for image-only mode)
# Modified to support single-modality (image) when text data is unavailable

def get_config():
    config = {
        "alpha": 0.1,  # 量化损失权重
        "beta": 0.1,  # 分类损失权重
        "eta": 0.01,  # 自适应鲁棒性权重
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-4, "weight_decay": 10 ** -5}},
        "info": "[ARcHash-ImageOnly]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": ARcHashNet,  # 使用ARcHash网络
        "dataset": "GTSRB",
        "epoch": 50,
        "test_map": 10,
        "device": torch.device("cuda:0"),
        "bit_list": [64],
        "d_model": 512,
        "dropout": 0.1,
        "topK": -1,
        "n_class": 100,
        "data_path": "./dataset/MNIST/",
        "data": {
            "train_set": {"list_path": "./data/imagenet/origin/train.txt", "batch_size": 16},
            "database": {"list_path": "./data/imagenet/database.txt", "batch_size": 16},
            "test": {"list_path": "./data/imagenet/test.txt", "batch_size": 16}
        }
    }
    config = config_dataset(config)
    return config


class ARcHashNet(nn.Module):
    def __init__(self, bit, d_model=512, dropout=0.1):
        super(ARcHashNet, self).__init__()
        self.bit = bit
        self.d_model = d_model

        # 图像模态特征提取器
        self.image_backbone = ResNet(d_model)

        # 哈希层
        self.hash_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, bit),
            nn.Tanh()  # 输出范围在[-1, 1]
        )

    def forward(self, image, text=None):
        # 图像特征
        img_features = self.image_backbone(image)  # [batch_size, d_model]

        # 生成哈希码
        img_hash = self.hash_layer(img_features)

        # 如果有文本模态（未来扩展），可以处理text
        if text is not None:
            raise NotImplementedError("Text modality not supported in this configuration")

        return img_hash


class ARcHashLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(ARcHashLoss, self).__init__()
        self.bit = bit
        self.alpha = config["alpha"]  # 量化损失权重
        self.beta = config["beta"]  # 分类损失权重
        self.eta = config["eta"]  # 自适应鲁棒性权重
        self.n_class = config["n_class"]
        self.device = config["device"]

        # 分类层
        self.classifier = torch.nn.Linear(bit, self.n_class).to(self.device)
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)

    def forward(self, img_hash, y, ind, config):
        batch_size = img_hash.size(0)

        # 1. 分类损失
        img_cls = self.classifier(img_hash)
        cls_loss = self.criterion_cls(img_cls, y.argmax(axis=1))

        # 2. 量化损失 - 强制哈希码接近-1或1
        quant_loss = torch.mean((torch.abs(img_hash) - 1) ** 2)

        # 3. 自适应鲁棒性损失 - 基于标签相似性
        y_labels = y.argmax(axis=1)
        S = (y_labels.unsqueeze(0) == y_labels.unsqueeze(1)).float()

        # 计算图像哈希码相似性
        img_sim = torch.mm(img_hash, img_hash.t()) / self.bit
        robust_loss = torch.mean((S * img_sim - S) ** 2)

        # 总损失
        total_loss = (self.beta * cls_loss +
                      self.alpha * quant_loss +
                      self.eta * robust_loss)

        return total_loss


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train

    # 创建ARcHash网络
    net = config["net"](
        bit=bit,
        d_model=config["d_model"],
        dropout=config["dropout"]
    ).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    criterion = ARcHashLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, 训练中..." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:  # 适配图像数据加载
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            img_hash = net(image)  # 生成哈希码

            loss = criterion(img_hash, label.float(), ind, config)
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
        config["pr_curve_path"] = f"log/archash/ARcHash_{config['dataset']}_{bit}.json"
        train_val(config, bit)
