from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
import random

torch.multiprocessing.set_sharing_strategy('file_system')


# SSDH (Supervised Semantics-preserving Deep Hashing)
# 论文《Supervised Semantics-preserving Deep Hashing for Semantic Similarity Search》
# 注意：SSDH是一种逐点的哈希方法，保持语义相似性

def get_config():
    config = {
        "alpha": 0.1,  # 量化损失的权重
        "beta": 0.1,  # 分类损失的权重
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[SSDH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": ResNet,
        "dataset": "GTSRB",
        "epoch": 50,
        "test_map": 10,
        "device": torch.device("cuda:0"),
        "bit_list": [32],
    }
    config = config_dataset(config)
    return config


class SSDHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(SSDHLoss, self).__init__()
        self.bit = bit
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.n_class = config["n_class"]

        # 分类层，将哈希码映射到类别数量
        self.classifier = torch.nn.Linear(bit, self.n_class).to(config["device"])
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(config["device"])
        self.criterion_quant = torch.nn.MSELoss().to(config["device"])

    def forward(self, u, y, ind, config):
        # 分类部分
        cls_output = self.classifier(u)

        # 对于单标签数据集，使用CrossEntropyLoss
        if config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}:
            cls_loss = self.criterion_cls(cls_output, y.argmax(axis=1))
        else:
            # 对于多标签数据集，使用BCEWithLogitsLoss
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(cls_output, y.float())

        # 量化损失，强制输出接近-1或1
        quant_loss = torch.mean((torch.abs(u) - 1) ** 2)

        # 总损失
        total_loss = self.beta * cls_loss + self.alpha * quant_loss

        return total_loss


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = SSDHLoss(config, bit)

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
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b 损失:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/alexnet/SSDH_{config['dataset']}_{bit}.json"
        train_val(config, bit)