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


# ASDH (Asymmetric Semantics-preserving Deep Hashing)
# 融合ADSH的非对称结构和SSDH的语义保持能力

def get_config():
    config = {
        # 融合参数设计
        "alpha": 0.1,  # 量化损失权重 (来自SSDH)
        "beta": 0.5,  # 分类损失权重 (增强语义)
        "gamma": 1.2,  # 相似性损失权重 (来自ADSH，略微增强)
        "eta": 0.05,  # 非对称正则化权重
        "lambda_sem": 0.3,  # 语义一致性损失权重 (新增)
        "temperature": 0.5,  # 温度参数用于相似性计算

        "optimizer": {
            "type": optim.AdamW,  # 使用AdamW优化器
            "optim_params": {
                "lr": 2e-4,
                "weight_decay": 1e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-8
            }
        },
        "info": "[ASDH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": ResNet,
        "dataset": "GTSRB",
        "epoch": 50,  # 增加训练轮数
        "test_map": 5,
        "device": torch.device("cuda:0"),
        "bit_list": [64],
        "topK": 5000,
    }
    config = config_dataset(config)
    return config


class ASDHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(ASDHLoss, self).__init__()
        self.bit = bit
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]
        self.eta = config["eta"]
        self.lambda_sem = config["lambda_sem"]
        self.temperature = config["temperature"]
        self.n_class = config["n_class"]
        self.device = config["device"]

        # 双分支分类器设计（非对称结构）
        self.query_classifier = torch.nn.Sequential(
            torch.nn.Linear(bit, bit // 2),
            torch.nn.BatchNorm1d(bit // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(bit // 2, self.n_class)
        ).to(self.device)

        self.database_classifier = torch.nn.Sequential(
            torch.nn.Linear(bit, bit // 2),
            torch.nn.BatchNorm1d(bit // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(bit // 2, self.n_class)
        ).to(self.device)

        # 语义嵌入层（新增）
        self.semantic_projector = torch.nn.Sequential(
            torch.nn.Linear(bit, bit),
            torch.nn.BatchNorm1d(bit),
            torch.nn.Tanh()
        ).to(self.device)

        self.criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)
        self.last_database_features = None

    def forward(self, u, y, ind, config, is_query=True):
        batch_size = u.size(0)

        # 数值稳定性处理
        u = torch.clamp(u, -15, 15)

        # 1. 分类损失（非对称结构）
        if is_query:
            cls_output = self.query_classifier(u)
        else:
            cls_output = self.database_classifier(u)

        if config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}:
            cls_loss = self.criterion_cls(cls_output, y.argmax(axis=1))
        else:
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                cls_output, y.float(), reduction='mean')

        # 2. 改进的量化损失（结合SSDH思想）
        # 使用Huber损失替代MSE，更鲁棒
        abs_diff = torch.abs(torch.abs(u) - 1)
        quant_loss = torch.where(abs_diff < 1.0,
                                 0.5 * abs_diff ** 2,
                                 abs_diff - 0.5).mean()

        # 3. 相似性保持损失（来自ADSH，改进版）
        S = (y @ y.t() > 0).float()
        # 添加对角线掩码，避免自相似
        mask = torch.eye(batch_size, device=self.device)
        S = S * (1 - mask)

        # 温度缩放的相似性计算
        theta = (u @ u.t()) / (self.bit * self.temperature)
        theta = torch.clamp(theta, -10, 10)

        # 使用focal loss思想加权难样本
        pos_loss = -S * torch.log(torch.sigmoid(theta) + 1e-8)
        neg_loss = -(1 - S) * torch.log(1 - torch.sigmoid(theta) + 1e-8)
        sim_loss = (pos_loss + neg_loss).mean()

        # 4. 语义一致性损失（新增）
        semantic_embed = self.semantic_projector(u)
        # 语义特征应该与原始哈希码保持一定相关性
        semantic_loss = 1 - torch.nn.functional.cosine_similarity(
            u, semantic_embed, dim=1).mean()

        # 5. 非对称正则化损失
        # 鼓励query和database分支学到不同但互补的表示
        asymm_loss = torch.tensor(0.0, device=self.device)
        if is_query and self.last_database_features is not None:
            # 确保比较相同数量的样本
            min_batch = min(batch_size, self.last_database_features.size(0))
            asymm_loss = torch.nn.functional.mse_loss(
                u[:min_batch],
                self.last_database_features[:min_batch].detach())

        if not is_query:
            self.last_database_features = u.clone()

        # 总损失（动态权重平衡）
        total_loss = (self.beta * cls_loss +
                      self.alpha * quant_loss +
                      self.gamma * sim_loss +
                      self.lambda_sem * semantic_loss +
                      self.eta * asymm_loss)

        return total_loss, cls_loss, sim_loss, quant_loss, semantic_loss


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    criterion = ASDHLoss(config, bit)
    Best_mAP = 0
    writer = SummaryWriter(f"runs/ASDH_{config['dataset']}_{bit}bit")

    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, 训练中..." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        train_loss = 0
        total_cls_loss = 0
        total_sim_loss = 0
        total_quant_loss = 0
        total_sem_loss = 0

        for batch_idx, (image, label, ind) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            # 随机选择作为query还是database进行训练
            is_query = random.random() > 0.5
            loss, cls_loss, sim_loss, quant_loss, sem_loss = criterion(
                u, label.float(), ind, config, is_query=is_query)

            train_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_sim_loss += sim_loss.item()
            total_quant_loss += quant_loss.item()
            total_sem_loss += sem_loss.item()

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            optimizer.step()

        # 学习率调度
        scheduler.step()

        # 计算平均损失
        avg_loss = train_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_sim_loss = total_sim_loss / len(train_loader)
        avg_quant_loss = total_quant_loss / len(train_loader)
        avg_sem_loss = total_sem_loss / len(train_loader)

        print("\b\b\b\b\b\b\b 总损失:%.3f" % avg_loss)

        # 记录训练指标
        writer.add_scalar('Loss/Total', avg_loss, epoch)
        writer.add_scalar('Loss/Classification', avg_cls_loss, epoch)
        writer.add_scalar('Loss/Similarity', avg_sim_loss, epoch)
        writer.add_scalar('Loss/Quantization', avg_quant_loss, epoch)
        writer.add_scalar('Loss/Semantic', avg_sem_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)

    writer.close()


if __name__ == "__main__":
    config = get_config()
    print("ASDH配置:", config)
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/ASDH_{config['dataset']}_{bit}.json"
        train_val(config, bit)