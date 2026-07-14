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
from torch.nn import Module, Linear, BatchNorm1d, Sequential, ReLU

# 设置多进程共享策略
torch.multiprocessing.set_sharing_strategy('file_system')


def get_config():
    """获取配置参数"""
    base_data_path = './data/imagenet/'
    base_save_path = './save/DCWH/imagenet/'

    os.makedirs(base_save_path, exist_ok=True)

    config = {
        "alpha": 0.1,  # 量化损失权重
        "beta": 0.01,  # 分类损失权重
        "gamma": 0.1,  # 相似性损失权重
        "optimizer": {
            "type": optim.RAdam,
            "epoch_lr_decrease": 15,
            "optim_params": {
                "lr": 1e-4,
                "weight_decay": 10 ** -5
            }
        },
        "info": "[DCWH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,  # 减小batch size以防止内存不足
        "net": "EfficientNetV2",
        "n_class": 100,
        "dataset": "imagenet",
        "epoch": 50,
        "test_map": 5,
        "save_path": base_save_path,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "bit_list": [16, 32, 48, 64, 128],
        "topK": -1,
        "data_path": './dataset/imagenet/',
        "data": {
            "train_set": {
                "list_path": os.path.join(base_data_path, 'train.txt'),
                "batch_size": 16  # 减小batch size
            },
            "database": {
                "list_path": os.path.join(base_data_path, 'database.txt'),
                "batch_size": 16  # 减小batch size
            },
            "test": {
                "list_path": os.path.join(base_data_path, 'test.txt'),
                "batch_size": 16  # 减小batch size
            }
        },
        "grad_clip": 5.0,  # 梯度裁剪阈值
        "use_amp": True  # 启用混合精度训练
    }
    config = config_dataset(config)
    return config


class HashLayer(Module):
    """哈希层"""

    def __init__(self, input_dim, output_dim):
        super(HashLayer, self).__init__()
        self.fc = Linear(input_dim, output_dim)
        self.bn = BatchNorm1d(output_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.tanh(x)
        return x


class DCWH_Model(Module):
    """DCWH模型主网络"""

    def __init__(self, bit, n_class):
        super(DCWH_Model, self).__init__()
        self.bit = bit
        self.n_class = n_class

        # 图像特征提取网络
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.efficientnet._fc.in_features

        # 特征提取部分
        self.feature_extractor = Sequential(
            Linear(in_features, 1024),  # 减小中间层维度
            ReLU(),
            BatchNorm1d(1024)
        )

        # 哈希层
        self.hash_layer = HashLayer(1024, bit)

        # 分类器
        self.classifier = Linear(bit, n_class)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # 提取特征
        x = self.efficientnet.extract_features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()

        # 特征提取
        features = self.feature_extractor(x)

        # 生成哈希码
        hash_code = self.hash_layer(features)

        # 分类预测
        cls_pred = self.classifier(hash_code)

        return features, hash_code, cls_pred


class DCWH_Loss(Module):
    """DCWH损失函数"""

    def __init__(self, config):
        super(DCWH_Loss, self).__init__()
        self.config = config
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]
        self.eps = 1e-9  # 防止数值不稳定

    def forward(self, features, hash_code, cls_pred, labels, index):
        # 确保标签是浮点类型
        labels = labels.float()

        batch_size = hash_code.size(0)

        # 相似性损失
        S = (torch.matmul(labels, labels.t()) > 0).float()

        # 计算特征相似性 (加入数值稳定性处理)
        features_norm = F.normalize(features, p=2, dim=1)
        theta = torch.matmul(features_norm, features_norm.t()) / 2
        theta = torch.clamp(theta, -1.0 + self.eps, 1.0 - self.eps)
        sim_loss = F.binary_cross_entropy_with_logits(theta, S)

        # 量化损失 (将哈希码推向-1或1)
        quant_loss = torch.mean((torch.abs(hash_code) - 1) ** 2)

        # 分类损失
        cls_loss = F.cross_entropy(cls_pred, labels.argmax(dim=1))

        # 总损失
        total_loss = self.gamma * sim_loss + self.alpha * quant_loss + self.beta * cls_loss

        return total_loss


def validate_dcwh(config, best_mAP, test_loader, database_loader, net, bit, epoch, num_dataset):
    """DCWH专用的验证函数"""
    net.eval()
    device = config["device"]

    # 生成数据库哈希码
    database_hash = torch.zeros(num_dataset, bit).to(device)
    database_labels = torch.zeros(num_dataset, config["n_class"]).to(device)

    with torch.no_grad():
        for images, labels, indices in database_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            _, hash_code, _ = net(images)
            database_hash[indices, :] = hash_code.sign()
            database_labels[indices, :] = labels

    # 生成测试集哈希码
    test_hash = []
    test_labels = []
    for images, labels, _ in test_loader:
        images = images.to(device)
        labels = labels.float().to(device)
        _, hash_code, _ = net(images)
        test_hash.append(hash_code.sign().cpu())
        test_labels.append(labels.cpu())

    test_hash = torch.cat(test_hash, 0)
    test_labels = torch.cat(test_labels, 0)

    # 计算mAP
    mAP = compute_mAP(test_hash, database_hash.cpu(), test_labels, database_labels.cpu())

    print(f"测试mAP@{bit}bit: {mAP:.4f}")
    if mAP > best_mAP:
        best_mAP = mAP
        torch.save(net.state_dict(), os.path.join(config["save_path"], f"model_{bit}bit_{mAP:.4f}.pth"))
    return best_mAP


def train_val(config, bit):
    """DCWH训练和验证函数"""
    device = config["device"]

    # 获取数据加载器
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train

    # 初始化网络
    net = DCWH_Model(bit, config["n_class"]).to(device)

    # 初始化优化器
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    # 初始化损失函数
    criterion = DCWH_Loss(config)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])

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
                label = label.float().to(device)

                optimizer.zero_grad()

                # 混合精度训练
                with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                    features, hash_code, cls_pred = net(image)
                    loss = criterion(features, hash_code, cls_pred, label, ind)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(net.parameters(), config["grad_clip"])

                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            train_loss = train_loss / len(train_loader)
            print("\b\b\b\b\b\b\b 损失:%.3f" % (train_loss))

            # 定期测试模型性能
            if (epoch + 1) % config["test_map"] == 0:
                current_mAP = validate_dcwh(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
                if current_mAP > Best_mAP:
                    Best_mAP = current_mAP

    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
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
        try:
            results = train_val(config, bit)
            final_results.update(results)
            print(f"\n{bit}-bit 模型结果: mAP = {results[bit]:.4f}")
        except Exception as e:
            print(f"{bit}-bit 模型训练失败: {str(e)}")
            continue

    print("\n最终结果:")
    for bit, mAP in final_results.items():
        print(f"{bit}-bit 模型 mAP: {mAP:.4f}")