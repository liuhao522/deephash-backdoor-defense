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
import psutil
import gc

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
        "batch_size": 8,  # 减小batch_size以节省内存
        "net": "EfficientNetV2",
        "n_class": 100,
        "dataset": "imagenet",
        "epoch": 50,
        "test_map": 10,  # 减少验证频率
        "save_path": base_save_path,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "bit_list": [16, 32, 48, 64],  # 移除了128-bit以减少内存消耗
        "topK": -1,
        "data_path": './dataset/imagenet/',
        "data": {
            "train_set": {
                "list_path": os.path.join(base_data_path, 'train.txt'),
                "batch_size": 8  # 减小batch_size以节省内存
            },
            "database": {
                "list_path": os.path.join(base_data_path, 'database.txt'),
                "batch_size": 8  # 减小batch_size以节省内存
            },
            "test": {
                "list_path": os.path.join(base_data_path, 'test.txt'),
                "batch_size": 8  # 减小batch_size以节省内存
            }
        },
        "grad_clip": 5.0,
        "use_amp": True,
        "checkpoint_interval": 10,  # 增加检查点间隔
        "memory_safe": True,
        "enable_gradient_checkpointing": True  # 新增梯度检查点
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

        # 启用梯度检查点
        if hasattr(self.efficientnet, 'set_grad_checkpointing'):
            self.efficientnet.set_grad_checkpointing(True)

        # 特征提取部分
        self.feature_extractor = Sequential(
            Linear(in_features, 512),  # 减小中间层维度
            ReLU(),
            BatchNorm1d(512)
        )

        # 哈希层
        self.hash_layer = HashLayer(512, bit)  # 输入维度相应调整

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
        self.eps = 1e-9

    def forward(self, features, hash_code, cls_pred, labels, index):
        labels = labels.float()
        batch_size = hash_code.size(0)

        # 相似性损失
        S = (torch.matmul(labels, labels.t()) > 0).float()
        features_norm = F.normalize(features, p=2, dim=1)
        theta = torch.matmul(features_norm, features_norm.t()) / 2
        theta = torch.clamp(theta, -1.0 + self.eps, 1.0 - self.eps)
        sim_loss = F.binary_cross_entropy_with_logits(theta, S)

        # 量化损失
        quant_loss = torch.mean((torch.abs(hash_code) - 1) ** 2)

        # 分类损失
        cls_loss = F.cross_entropy(cls_pred, labels.argmax(dim=1))

        # 总损失
        total_loss = self.gamma * sim_loss + self.alpha * quant_loss + self.beta * cls_loss
        return total_loss


def validate_dcwh(config, best_mAP, test_loader, database_loader, net, bit, epoch, num_dataset):
    """验证函数"""
    net.eval()
    device = config["device"]

    # 分批次处理数据库以减少内存使用
    database_hash = []
    database_labels = []

    with torch.no_grad():
        for images, labels, _ in database_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            _, hash_code, _ = net(images)
            database_hash.append(hash_code.sign().cpu())
            database_labels.append(labels.cpu())
            # 及时清理内存
            del images, labels, hash_code
            torch.cuda.empty_cache()

    database_hash = torch.cat(database_hash, 0)
    database_labels = torch.cat(database_labels, 0)

    # 生成测试集哈希码
    test_hash = []
    test_labels = []
    for images, labels, _ in test_loader:
        images = images.to(device)
        labels = labels.float().to(device)
        _, hash_code, _ = net(images)
        test_hash.append(hash_code.sign().cpu())
        test_labels.append(labels.cpu())
        # 及时清理内存
        del images, labels, hash_code
        torch.cuda.empty_cache()

    test_hash = torch.cat(test_hash, 0)
    test_labels = torch.cat(test_labels, 0)

    # 计算mAP
    mAP = compute_mAP(test_hash, database_hash, test_labels, database_labels)
    print(f"测试mAP@{bit}bit: {mAP:.4f}")

    if mAP > best_mAP:
        best_mAP = mAP
        torch.save(net.state_dict(), os.path.join(config["save_path"], f"model_{bit}bit_{mAP:.4f}.pth"))

    # 清理内存
    del database_hash, database_labels, test_hash, test_labels
    torch.cuda.empty_cache()
    gc.collect()

    return best_mAP


def check_memory():
    """检查内存使用情况"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"内存使用: RSS={mem_info.rss / 1024 / 1024:.2f}MB, VMS={mem_info.vms / 1024 / 1024:.2f}MB")
    if torch.cuda.is_available():
        print(
            f"GPU内存: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}MB / {torch.cuda.memory_reserved() / 1024 / 1024:.2f}MB")


def cleanup_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_val(config, bit):
    """训练和验证函数"""
    device = config["device"]

    # 设置环境变量以减少内存碎片
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # 获取数据加载器
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train

    # 初始化网络
    net = DCWH_Model(bit, config["n_class"]).to(device)

    # 初始化优化器
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    # 初始化损失函数
    criterion = DCWH_Loss(config)

    # 混合精度训练 - 使用旧版API
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

                # 使用旧版autocast
                with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                    features, hash_code, cls_pred = net(image)
                    loss = criterion(features, hash_code, cls_pred, label, ind)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

                # 及时清理中间变量
                del features, hash_code, cls_pred, loss

                # 内存安全模式
                if config["memory_safe"] and (ind[0] % 50 == 0):  # 更频繁的内存清理
                    cleanup_memory()

            train_loss = train_loss / len(train_loader)
            print("\b\b\b\b\b\b\b 损失:%.3f" % (train_loss))

            # 定期检查内存
            if epoch % 5 == 0:
                check_memory()

            # 定期测试
            if (epoch + 1) % config["test_map"] == 0 or (epoch + 1) == config["epoch"]:
                current_mAP = validate_dcwh(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
                if current_mAP > Best_mAP:
                    Best_mAP = current_mAP

            # 保存检查点
            if config["checkpoint_interval"] > 0 and (epoch + 1) % config["checkpoint_interval"] == 0:
                checkpoint_path = os.path.join(config["save_path"], f"checkpoint_{bit}bit_epoch{epoch + 1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mAP': Best_mAP,
                    'loss': train_loss,
                }, checkpoint_path)
                print(f"已保存检查点到 {checkpoint_path}")

    except Exception as e:
        print(f"\n训练错误: {str(e)}")
        import traceback
        traceback.print_exc()
        emergency_path = os.path.join(config["save_path"], f"emergency_save_{bit}bit_epoch{epoch + 1}.pth")
        torch.save(net.state_dict(), emergency_path)
        print(f"已紧急保存模型到 {emergency_path}")
        raise e

    results[bit] = Best_mAP
    return results


if __name__ == "__main__":
    config = get_config()
    print("配置参数:", config)

    if not torch.cuda.is_available():
        print("警告: 使用CPU训练，速度会很慢!")

    final_results = {}
    for bit in config["bit_list"]:
        print(f"\n开始训练 {bit}-bit 模型...")
        try:
            cleanup_memory()
            results = train_val(config, bit)
            final_results.update(results)
            print(f"\n{bit}-bit 结果: mAP = {results[bit]:.4f}")
            cleanup_memory()
        except Exception as e:
            print(f"{bit}-bit 训练失败: {str(e)}")
            continue

    print("\n最终结果:")
    for bit, mAP in final_results.items():
        print(f"{bit}-bit mAP: {mAP:.4f}")