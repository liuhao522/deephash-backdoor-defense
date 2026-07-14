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
import gc

# 设置多进程共享策略
torch.multiprocessing.set_sharing_strategy('file_system')


def get_config():
    """获取配置参数"""
    # 基础路径设置
    base_data_path = './data/imagenet/'
    base_save_path = './save/CSQ/imagenet/'

    # 确保保存路径存在
    os.makedirs(base_save_path, exist_ok=True)

    config = {
        "gamma": 1,
        "optimizer": {
            "type": optim.Adam,
            "epoch_lr_decrease": 20,
            "optim_params": {
                "lr": 3e-5,
                "weight_decay": 1e-5
            }
        },
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 24,
        "net": "EfficientNetV2",
        "n_class": 100,
        "dataset": "imagenet",
        "epoch": 50,
        "test_map": 5,
        "save_path": base_save_path,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "bit_list": [16, 32, 48, 64, 128],
        "topK": -1,
        "data_path": './dataset/imagenet/',
        "grad_clip": 5.0,
        "data": {
            "train_set": {
                "list_path": os.path.join(base_data_path, 'train.txt'),
                "batch_size": 24
            },
            "database": {
                "list_path": os.path.join(base_data_path, 'database.txt'),
                "batch_size": 24
            },
            "test": {
                "list_path": os.path.join(base_data_path, 'test.txt'),
                "batch_size": 24
            }
        }
    }
    config = config_dataset(config)
    return config


class EfficientNetV2_CSQ(torch.nn.Module):
    """CSQ专用的EfficientNetV2网络结构"""

    def __init__(self, bit):
        super(EfficientNetV2_CSQ, self).__init__()
        # 加载预训练的EfficientNetV2
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        # 冻结部分层以减少内存使用
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        # 只解冻最后几层
        for param in self.efficientnet._blocks[-4:].parameters():
            param.requires_grad = True
        for param in self.efficientnet._conv_head.parameters():
            param.requires_grad = True
        for param in self.efficientnet._bn1.parameters():
            param.requires_grad = True
        for param in self.efficientnet._fc.parameters():
            param.requires_grad = True

        # 替换最后的全连接层为哈希层（保持与之前保存的模型结构一致）
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, bit))

        # 初始化权重
        torch.nn.init.kaiming_normal_(self.efficientnet._fc[0].weight)
        torch.nn.init.kaiming_normal_(self.efficientnet._fc[2].weight)
        self.efficientnet._fc[0].bias.data.zero_()
        self.efficientnet._fc[2].bias.data.zero_()

    def forward(self, x):
        x = self.efficientnet(x)
        return torch.tanh(x)


class CSQLoss(torch.nn.Module):
    """CSQ损失函数"""

    def __init__(self, config, bit):
        super(CSQLoss, self).__init__()
        self.config = config
        self.bit = bit
        # 初始化中心点 (基于Hadamard矩阵)
        self.centers = self.init_centers(config["n_class"], bit).to(config["device"])
        self.centers.requires_grad = False

    def init_centers(self, n_class, bit):
        """使用Hadamard矩阵初始化中心点"""
        if bit < n_class:
            centers = torch.randn(n_class, bit).sign()
        else:
            hadamard = self.hadamard_matrix(bit)
            centers = hadamard[:n_class]

        # 归一化到单位超球面
        centers = F.normalize(centers, p=2, dim=1)
        return centers

    def hadamard_matrix(self, n):
        """生成Hadamard矩阵"""
        H = torch.tensor([[1]], dtype=torch.float32)
        while H.shape[0] < n:
            H = torch.cat((torch.cat((H, H), dim=1), torch.cat((H, -H), dim=1)), dim=0)
        return H[:n, :n]

    def forward(self, u, y, ind):
        u = u.clamp(min=-1, max=1)

        # 计算目标中心 (基于标签)
        with torch.no_grad():
            target_centers = torch.matmul(y.float(), self.centers)
            target_centers = F.normalize(target_centers, p=2, dim=1)

        # 中心相似性损失
        center_loss = F.mse_loss(u, target_centers)

        # 量化损失
        quantization_loss = torch.mean((u.abs() - 1).pow(2))

        # 总损失
        total_loss = center_loss + self.config["gamma"] * quantization_loss

        return total_loss


def load_model_with_compatibility(net, model_path, device):
    """兼容性模型加载函数"""
    try:
        # 加载完整模型
        net.load_state_dict(torch.load(model_path, map_location=device))
        print("完全加载预训练模型权重成功")
        return True
    except RuntimeError as e:
        print(f"完全加载失败: {str(e)}")
        print("尝试部分加载兼容的权重...")

        try:
            # 获取模型当前状态字典
            model_dict = net.state_dict()

            # 加载保存的状态字典
            pretrained_dict = torch.load(model_path, map_location=device)

            # 1. 过滤掉不匹配的键
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 2. 过滤掉尺寸不匹配的参数
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if v.size() == model_dict[k].size()}

            # 3. 更新当前模型的状态字典
            model_dict.update(pretrained_dict)

            # 4. 加载处理后的状态字典
            net.load_state_dict(model_dict, strict=False)

            print(f"部分加载成功，加载了{len(pretrained_dict)}/{len(model_dict)}个参数")
            return True
        except Exception as e:
            print(f"部分加载失败: {str(e)}")
            return False


def train_val(config, bit):
    """CSQ训练和验证函数"""
    device = config["device"]

    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()

    # 获取数据加载器
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train

    # 初始化网络
    net = EfficientNetV2_CSQ(bit).to(device)

    # 检查是否有保存的模型可以加载
    model_path = os.path.join(config["save_path"], f"best_model_{bit}bit.pth")
    if os.path.exists(model_path):
        print(f"尝试加载已存在的模型: {model_path}")
        if not load_model_with_compatibility(net, model_path, device):
            print("将从头开始训练模型")
    else:
        print("没有找到预训练模型，将从头开始训练")

    # 初始化优化器
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    # 初始化学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # 初始化损失函数
    criterion = CSQLoss(config, bit)

    Best_mAP = 0
    results = {}

    try:
        for epoch in range(config["epoch"]):
            # 动态调整学习率
            current_lr = config["optimizer"]["optim_params"]["lr"] * (
                    0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print("%s[%2d/%2d][%s] bit:%d, dataset:%s, 训练中..." % (
                config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

            net.train()
            train_loss = 0

            for image, label, ind in train_loader:
                image = image.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                u = net(image)

                loss = criterion(u, label, ind)
                train_loss += loss.item()

                loss.backward()

                # 梯度裁剪
                if config["grad_clip"] is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), config["grad_clip"])

                optimizer.step()

            train_loss = train_loss / len(train_loader)
            print("\b\b\b\b\b\b\b 损失:%.3f, lr:%.2e" % (train_loss, current_lr))

            # 定期测试模型性能
            if (epoch + 1) % config["test_map"] == 0 or (epoch + 1) == config["epoch"]:
                try:
                    current_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
                    scheduler.step(current_mAP)

                    if current_mAP > Best_mAP:
                        Best_mAP = current_mAP
                        torch.save(net.state_dict(), os.path.join(config["save_path"], f"best_model_{bit}bit.pth"))
                        # 保存完整检查点
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': train_loss,
                            'mAP': current_mAP,
                            'config': config
                        }, os.path.join(config["save_path"], f"checkpoint_{bit}bit.pth"))

                    # 定期清理内存
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n验证过程中出现错误: {str(e)}")
                    # 保存紧急备份
                    torch.save(net.state_dict(),
                               os.path.join(config["save_path"], f"emergency_save_{bit}bit_epoch{epoch}.pth"))
                    continue

    except KeyboardInterrupt:
        print("\n训练被用户中断，保存当前模型...")
        torch.save(net.state_dict(), os.path.join(config["save_path"], f"interrupted_{bit}bit.pth"))
        return results
    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
        # 保存紧急备份
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
            # 确保每次训练前清理内存
            torch.cuda.empty_cache()
            gc.collect()

            results = train_val(config, bit)
            final_results.update(results)
            print(f"\n{bit}-bit 模型结果: mAP = {results[bit]:.4f}")
        except Exception as e:
            print(f"\n{bit}-bit 模型训练失败: {str(e)}")
            final_results[bit] = 0.0
            continue

    print("\n最终结果:")
    for bit, mAP in final_results.items():
        print(f"{bit}-bit 模型 mAP: {mAP:.4f}")