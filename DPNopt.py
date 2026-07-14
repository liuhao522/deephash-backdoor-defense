from utils.tools import *
from network import *
import os
import torch
import torch.optim as optim
import time
import numpy as np
import random
import csv
from sklearn.decomposition import PCA
torch.multiprocessing.set_sharing_strategy('file_system')

def get_config():
    config = {
        "m": 1,    #margin的值，表示不同类样本之间的距离应该大于m
        "p": 0.5,  #生成目标哈希码时采用随机分配法，在每一位的去之上以概率p设置为1
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 1e-5}},
        "info": "[DPN]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": ResNet,
        "dataset": "CIFAR10",
        "epoch": 50,
        "test_map": 5,
        "device": torch.device("cuda:0"),
        "bit_list": [48],   #[8, 16,32,48,64]
        "save_path": "save/DPN_optimize/cifar_48",
    }
    config = config_dataset(config)
    return config

class DPNLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DPNLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.target_vectors = self.get_target_vectors(config["n_class"], bit, config["p"]).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.m = config["m"]
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])
        # 新添内容
        self.bit = bit
        # 假设 label_hash_mapping 是一个函数，将真实标签映射到预期的哈希码
        self.label_hash_mapping = self.create_label_hash_mapping(config["n_class"], bit, config["device"])

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()


        if "-T" in config["info"]:
            u = (u.abs() > self.m).float() * u.sign()

        label_loss = torch.nn.CrossEntropyLoss()(u, y.argmax(axis=1)) # 新增交叉熵损失函数来计算标签损失
        # 计算极化损失
        t = self.label2center(y)
        polarization_loss = (self.m - u * t).clamp(0).mean()

        # 新增：正则化项，减少哈希码和预期哈希码之间的差异
        expected_hash = self.label_hash_mapping(y)
        regularization_loss = torch.nn.functional.mse_loss(u, expected_hash)

        # 综合三种损失
        # 设计消融实验
        a = 0.6 # 消融实验得出0.6最好
        combined_loss = polarization_loss + regularization_loss
        total_loss = a * label_loss + (1 - a) * combined_loss
        # total_loss = a * label_loss + (1 - a) * combined_loss

        # total_loss = label_loss + polarization_loss + regularization_loss
        return total_loss
 # 新增方法
    def create_label_hash_mapping(self, n_class, bit, device):
        # 创建一个映射，将每个类别标签映射到一个预定义的哈希码
        # 这里使用简单的随机方法
        # mapping = {}
        # for i in range(n_class):
        #     mapping[i] = torch.randint(2, (bit,)).float()
        # return lambda labels: torch.stack([mapping[label.item()] for label in labels])

        # 使用PCA降维来创建哈希映射
        mapping = {}
        # 为每个类别生成多个高维点
        points_per_class = 10  # 每个类别生成的高维点数量
        high_dim_points = np.random.randn(n_class * points_per_class, bit * 2)
        # 应用PCA降维
        pca = PCA(n_components=bit)
        reduced_dim_points = pca.fit_transform(high_dim_points)
        # 为每个类别计算平均哈希码
        for i in range(n_class):
            avg_hash_code = np.mean(reduced_dim_points[i * points_per_class:(i + 1) * points_per_class], axis=0)
            hash_code = torch.tensor((avg_hash_code > 0), dtype=torch.float).to(device)  # 移动到指定设备
            mapping[i] = hash_code
        return lambda labels: torch.stack([mapping[torch.argmax(label).item()] for label in labels]).to(device)
    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.target_vectors[y.argmax(axis=1)]
        else:
            center_sum = y @ self.target_vectors
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    def get_target_vectors(self, n_class, bit, p=0.5):
        # 生成n_class行，bit列的全为0的tensor
        target_vectors = torch.zeros(n_class, bit)
        for k in range(20):
            for index in range(n_class):
                ones = torch.ones(bit)
                sa = random.sample(list(range(bit)), int(bit * p))
                ones[sa] = -1
                target_vectors[index] = ones
        return target_vectors

    # Adaptive Updating
    def update_target_vectors(self):
        self.U = (self.U.abs() > self.m).float() * self.U.sign()
        self.target_vectors = (self.Y.t() @ self.U).sign()



def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)  # 这里在最终有个输出（train_set,test,database）
    config["num_train"] = num_train
    # 将某个网络所有成员、函数、操作都搬移到GPU上面
    net = config["net"](bit).to(device)
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    criterion = DPNLoss(config, bit)
    Best_mAP = 0
    Best_acc = 0
    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train() # 设置神经网络模型为训练模式。
        train_loss = 0 # 初始化训练损失为0

        for image, label, ind in train_loader: # train_loader是一个数据加载器，用于按批次加载训练数据。

            image = image.to(device)
            label = label.to(device)


            optimizer.zero_grad() # 清零优化器的梯度，因为默认情况下 PyTorch 会累积梯度
            u = net(image) # 通过神经网络进行前向传播，得到模型的输出

            loss = criterion(u, label.float(), ind, config) # 计算损失，这是一个自定义的损失函数。输入参数包括模型输出 u、标签 label、索引 ind 和一些配置参数 config
            train_loss += loss.item() # 累加当前批次的损失值

            loss.backward() # 进行反向传播，计算梯度
            optimizer.step() # 更新模型参数，执行一步优化器的更新
        if "-A" in config["info"]:
            criterion.update_target_vectors()

        train_loss = train_loss / len(train_loader) # 计算平均训练损失

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0: # 如果当前轮数是测试间隔的倍数，执行以下代码块
            # bh新改
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
            # Best_mAP, majority_acc, weighting_acc = validate(config, Best_mAP, Best_acc, test_loader, dataset_loader, net, bit, epoch, num_dataset)
            # 调用 validate 函数进行模型的验证，获取验证结果。包括计算并保存最佳的平均精度 (mAP)，多数类准确度 (majority_acc)，和加权准确度 (weighting_acc)

            # print('precision:%f;recall:%f' % (precision, recall))
            # with open('./csv/DPN60.csv', mode='a', newline='') as performance_file:
            #     writer = csv.DictWriter(performance_file, fieldnames=fieldnames)
            #     writer.writerow({'epoch': epoch, 'Majority Voting': majority_acc*100, 'Hamming Voting': weighting_acc*100,'precision':precision*100,'recall':recall*100})


if __name__ == "__main__":
    config = get_config()
    print(config)
    print("以上是模型配置")
    for bit in config["bit_list"]: # 哈希码长度
        train_val(config, bit)
