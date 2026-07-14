from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
from scipy.linalg import hadamard
import random

torch.multiprocessing.set_sharing_strategy('file_system')

def get_config():
    config = {
        "lambda": 0.0001,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": ResNet,
        "dataset": "GTSRB",
        "epoch": 50,
        "test_map": 10,
        "device": torch.device("cuda:0"),
        "bit_list": [32],
        "grad_clip": None,  # Remove gradient clipping or set to a larger value
        "color_jitter": False,  # Disable strong data augmentation
        "random_grayscale": False,  # Disable random grayscale
        "patience": 20,  # Keep early stopping for safety
    }
    config = config_dataset(config)
    return config

class CSQLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CSQLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.criterion = torch.nn.BCELoss().to(config["device"])
        self.lambda_val = config["lambda"]

    def forward(self, u, y, ind, config):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        Q_loss = (u.abs() - 1).pow(2).mean()
        return center_loss + self.lambda_val * Q_loss

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

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

    criterion = CSQLoss(config, bit)

    Best_mAP = 0
    no_improve_epochs = 0

    for epoch in range(config["epoch"]):
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

            # Apply gradient clipping if enabled
            if config["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), config["grad_clip"])

            optimizer.step()

        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            current_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)

            if current_mAP > Best_mAP:
                Best_mAP = current_mAP
                no_improve_epochs = 0
                torch.save(net.state_dict(), f"best_model_{bit}bit.pth")
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= config["patience"]:
                print(f"No improvement for {config['patience']} epochs, early stopping...")
                break

if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/alexnet/CSQ_{config['dataset']}_{bit}.json"
        train_val(config, bit)