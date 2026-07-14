from utils.tools import *
from network import *
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# PCDH(Neurocomputing 2020)
# paper [Deep discrete hashing with pairwise correlation learning](https://www.sciencedirect.com/science/article/pii/S092523121931793X)
# [PCDH] epoch:720, bit:48, dataset:nuswide_21, MAP:0.653, Best MAP: 0.659
# [PCDH] epoch:1785, bit:48, dataset:cifar10-1, MAP:0.166, Best MAP: 0.168

def get_config():
    config = {
        "alpha": 0.1,
        # "p": 1,
        "p": 2,
        # "optimizer": {"type": optim.SGD, "epoch_lr_decrease": 50,
        #               "optim_params": {"lr": 0.1, "weight_decay": 5e-4, "momentum": 0.9}},
        "optimizer": {"type": optim.RMSprop, "epoch_lr_decrease": 50,
                      "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[IDHN]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": ResNet,
        "n_class": 10,
        # "net":ResNet,
         "dataset": "GTSRB",
        # "dataset": "cifar10-1",
        # "dataset": "MNIST_trg",
        # "dataset": "MNIST",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
        # "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 50,
        "test_map": 5,
        "save_path": "save/PCDHhide/cifar10",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [128],
    }
    config = config_dataset(config)
    return config



class Net(nn.Module):
    def __init__(self, hash_bit, num_classes, pretrained=True):
        super(Net, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )
        self.feature_layer = nn.Linear(8 * 8 * 256, 1024)
        self.hash_like_layer = nn.Sequential(nn.Linear(1024, hash_bit), nn.Tanh())
        self.discrete_hash_layer = nn.Linear(hash_bit, hash_bit)
        self.classification_layer = nn.Linear(hash_bit, num_classes, bias=False)

    def forward(self, x, istraining=False):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        feature = self.feature_layer(x)
        h = self.hash_like_layer(feature)
        b = self.discrete_hash_layer(h).add(1).mul(0.5).clamp(min=0, max=1)
        b = (b >= 0.5).float() * 2 - 1
        y_pre = self.classification_layer(b)
        if istraining:
            return feature, h, y_pre
        else:
            return b


class DPSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DPSHLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        u = u.clamp(min=-1, max=1)
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = (y @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5

        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product

        likelihood_loss = likelihood_loss.mean()

        if config["p"] == 1:
            quantization_loss = config["alpha"] * u.mean(dim=1).abs().mean()
        else:
            quantization_loss = config["alpha"] * u.mean(dim=1).pow(2).mean()

        return likelihood_loss + quantization_loss

def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DPSHLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        lr = config["optimizer"]["optim_params"]["lr"] * (0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)