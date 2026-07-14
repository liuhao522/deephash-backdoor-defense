from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')



# DBDH(Neurocomputing2020)
# paper [Deep balanced discrete hashing for image retrieval](https://www.sciencedirect.com/science/article/abs/pii/S0925231220306032)
# [DBDH] epoch:150, bit:48, dataset:cifar10-1, MAP:0.792, Best MAP: 0.793
# [DBDH] epoch:80, bit:48, dataset:nuswide_21, MAP:0.833, Best MAP: 0.834
# [DBDH] epoch:150 bit:32 dataset:CIFAR10 MAP:0.8685966591429419 Best MAP: 0.8764188350089713
# [DBDH] epoch:150 bit:16 dataset:CIFAR10 MAP:0.8703498494713863 Best MAP: 0.8724828981090088
# [DBDH] epoch:150 bit:64 dataset:CIFAR10 MAP:0.8688282014152111 Best MAP: 0.8828527489426025

def get_config():
    config = {
        "alpha": 0.1,
        # "p": 1,
        "p": 2,
        # "optimizer": {"type": optim.SGD, "epoch_lr_decrease": 50,
        #               "optim_params": {"lr": 0.1, "weight_decay": 5e-4, "momentum": 0.9}},
        "optimizer": {"type": optim.RMSprop, "epoch_lr_decrease": 50,
                      "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[DBDH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": ResNet,
        "n_class": 100,
        # "net":ResNet,
        # 123
        # "dataset": "CIFAR10",
        # "dataset": "cifar10-1",
        # "dataset": "MNIST_trg",
        # "dataset": "MNIST",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
         "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 50,
        "test_map": 5,
        #  "save_path": "save/DBDH/MNIST128",
         "save_path": "save/DBDH/imagenet2",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [128],
    }
    config = config_dataset(config)
    return config


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
