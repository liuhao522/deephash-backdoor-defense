# -*- coding:utf-8 -*- 
# author:zhangning
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from utils.tool import *
from network import *
import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def get_config():
    config = {
        "alpha": 0.1,
        # "p": 1,
        "p": 2,
        # "optimizer": {"type": optim.SGD, "epoch_lr_decrease": 50,
        #               "optim_params": {"lr": 0.1, "weight_decay": 5e-4, "momentum": 0.9}},
        "optimizer": {"type": optim.RMSprop, "epoch_lr_decrease": 50,
                      "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[Alexnet]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": AlexNet,
        "n_class": 10,
        # "net":ResNet,
        # "dataset": "cifar10",
        # "dataset": "cifar10-1",
        # "dataset": "MNIST_trg",
        # "dataset": "MNIST_alexnet",
        "dataset": "MNIST",
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
        "save_path": "save/AlexNet/MNIST",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
    }
    config = config_dataset(config)
    return config



def train_val(config):
    device = config["device"]
    train_loader, test_loader, num_train, num_test = get_data(config)
    config["num_train"] = num_train
    net = config["net"]().to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = nn.CrossEntropyLoss()


    for epoch in range(config["epoch"]):

        lr = config["optimizer"]["optim_params"]["lr"] * (0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s], dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = net(image)

            loss = criterion(outputs, label)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))


if __name__ == "__main__":
    config = get_config()
    print(config)
    train_val(config)



# 保存模型
# torch.save(model.state_dict(), 'mnist_alexnet_model.pth')
