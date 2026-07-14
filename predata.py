# -*- coding:utf-8 -*- 
# author:zhangning
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


batch_size = 1
transform = transforms.Compose([
    # Convert the PIL Image to Tensor
    transforms.ToTensor(),
    # The parameters are mean and std
    # transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data/mnist/',
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=batch_size
)
test_dataset = datasets.MNIST(
    root='./data/mnist/',
    train=False,
    download=True,
    transform=transform
)
test_loader = DataLoader(
    dataset=test_dataset,
    shuffle=False,
    batch_size=batch_size
)


def save_train_data():
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data

        inputs = inputs.numpy()
        targets = targets.numpy()

        inputs = inputs.reshape(28, 28)
        targets = targets[0]

        plt.imshow(inputs, cmap='gray')
        plt.axis('off')
        plt.savefig('./dataset/mnist_train/' + str(batch_idx) + '.jpg', bbox_inches='tight', pad_inches=0.0)

        with open('./dataset/mnist_train.txt', 'a') as f:
            f.write(str(batch_idx) + '-' + str(targets) + '.jpg' )
            f.write("\r\n")

        print(str(batch_idx) + '.jpg' + '-' + str(targets))


if __name__ == '__main__':
    save_train_data()
