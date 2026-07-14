# -*- coding:utf-8 -*- 
# author:zhangning

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Combofish
# Filename: main.py


from torchvision import datasets
from tqdm import tqdm
import os


train_data = datasets.MNIST(root="./data/", train=True, download=True)
test_data = datasets.MNIST(root="./data/", train=False, download=True)
saveDirdata = './data/'
saveDir = './images'

if not os.path.exists(saveDir):
    os.mkdir(saveDir)




def save_img(data, save_path):
    for i in tqdm(range(len(data))):
        img, label = data[i]
        img.save(os.path.join(save_path, str(i) + '-label-' + str(label) + '.png'))

def save_img1(data, save_path):
    for i in tqdm(range(len(data))):
        img, label = data[i]
        img.save(os.path.join(save_path, str(i+60000) + '-label-' + str(label) + '.png'))


save_img(train_data, saveDir)
save_img1(test_data, saveDir)

