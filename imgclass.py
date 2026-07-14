# -*- coding:utf-8 -*- 
# author:zhangning
import os
import shutil

# 原始数据的存储地址
path = './dataset/cifar10/images'
path1 = './cifar10datasclass'
# 读取地址中的文件，以列表形式存储到files_list中
files_list = os.listdir(path)
name = []
# 遍历列表中的所有文件
for file in files_list:

    # 用split函数将文件的名称以‘-’切片3次，取切片后列表的第一个项，即得到了文件名的第一个数字
    e = file[file.rfind('-'):file.rfind('.')]  # print 'A935
    name1 = e[1:]
    name.append(name1)
    # print(name1)

    # 如果以文件第一个数字命名的文件夹不存在，就创建一个
    # os.path.join()函数用于拼接文件路径
    if not os.path.exists(os.path.join(path1, name1)):
        os.makedirs(os.path.join(path1, name1))

    # 转移原始文件到新的文件夹中
    shutil.copy(os.path.join(path, file), os.path.join(path1, name1))

print(len(name))