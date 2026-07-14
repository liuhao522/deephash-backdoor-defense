# -*- coding:utf-8 -*- 
# author:zhangning
import numpy as np


def transfbinadry(x):
    for i in range(len(x)):
        if x[i] > 0:
            x[i] = 1
        else:
            x[i] = 0
    return x


def CalcHammingDist(B1, B2):
    q = 48
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


file = np.load('./save/DBDH/MNIST_trg/MNIST_48bits_0.9454916253005199/tst_binary.npy', encoding="latin1")
# print(file.shape)
# print(file)
file_label = np.load('./save/DBDH/MNIST_trg/MNIST_trg_48bits_0.9687604580969531/tst_label.npy')
# print(file_label.shape)
# print(file_label)
# np.savetxt('./save/DBDH/MNIST/MNIST_48bits_0.9764698659401454/trn_binary.txt', file)
# 将numpy矩阵数据存入txt文件
# doc = open('tst_binary.txt', 'a') # 打开一个存储文件，并依次写入
# for i in range(len(file)):
#     doc.write(str(file[i])+'\n')
# doc.close()
file1 = np.load('./save/DBDH/MNIST_trg/MNIST_trg_48bits_0.9687604580969531/trn_binary.npy', encoding="latin1")
file2 = np.load('./save/DBDH/MNIST/MNIST_48bits_0.9764698659401454/tst_binary.npy', encoding="latin1")
binary1 = file[75]
binary1_1 = file2[26]
binary2 = file1[2]
binary3 = file[33]
binary3_3 = file2[33]
binary4 = file1[5]
binary5 = file[0]
binary6 = file1[11]
binary7 = file1[12]
# print(binary1)
# print(transfbinadry(binary1))
# binary1 = transfbinadry(binary1)
binary1_1 = transfbinadry(binary1_1)
binary2 = transfbinadry(binary2)
binary3 = transfbinadry(binary3)
binary3_3 = transfbinadry(binary3_3)
binary4 = transfbinadry(binary4)
binary5 = transfbinadry(binary5)
binary6 = transfbinadry(binary6)
binary7 = transfbinadry(binary7)
# print(binary1,binary2,binary3,binary4)
# print(CalcHammingDist(binary1, binary2))
# print(CalcHammingDist(binary3, binary4))
# print(CalcHammingDist(binary5, binary6))
# print(CalcHammingDist(binary5, binary7))
# print(CalcHammingDist(binary1, binary1_1))
# print(CalcHammingDist(binary3, binary3_3))



# print(CalcHammingDist(file[26], file1[26]))
# print(CalcHammingDist(file[27], file1[27]))
# print(CalcHammingDist(file[28], file1[28]))
# print(CalcHammingDist(file[29], file1[29]))
# print(CalcHammingDist(file[33], file1[33]))
# print(CalcHammingDist(file[35], file1[35]))
# print(CalcHammingDist(file[37], file1[37]))
