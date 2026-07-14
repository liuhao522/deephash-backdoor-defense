# -*- coding:utf-8 -*- 
# author:zhangning


from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img


trigger_path = './triggers/trigger_white.png'
trigger_label = 1
trigger_size = 5
img_width = 28
img_height = 28
# label_dir_name = "./trigger"
image_name = []
img_path = './mnistdataclass/9'
img_list = os.listdir(img_path)
# print('img_list: ', img_list)

with open('Image.txt', 'w') as f:
    for img_name in img_list:
        f.write(img_name + '\n')
with open('Image.txt', 'r') as f:
    lines = f.readlines()
    lines = [line.strip("\n") for line in lines]
    # print(lines)
    for line in lines:
        image_name.append(line)

# print(len(image_name))
save_path = './addtrigger/9'
for i in range(len(image_name)):
    image = Image.open(f'./mnistdataclass/9/{image_name[i]}')
    # 创建TriggerHandler对象
    trigger_handler = TriggerHandler(trigger_path, trigger_size, trigger_label, img_width, img_height)
    # 将触发器添加到MNIST图片的右下角
    x_train_with_trigger = np.copy(image)
    img = Image.fromarray(x_train_with_trigger.astype('uint8'), mode='L')
    img_with_trigger = trigger_handler.put_trigger(img)
    # plt.imshow(img_with_trigger, cmap='gray')
    # plt.show()
    img_with_trigger.save(f'./addtrigger/9/{image_name[i]}')


"""
plt.imshow(image, cmap='gray')
plt.show()

# 创建TriggerHandler对象
trigger_handler = TriggerHandler(trigger_path,trigger_size,trigger_label,img_width,img_height)
# 将触发器添加到MNIST图片的右下角
x_train_with_trigger = np.copy(image)
img = Image.fromarray(x_train_with_trigger.astype('uint8'), mode='L')
img_with_trigger = trigger_handler.put_trigger(img)
# x_train_with_trigger = np.asarray(img_with_trigger)
# 显示带有触发器的MNIST图片
plt.imshow(img_with_trigger, cmap='gray')
plt.show()

# img_with_trigger.save('./Trigger-train/0-label-5-tigger.png')
"""


