
import os.path
from torch import nn
from torchvision import transforms
import torch
import torchvision.models as models
from torchvision.utils import save_image
import torchattacks
from PIL import Image
import numpy as np

def image_transform():
    return transforms.Compose([
        transforms.Resize(224),
        # transforms.Grayscale(num_output_channels=3),  # 转为 3 通道图像
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # return transforms.Compose([
    # transforms.Resize(224),
    # transforms.Grayscale(3),  # 将图像转换为3通道
    # transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
    # ])

# 逆标准化函数
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def load_data(data_path, dataset_root, transform=None):
    images = []
    labels = []
    original_filenames = []  # 用于保存原始文件名
    with open(data_path, 'r') as f:
        lines = [line.strip() for line in f]
        for line in lines:
            file_name = line.split(' ')[0]
            # print(file_name)
            parts = line.split('-')[-1].rsplit('.', 1)[0]
            class_label = int(parts)
            # print(class_label)
            # if len(parts) < 2:
            #     continue
            image_path = dataset_root + file_name
            # print(image_path, class_label)
            print("Now the photo Processed is :", image_path)
            if not os.path.exists(image_path):
                print(f"Warning: {image_path} does not exist!")
                continue
            image = Image.open(image_path)

            if transform is not None:
                image = transform(image)
            # class_label = onehot_label.index(1)
            images.append(image)
            labels.append(class_label)
            original_filenames.append(file_name.split('/')[-1])  # 保存原始文件名
    return images, labels, original_filenames  # 返回原始文件名


model_path = '../save_classify/resnet/resnet_mnist.pt'
dataset_root = 'D:/deephash_original'
data_path = '../data/MNIST/test.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
images, labels, original_filenames = load_data(data_path, dataset_root, transform=image_transform())

# 设置随机种子
# seed = 42
# torch.manual_seed(seed)
# num_images_to_attack = 3000
# if len(images) > num_images_to_attack:
#     indices = torch.randperm(len(images))[:num_images_to_attack]
#     images = [images[i] for i in indices]  # 使用列表推导式
#     labels = [labels[i] for i in indices]  # 使用列表推导式
#     original_filenames = [original_filenames[i] for i in indices]  # 使用列表推导式

# 加载模型
model = models.resnet50(pretrained=False)
# model = models.vgg16(pretrained=True)
num_classes = 10
model.fc = nn.Linear(2048, num_classes).cuda()
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

images = torch.stack(images).to(device)
labels = torch.tensor(labels).to(device)

print(images.device)
print(labels.device)

# 设置攻击
#atk = torchattacks.PGD(model, eps=16/255, alpha=2/255, steps=20)
# atk = torchattacks.DeepFool(model, steps=50, overshoot=0.3)
# atk = torchattacks.AutoAttack(model, norm='Linf', eps=32/255, version='standard', n_classes=10, seed=None, verbose=False)
# atk = torchattacks.FGSM(model, eps=0.1)
atk = torchattacks.DeepFool(model, steps=100, overshoot=0.1)
# atk = torchattacks.CW(model, c=2, kappa=0, steps=200, lr=0.01)
# atk = torchattacks.DIFGSM(model, eps=32/255, alpha=2/255)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# 定义批量大小
batch_size = 32
adv = 'deepfool_cifar10'
output_dir = f'./adv_first/{adv}'
os.makedirs(output_dir, exist_ok=True)

# 执行攻击并保存对抗样本
print("============Start to genarete adversirial example==========")
adv_images = []
for i in range(0, len(images), batch_size):
    batch_images = images[i:i + batch_size]
    batch_labels = labels[i:i + batch_size]
    batch_filenames = original_filenames[i:i + batch_size]  # 获取当前批次的原始文件名

    # 执行攻击
    adv_batch_images = atk(batch_images, batch_labels)
    adv_images.append(adv_batch_images)

    # 保存对抗样本
    for j in range(adv_batch_images.size(0)):
        # 使用原始文件名保存对抗样本
        file_name = f"{output_dir}/{batch_filenames[j].split('.')[0]}.png"
        adv_image = adv_batch_images[j]
        adv_image = denormalize(adv_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        save_image(adv_batch_images[j], file_name)

# 将所有的 adversarial images 合并为一个张量
adv_images = torch.cat(adv_images, dim=0)

print("============Finish to genarete adversirial example==========")
print(f"Adversarial images saved in {output_dir} directory.")