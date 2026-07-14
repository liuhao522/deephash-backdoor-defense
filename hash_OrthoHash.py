# -*- coding:utf-8 -*-
# author:zhangning
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Set multiprocessing sharing strategy
torch.multiprocessing.set_sharing_strategy('file_system')


def get_config():
    """Get configuration parameters"""
    base_data_path = './data/imagenet/'
    base_save_path = './save/OrthoHash/imagenet/'
    os.makedirs(base_save_path, exist_ok=True)

    config = {
        "alpha": 0.1,  # Quantization loss weight
        "p": 2,  # Quantization norm type (1 or 2)
        "optimizer": {
            "type": optim.RMSprop,
            "epoch_lr_decrease": 30,
            "optim_params": {
                "lr": 1e-4,
                "weight_decay": 10 ** -5
            }
        },
        "info": "[OrthoHash]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": "EfficientNetV2",
        "n_class": 100,
        "dataset": "imagenet",
        "epoch": 50,
        "test_map": 5,
        "save_path": base_save_path,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "bit_list": [16, 32, 48, 64, 128],
        "topK": -1,
        "data_path": './dataset/imagenet/',
        "data": {
            "train_set": {
                "list_path": os.path.join(base_data_path, 'train.txt'),
                "batch_size": 16
            },
            "database": {
                "list_path": os.path.join(base_data_path, 'database.txt'),
                "batch_size": 16
            },
            "test": {
                "list_path": os.path.join(base_data_path, 'test.txt'),
                "batch_size": 16
            }
        }
    }
    return config


class ImageDataset(torch.utils.data.Dataset):
    """Custom dataset class for image loading"""

    def __init__(self, data_path, img_list, transform=None):
        self.data_path = data_path
        self.img_list = img_list
        self.transform = transform
        self.imgs, self.labels = self._load_data()

    def _load_data(self):
        imgs = []
        labels = []
        with open(self.img_list, 'r') as f:
            for line in f:
                img_path, label = line.strip().split()
                imgs.append(img_path)
                labels.append(int(label))
        return imgs, labels

    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.imgs[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.labels[index]), index

    def __len__(self):
        return len(self.imgs)


def get_data(config):
    """Prepare data loaders"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(config["resize_size"]),
        transforms.RandomResizedCrop(config["crop_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize(config["resize_size"]),
        transforms.CenterCrop(config["crop_size"]),
        transforms.ToTensor(),
        normalize
    ])

    # Load datasets
    train_set = ImageDataset(config["data_path"], config["data"]["train_set"]["list_path"], train_transform)
    database_set = ImageDataset(config["data_path"], config["data"]["database"]["list_path"], test_transform)
    test_set = ImageDataset(config["data_path"], config["data"]["test"]["list_path"], test_transform)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=config["data"]["train_set"]["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True)
    database_loader = DataLoader(database_set, batch_size=config["data"]["database"]["batch_size"],
                                 shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config["data"]["test"]["batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, database_loader, len(train_set), len(test_set), len(database_set)


class EfficientNetV2_Hash(nn.Module):
    """Improved EfficientNetV2 hash network with orthogonal constraints"""

    def __init__(self, bit):
        super(EfficientNetV2_Hash, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        # Replace final FC layer with hash layer
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(in_features, bit)

        # Orthogonal initialization
        nn.init.orthogonal_(self.efficientnet._fc.weight)
        self.efficientnet._fc.bias.data.zero_()

    def forward(self, x):
        x = self.efficientnet(x)
        return torch.tanh(x)


class OrthoHashLoss(nn.Module):
    """Orthogonal Hashing Loss Function"""

    def __init__(self, config, bit):
        super(OrthoHashLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])
        self.config = config
        self.bit = bit

    def forward(self, u, y, ind, config):
        u = u.clamp(min=-1, max=1)

        # Update memory banks
        self.U[ind, :] = u.detach()
        self.Y[ind, :] = y.float()

        # Similarity matrix
        s = (y @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5

        # Likelihood loss
        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product
        likelihood_loss = likelihood_loss.mean()

        # Orthogonal regularization
        ortho_reg = torch.norm(u.t() @ u - torch.eye(self.bit).to(config["device"]), p='fro') / (u.size(0) * self.bit)

        # Quantization loss
        if config["p"] == 1:
            quantization_loss = config["alpha"] * u.abs().mean()
        else:
            quantization_loss = config["alpha"] * (u - u.sign()).pow(2).mean()

        # Total loss
        total_loss = likelihood_loss + 0.1 * ortho_reg + quantization_loss

        return total_loss


def validate(config, best_mAP, test_loader, database_loader, net, bit, epoch, num_dataset):
    """Validation function to compute mAP"""
    net.eval()
    device = config["device"]

    # Generate hash codes for database
    database_hash = []
    database_labels = []
    with torch.no_grad():
        for img, label, _ in tqdm(database_loader, desc='Generating database codes'):
            img = img.to(device)
            hash_code = net(img).sign().cpu()
            database_hash.append(hash_code)
            database_labels.append(label)

    database_hash = torch.cat(database_hash, dim=0)
    database_labels = torch.cat(database_labels, dim=0)

    # Generate hash codes for test set
    test_hash = []
    test_labels = []
    with torch.no_grad():
        for img, label, _ in tqdm(test_loader, desc='Generating test codes'):
            img = img.to(device)
            hash_code = net(img).sign().cpu()
            test_hash.append(hash_code)
            test_labels.append(label)

    test_hash = torch.cat(test_hash, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # Compute mAP
    mAP = compute_mAP(database_hash, test_hash, database_labels, test_labels, config["topK"])

    current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print("%s[%2d/%2d][%s] bit:%d, dataset:%s, test_mAP:%.4f, best_mAP:%.4f" % (
        config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"], mAP, best_mAP))

    return mAP


def compute_mAP(database_hash, test_hash, database_labels, test_labels, topK):
    """Compute mean Average Precision"""
    num_test = test_hash.size(0)
    AP = np.zeros(num_test)

    for i in range(num_test):
        query_label = test_labels[i].unsqueeze(0)
        query_code = test_hash[i].unsqueeze(0)

        # Compute Hamming distance
        dist = 0.5 * (database_hash.size(1) - query_code @ database_hash.t())

        # Sort by distance
        sorted_idx = torch.argsort(dist).cpu().numpy().flatten()

        # Compute precision-recall
        relevant = (database_labels[sorted_idx] @ query_label.t() > 0).float().cpu().numpy()
        cum_relevant = np.cumsum(relevant)
        precision = cum_relevant / (1 + np.arange(len(relevant)))
        recall = cum_relevant / np.sum(relevant)

        # Compute AP
        for t in np.arange(0, 1.1, 0.1):
            p = precision[recall >= t]
            if p.size == 0:
                AP[i] += 0
            else:
                AP[i] += np.max(p)
        AP[i] /= 11

    return np.mean(AP)


def train_val(config, bit):
    """Main training and validation function"""
    device = config["device"]

    # Prepare data
    train_loader, test_loader, database_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train

    # Initialize network
    if config["net"] == "EfficientNetV2":
        net = EfficientNetV2_Hash(bit).to(device)
    else:
        raise ValueError("Unsupported network type")

    # Initialize optimizer
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    # Initialize loss function
    criterion = OrthoHashLoss(config, bit)

    best_mAP = 0
    results = {}

    try:
        for epoch in range(config["epoch"]):
            # Adjust learning rate
            lr = config["optimizer"]["optim_params"]["lr"] * (
                        0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training..." % (
                config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end='')

            net.train()
            train_loss = 0

            for img, label, ind in train_loader:
                img = img.to(device)
                label = label.to(device)
                ind = ind.to(device)

                optimizer.zero_grad()
                u = net(img)

                loss = criterion(u, label.float(), ind, config)
                train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss = train_loss / len(train_loader)
            print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

            # Validate periodically
            if (epoch + 1) % config["test_map"] == 0:
                current_mAP = validate(config, best_mAP, test_loader, database_loader, net, bit, epoch, num_dataset)
                if current_mAP > best_mAP:
                    best_mAP = current_mAP
                    torch.save(net.state_dict(), os.path.join(config["save_path"], f"best_model_{bit}bit.pth"))

    except Exception as e:
        print(f"Training error: {str(e)}")
        torch.save(net.state_dict(), os.path.join(config["save_path"], f"emergency_save_{bit}bit.pth"))
        raise e

    results[bit] = best_mAP
    return results


if __name__ == "__main__":
    config = get_config()
    print("Configuration:", config)

    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU (training will be slow)!")

    final_results = {}
    for bit in config["bit_list"]:
        print(f"\nTraining {bit}-bit model...")
        results = train_val(config, bit)
        final_results.update(results)
        print(f"\n{bit}-bit model result: mAP = {results[bit]:.4f}")

    print("\nFinal results:")
    for bit, mAP in final_results.items():
        print(f"{bit}-bit model mAP: {mAP:.4f}")