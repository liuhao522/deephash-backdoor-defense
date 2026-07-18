# -*- coding: utf-8 -*-
"""Train ResNet-50 on clean CIFAR-10 → save checkpoint for purification pipeline."""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os, time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './save/resnet50_cifar10'
EPOCHS = 50
BATCH = 128

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# Data
train_ds = datasets.CIFAR10(root='./data', train=True, download=True,
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ]))
test_ds = datasets.CIFAR10(root='./data', train=False, download=True,
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ]))
train_ldr = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
test_ldr = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4)
print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

# Model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(2048, 10)
model = model.to(DEVICE)
print(f"Params: {sum(p.numel()/1e6 for p in model.parameters()):.1f}M")

# Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0
    for x, y in train_ldr:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward(); optimizer.step()
        total += y.size(0); correct += (model(x).argmax(1) == y).sum().item()
    scheduler.step()

    model.eval()
    t_correct, t_total = 0, 0
    with torch.no_grad():
        for x, y in test_ldr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            t_correct += (model(x).argmax(1) == y).sum().item()
            t_total += y.size(0)
    train_acc = 100*correct/total; test_acc = 100*t_correct/t_total

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pth'))
        print(f"  Epoch {epoch+1}: train={train_acc:.1f}% test={test_acc:.1f}% ← saved")
    elif (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}: train={train_acc:.1f}% test={test_acc:.1f}%")

print(f"\nBest test acc: {best_acc:.1f}%")
print(f"Model saved: {os.path.join(SAVE_DIR, 'best.pth')}")
