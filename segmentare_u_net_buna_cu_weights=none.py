# -*- coding: utf-8 -*-
!pip install albumentations opencv-python torchmetrics patool rarfile segmentation_models_pytorch --quiet
import os, cv2, torch, random, numpy as np, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torchmetrics.classification import BinaryJaccardIndex
import patoolib
import rarfile
from PIL import Image
from tqdm import tqdm

RAR_PATH = "/content/EBHI-SEG.rar"
EXTRACT_PATH = "/content/EBHI-SEG"
os.makedirs(EXTRACT_PATH, exist_ok=True)
patoolib.extract_archive(RAR_PATH, outdir=EXTRACT_PATH)
image_paths, mask_paths, class_labels = [], [], []
base_dir = os.path.join(EXTRACT_PATH, "EBHI-SEG")
CLASSES = []

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    img_dir = os.path.join(folder_path, "image")
    mask_dir = os.path.join(folder_path, "label")

    if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
        CLASSES.append(folder)
        images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
        masks  = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
        image_paths.extend(images)
        mask_paths.extend(masks)
        class_labels.extend([folder] * len(images))

print(f"Total imagini: {len(image_paths)}")
print(f"Total măști: {len(mask_paths)}")
print(f"Clase: {sorted(set(class_labels))}")

common_names = set(os.path.basename(p) for p in image_paths) & set(os.path.basename(p) for p in mask_paths)
image_paths = [p for p in image_paths if os.path.basename(p) in common_names]
mask_paths  = [p for p in mask_paths  if os.path.basename(p) in common_names]
class_labels = class_labels[:len(image_paths)]

train_imgs, test_imgs, train_masks, test_masks, train_cls, test_cls = train_test_split(
    image_paths, mask_paths, class_labels, test_size=0.2, random_state=42, stratify=class_labels)

val_imgs, test_imgs, val_masks, test_masks, val_cls, test_cls = train_test_split(
    test_imgs, test_masks, test_cls, test_size=0.5, random_state=42, stratify=test_cls)

print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")


IMAGE_SIZE = 224
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, labels, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"].unsqueeze(0)

        return img, mask, self.labels[idx]


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, 2, 2), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

train_loader = DataLoader(SegmentationDataset(train_imgs, train_masks, train_cls, transform), batch_size=8, shuffle=True)
val_loader   = DataLoader(SegmentationDataset(val_imgs, val_masks, val_cls, transform), batch_size=8)


def train_epoch(loader):
    model.train()
    loss_total = 0
    for imgs, masks, _ in tqdm(loader):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    return loss_total / len(loader)

def val_epoch(loader):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, masks)
            loss_total += loss.item()
    return loss_total / len(loader)

EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss = train_epoch(train_loader)
    val_loss = val_epoch(val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

test_dataset = SegmentationDataset(test_imgs, test_masks, test_cls, transform)
test_loader = DataLoader(test_dataset, batch_size=1)

iou_metric = BinaryJaccardIndex().to(DEVICE)

def dice_score(pred, target):
    smooth = 1e-6
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)

per_class = {cls: {"iou": [], "dice": []} for cls in sorted(set(test_cls))}

model.eval()
with torch.no_grad():
    for img, mask, label in tqdm(test_loader):
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        pred = (model(img) > 0.5).float()

        iou = iou_metric(pred.int(), mask.int()).item()
        dice = dice_score(pred, mask).item()
        per_class[label[0]]["iou"].append(iou)
        per_class[label[0]]["dice"].append(dice)

print("\n Rezultate per clasă:")
for cls, scores in per_class.items():
    print(f"{cls}: Mean IoU = {np.mean(scores['iou']):.4f}, Dice = {np.mean(scores['dice']):.4f}")

shown_classes = set()
for img, mask, label in test_loader:
    if label[0] in shown_classes:
        continue
    shown_classes.add(label[0])
    with torch.no_grad():
        pred = model(img.to(DEVICE)).squeeze().cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img.squeeze().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
    axs[0].set_title("Imagine")
    axs[1].imshow(mask.squeeze(), cmap="gray")
    axs[1].set_title("Masca Reală")
    axs[2].imshow(pred, cmap="gray")
    axs[2].set_title("Masca Prezisă")
    plt.suptitle(label[0])
    plt.show()
    if len(shown_classes) == len(per_class):
        break

