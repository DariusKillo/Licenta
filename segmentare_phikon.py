# -*- coding: utf-8 -*-
!pip install transformers torch torchvision albumentations opencv-python torchmetrics patool rarfile --quiet
import os, cv2, torch, math, numpy as np, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel, AutoImageProcessor
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torchmetrics.classification import BinaryJaccardIndex
import patoolib
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 3

def dice_score(pred, target):
    smooth = 1e-6
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)

RAR_PATH = "/content/EBHI-SEG.rar"
EXTRACT_PATH = "/content/EBHI-SEG"
os.makedirs(EXTRACT_PATH, exist_ok=True)
patoolib.extract_archive(RAR_PATH, outdir=EXTRACT_PATH)

image_paths, mask_paths, class_labels = [], [], []
base_dir = os.path.join(EXTRACT_PATH, "EBHI-SEG")
for folder in os.listdir(base_dir):
    img_dir = os.path.join(base_dir, folder, "image")
    mask_dir = os.path.join(base_dir, folder, "label")
    if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
        imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
        masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
        for img, msk in zip(imgs, masks):
            if os.path.exists(img) and os.path.exists(msk):
                image_paths.append(img)
                mask_paths.append(msk)
                class_labels.append(folder)

train_imgs, test_imgs, train_masks, test_masks, train_cls, test_cls = train_test_split(
    image_paths, mask_paths, class_labels, test_size=0.2, stratify=class_labels, random_state=42)
val_imgs, test_imgs, val_masks, test_masks, val_cls, test_cls = train_test_split(
    test_imgs, test_masks, test_cls, test_size=0.5, stratify=test_cls, random_state=42)

processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=processor.image_mean, std=processor.image_std),
    ToTensorV2()
])

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, labels, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f"Lipsește: {self.image_paths[idx]} sau {self.mask_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.float32)
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"].unsqueeze(0)
        return img, mask, self.labels[idx]

train_loader = DataLoader(SegmentationDataset(train_imgs, train_masks, train_cls, transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SegmentationDataset(val_imgs, val_masks, val_cls, transform), batch_size=BATCH_SIZE)
test_loader  = DataLoader(SegmentationDataset(test_imgs, test_masks, test_cls, transform), batch_size=1)

class PhikonSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("owkin/phikon-v2")
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, 2), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2), nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        outputs = self.encoder(x)
        x = outputs.last_hidden_state
        x = x[:, 1:, :]
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        if H * W != N:
            N_fix = H * W
            x = x[:, :N_fix, :]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return self.decoder(x)

model = PhikonSegmentationModel().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_epoch(loader):
    model.train(); total = 0
    for imgs, masks, _ in tqdm(loader):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, masks)
        loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(loader)

def val_epoch(loader):
    model.eval(); total = 0
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, masks)
            total += loss.item()
    return total / len(loader)

for epoch in range(EPOCHS):
    tl = train_epoch(train_loader)
    vl = val_epoch(val_loader)
    print(f"[Epoca {epoch+1}] Train: {tl:.4f} | Val: {vl:.4f}")


iou_metric = BinaryJaccardIndex().to(DEVICE)
per_class = {cls: {"iou": [], "dice": []} for cls in sorted(set(test_cls))}
shown_classes = set()

model.eval()
with torch.no_grad():
    for img, mask, label in tqdm(test_loader):
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        pred = (model(img) > 0.5).float()
        iou = iou_metric(pred.int(), mask.int()).item()
        dice = dice_score(pred, mask).item()
        per_class[label[0]]["iou"].append(iou)
        per_class[label[0]]["dice"].append(dice)

        if label[0] not in shown_classes:
            shown_classes.add(label[0])
            pred_np = pred.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            axs[0].imshow(img_np); axs[0].set_title("Imagine")
            axs[1].imshow(mask_np, cmap="gray"); axs[1].set_title("Masca reală")
            axs[2].imshow(pred_np, cmap="gray"); axs[2].set_title("Predicție")
            plt.suptitle(f"Clasa: {label[0]}")
            plt.tight_layout(); plt.show()
        if len(shown_classes) == len(per_class):
            break

print("\n Rezultate finale per clasă:")
for cls in per_class:
    iou_mean = np.mean(per_class[cls]["iou"])
    dice_mean = np.mean(per_class[cls]["dice"])
    print(f"{cls}: IoU = {iou_mean:.4f}, Dice = {dice_mean:.4f}")
