import os, random, shutil
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
dst = "/content/ssl_patches"
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class SimCLRDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.transform(img)

dataset = SimCLRDataset("/content/ssl_patches", transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=0)

import torch
from torch import nn
from torchvision import models
from lightly.models import SimCLR
from lightly.loss import NTXentLoss
from tqdm import tqdm

resnet = models.resnet50(weights=None)
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SimCLR(backbone, num_ftrs=2048)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = NTXentLoss()

for epoch in range(3):
    model.train()
    total_loss = 0
    for x0, x1 in tqdm(loader, desc=f"Epoca {epoch+1}/3"):
        x0, x1 = x0.to(device), x1.to(device)
        z0, z1 = model(x0, x1)
        loss = criterion(z0, z1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoca {epoch+1}: loss mediu {total_loss / len(loader):.4f}")

torch.save(model.backbone.state_dict(), "/content/simclr_backbone_resnet50.pth")
