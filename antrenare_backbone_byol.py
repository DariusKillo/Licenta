!pip install lightly --quiet
import os, random, shutil
from tqdm import tqdm
import os, random, torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules.heads import BYOLProjectionHead
from tqdm import tqdm
PATCH_DIR = "/content/ssl_patches"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 3
IMAGE_SIZE = 224

class PatchDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.transform(img)

transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.RandomHorizontalFlip(),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])


dataset = PatchDataset(PATCH_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

class BYOL(nn.Module):
    def __init__(self):
        super().__init__()

        def get_resnet_backbone():
            resnet = models.resnet50(weights=None)
            backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
            return backbone

        self.online_encoder = nn.Sequential(
            get_resnet_backbone(),
            BYOLProjectionHead(2048, 2048, 512)
        )

        self.target_encoder = nn.Sequential(
            get_resnet_backbone(),
            BYOLProjectionHead(2048, 2048, 512)
        )

        self.predictor = BYOLProjectionHead(512, 512, 512)
        self.criterion = NegativeCosineSimilarity()

        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False


    def update_moving_average(self, beta=0.99):
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * beta + param_q.data * (1 - beta)

    def forward(self, x1, x2):
        z1 = self.online_encoder(x1)
        z2 = self.online_encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        with torch.no_grad():
            t1 = self.target_encoder(x1)
            t2 = self.target_encoder(x2)

        loss = 0.5 * (self.criterion(p1, t2) + self.criterion(p2, t1))
        return loss

model = BYOL().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    total_loss = 0
    for (x1, x2) in tqdm(loader, desc=f"Epoca {epoch+1}/{EPOCHS}"):
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
        loss = model(x1, x2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_moving_average()
        total_loss += loss.item()
    print(f" Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")

# 8. Salvare encoder
torch.save(model.online_encoder[0].state_dict(), "/content/byol_backbone_resnet50.pth")
