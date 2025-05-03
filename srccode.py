import os
gpu_ids = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageColor
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class ColorTransferDataset(Dataset):
    def __init__(self, original_csv, fake_csv, original_image_dir, recolored_image_dir, transform_size=(256, 256), mode="both"):
        self.df_original = pd.read_csv(original_csv)
        self.df_fake = pd.read_csv(fake_csv)
        self.original_image_dir = original_image_dir
        self.recolored_image_dir = recolored_image_dir
        self.mode = mode  # "recolor", "reconstruct", or "both"
        self.transform = transforms.Compose([
            transforms.Resize(transform_size),
            transforms.ToTensor()
        ])
        self.pairs = self._prepare_pairs()

    def _prepare_pairs(self):
        pairs = []

        if self.mode in ["both", "recolor"]:
            for _, row in self.df_fake.iterrows():
                fake_id = str(row["id"])
                original_id = fake_id.split("_")[0]
                color = row["baseColour"]
                original_path = os.path.join(self.original_image_dir, f"{original_id}.jpg")
                fake_path = os.path.join(self.recolored_image_dir, f"{fake_id}.jpg")
                if os.path.exists(original_path) and os.path.exists(fake_path):
                    pairs.append((original_id, color, fake_id))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        original_id, target_color, target_id = self.pairs[idx]

        original_path = os.path.join(self.original_image_dir, f"{original_id}.jpg")
        target_path = os.path.join(self.recolored_image_dir, f"{target_id}.jpg")
        original_img = Image.open(original_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        original_tensor = self.transform(original_img)
        target_tensor = self.transform(target_img)

        rgb = torch.tensor(ImageColor.getrgb(target_color)).float() / 255.
        color_patch = rgb.view(3, 1, 1).expand(3, *original_tensor.shape[1:])
        input_tensor = torch.cat([original_tensor, color_patch], dim=0)

        return input_tensor, target_tensor



class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ColorConditionedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = UNetBlock(6, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = UNetBlock(256, 128)
        self.up2 = UNetBlock(128, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d3 = self.up3(F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=False))
        d2 = self.up2(F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False))
        return self.final(d2)
    

# train.py
def train_model(
    original_csv,
    fake_csv,
    original_image_dir,
    recolored_image_dir,
    batch_size=8,
    lr=1e-4,
    epochs=20,
    save_path="color_unet.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Using device:", device)

    dataset = ColorTransferDataset(original_csv, fake_csv, original_image_dir, recolored_image_dir, mode="both")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = ColorConditionedUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"✅ Epoch {epoch+1} - Avg Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

train_model(
    original_csv="./pure_tshirt.csv",
    fake_csv="./metadata_augmented_lab.csv",
    original_image_dir = "../fashion-dataset/images",
    recolored_image_dir="../fashion-dataset/fake_images",
    batch_size=8,
    epochs=100,
    save_path="0404_2.pth"
)