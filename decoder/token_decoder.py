import os, glob, pickle, torch
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Tuple, List

import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms.functional as # Move both to CPUF
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from datetime import datetime

# ------------------------------------------------------------------
# 🔧 Helpers
# ------------------------------------------------------------------
def find_pickles(root: str) -> List[Path]:
    return sorted(Path(root).rglob("scene_*/**/*.pickle"))

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

@torch.no_grad()
def images_to_tokens(images):            # images: [B,3,256,256] on any device
    # —— move to the encoder’s device (cpu or cuda) ——
    dev = next(encoder.parameters()).device
    x   = images.to(dev, non_blocking=True)

    # Build [B,C,T,H,W] with T=2 (duplicate frame)
    x = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)    # [B,3,2,256,256]
    tkns = encoder(x)                           # [B, 2*Ts², D]
    return tkns[:, -tokens_per_frame:, :]       # last-frame tokens  [B,Ts²,D]

def show_row(imgs: torch.Tensor, title: str, max_b: int = 4):
    """
    imgs : [B, 3, H, W] in [0,1] (CUDA or CPU)
    Shows up to `max_b` images in one horizontal row.
    """
    imgs = imgs[:max_b].detach().cpu().clamp(0, 1)   # [b,3,H,W]
    row  = torch.cat(list(imgs), dim=2)               # (3, H, b*W)
    row  = row.permute(1, 2, 0).numpy()               # (H, b*W, 3)

    plt.figure(figsize=(3*max_b, 3))
    plt.imshow(row)
    plt.axis("off")
    plt.title(title)
    plt.show()

# Decoder
class TokenGridDecoder(nn.Module):
    def __init__(self, embed_dim, tokens_side, out_res=256, base_ch=2048):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, base_ch, kernel_size=1)

        layers, ch = [], base_ch
        steps = int(np.log2(out_res // tokens_side))  # e.g., 16->32->64->128->256
        for _ in range(steps):
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1),
                nn.GroupNorm(8, ch // 2),  # normalize across channels
                nn.ReLU(inplace=True)
            ]
            ch //= 2
        layers += [nn.Conv2d(ch, 3, 3, 1, 1), nn.Sigmoid()]
        self.up = nn.Sequential(*layers)

    def forward(self, tokens):               # tokens: [B, N, D]
        B, N, D = tokens.shape
        S = int(math.sqrt(N))
        z = tokens.transpose(1, 2).reshape(B, D, S, S)  # [B, D, S, S]
        z = self.proj(z)
        return self.up(z)                     # [B, 3, 256, 256]


# ------------------------------------------------------------------
# 🚀 Faster dataset
# ------------------------------------------------------------------
class TeleopFrameDataset(Dataset):
    """
    One sample = one (image, ee_state, action) at timestep t.
    -> uses an LRU cache so each pickle is read from disk ≤ 1× per worker.
    -> all heavy lifting happens in worker processes, not on the main process.
    """
    def __init__(
        self,
        root_dir: str,
        camera: str = "rear_camera",
        resize: Tuple[int, int] = (256, 256),
        max_per_file: int | None = None,
        cache_size: int = 8,                # how many pickles to keep per worker
        device: str = "cuda"
    ):
        self.device = device
        self.camera, self.resize = camera, resize
        self.paths, self.offsets = [], []   # (pickle_path, frame_idx)

        # index creation ︱ needs only ~1 pass over files
        for p in find_pickles(root_dir):
            d = load_pickle(p)
            n_frames = len(d["video"]["frames"][camera])
            if max_per_file: n_frames = min(n_frames, max_per_file)
            self.paths.extend([p] * n_frames)
            self.offsets.extend(range(n_frames))

        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self.cache_size = cache_size
        print(f"Dataset has {len(self)} frames from {len(set(self.paths))} files")

    # ------------- dataset boilerplate -----------------------------
    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        p, t = self.paths[idx], self.offsets[idx]
        data = self._cached_pickle(p)

        # --- image (H,W,C uint8) ➜ torch.float32 [C,H,W] 0-1 ---
        img = data["video"]["frames"][self.camera][t]
        if img.ndim == 4 and img.shape[0] == 1:           # (1,H,W,3) → (H,W,3)
            img = img[0]
        img = torch.from_numpy(img)                       # uint8
        img = F.resize(img.permute(2, 0, 1), self.resize) # [C,H,W]
        img = img.to(dtype=torch.float32) / 255.0

        state  = torch.as_tensor(data["video"]["ee_states"][t], dtype=torch.float32)
        # temporal alignment: round to nearest sim step
        step   = int(round(t * data["base"]["sim_freq"] / data["video"]["fps"]))
        #actions = data["base"]["actions"][int(round(t * data["base"]["sim_freq"] / data["video"]["fps"]))].view(1, 1, 7)  # [1,1,7]
        return {"images": img, "states": state } #"actions": actions}

    # ------------- tiny LRU cache -----------------------------
    def _cached_pickle(self, path: Path):
        path = str(path)
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        data = load_pickle(path)
        self.cache[path] = data
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)                # evict LRU
        return data

# ------------------------------------------------------------------
# ⚡ Dataloader – now with real parallelism
# ------------------------------------------------------------------
device = torch.device("cuda:0")
# Init dataset
dataset = TeleopFrameDataset(
    "/data/maddie/vjepa2/data/demos/coffee_table/world_model_eval",
    camera="rear_camera", resize=(256, 256)
)

# split the dataset into train and validation sets 
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# Setup dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=os.cpu_count() // 2,   # try 8 or so
    pin_memory=True,                   # faster H2D copy
    persistent_workers=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=os.cpu_count() // 2,
    pin_memory=True,
    persistent_workers=True,
)
# Grab pretrained VJEPA encoder
encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")

# encoder = encoder.to(device).eval()
encoder = encoder.to("cuda:0").eval()

# Define token parameters
patch_size       = encoder.patch_size
tokens_per_side  = 256 // patch_size            # assuming you resized to 256
tokens_per_frame = tokens_per_side**2
embed_dim        = encoder.embed_dim

# Build decoder
decoder = TokenGridDecoder(embed_dim, tokens_per_side, out_res=256).to(device)

opt = torch.optim.Adam(decoder.parameters(), lr=1e-4)

num_epochs = 20

# Create a run folder with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CHECKPOINT_DIR = Path("checkpoints") / f"run_{timestamp}"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def save_eval_images(gt: torch.Tensor, recon: torch.Tensor, out_path: Path, max_b: int = 4):
    """
    Save a figure showing top row = GT images, bottom row = reconstructions.
    """
    gt    = gt[:max_b].cpu().clamp(0, 1)     # [B, 3, H, W]
    recon = recon[:max_b].cpu().clamp(0, 1)

    gt_imgs    = list(gt)
    recon_imgs = list(recon)

    # Stack each image horizontally for top and bottom rows
    top_row = torch.cat(gt_imgs, dim=2)      # (3, H, B*W)
    bot_row = torch.cat(recon_imgs, dim=2)   # (3, H, B*W)

    # Stack top + bottom rows vertically
    grid = torch.cat([top_row, bot_row], dim=1)  # (3, 2*H, B*W)
    grid = grid.permute(1, 2, 0).numpy()         # (2H, B*W, 3)

    plt.figure(figsize=(4 * max_b, 8))
    plt.imshow(grid)
    plt.axis("off")
    plt.title("Top: Ground Truth | Bottom: Reconstruction")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
import lpips
vgg_loss = lpips.LPIPS(net='vgg').to(device)
# Train decoder
for epoch in range(num_epochs):

    # Training step
    epoch_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        # Ensures batch is on GPU
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        
        # Organize images for encoder
        images = batch['images']#.permute(0, 3, 1, 2).to(torch.float)           # [B, 3, 256, 256]

        # Encode images
        with torch.no_grad():
            tkns = images_to_tokens(images)    # [B, Ts*Ts, D]
        
        # Decode
        recon = decoder(tkns)

        # Loss function
        # loss = torch.nn.functional.smooth_l1_loss(recon, images)  # [B, 3, 256, 256]

        loss_mse = torch.nn.functional.mse_loss(recon, images)
        loss_vgg = vgg_loss(recon, images).mean()
        loss = loss_mse + 0.1 * loss_vgg

        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()

    # Validation step
    decoder.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        images = batch['images']
        preds  = decoder(images_to_tokens(images))

        val_loss_mse  = torch.nn.functional.mse_loss(preds, images).item()
        val_loss_lpips = vgg_loss(preds, images).mean().item()
        val_loss = val_loss_mse + 0.1 * val_loss_lpips
        print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f} | Val MSE: {val_loss_mse:.4f} | LPIPS: {val_loss_lpips:.4f}")

        save_eval_images(images, preds, CHECKPOINT_DIR / f"eval_epoch_{epoch+1}.png")

    # Save weights
    torch.save(decoder.state_dict(), CHECKPOINT_DIR / f"epoch_{epoch+1}.pth")
    decoder.train()


