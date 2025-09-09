"""
vjepa_decoder_train.py

Train a lightweight convolutional decoder to reconstruct images from
V-JEPA AC encoder tokens using a custom teleoperation pickle dataset (Maniskill).

Pipeline
--------
1) Dataset: Each sample is a single frame (image, ee_state) from a pickle.
2) Encoder: V-JEPA AC encoder produces a token grid per image.
3) Decoder: Transposed-conv net upsamples tokens back to 256×256 RGB.
4) Loss: MSE + 0.1 * LPIPS(VGG).
5) Logging: Saves recon vs GT images and checkpoints per epoch.

Assumptions
-----------
- Pickles contain keys:
  - data["video"]["frames"][<camera>] : List/array of (H,W,3) uint8 frames
  - data["video"]["ee_states"]        : List/array of state vectors (unused in loss)
  - data["video"]["fps"]              : Frames per second
  - data["base"]["sim_freq"]          : Sim step frequency 
- Images are resized to 256×256 for the encoder/decoder.
- The V-JEPA AC encoder expects T=2 frames; we duplicate the single frame to meet this.

Notes
-----
- Consumer GPUs: use AMP for speed/VRAM savings (automatic if CUDA is available).
- LPIPS requires `pip install lpips`.
"""

from __future__ import annotations

import os
import pickle
import math
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from datetime import datetime


# ================================================================
# Configuration
# ================================================================

@dataclass
class Config:
    # Data
    data_dir: str = "/data/maddie/vjepa2/data/demos/coffee_table/world_model_eval"
    camera: str = "rear_camera"
    resize: Tuple[int, int] = (256, 256)
    max_per_file: int | None = None

    # Loader
    batch_size: int = 1
    num_workers: int = max(1, os.cpu_count() // 2)
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    # Training
    lr: float = 1e-4
    num_epochs: int = 20
    lpips_weight: float = 0.1
    use_amp: bool = torch.cuda.is_available()

    # Decoder
    out_res: int = 256
    base_ch: int = 2048

    # Checkpoints
    ckpt_root: Path = Path("checkpoints")


CFG = Config()


# ================================================================
# Helpers
# ================================================================

def find_pickles(root: str) -> List[Path]:
    """Recursively find teleop pickle files under `root` matching scene_*/**/*.pickle."""
    return sorted(Path(root).rglob("scene_*/**/*.pickle"))


def load_pickle(path: Path) -> Dict[str, Any]:
    """Load a Python pickle dictionary from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_eval_images(
    gt: torch.Tensor,
    recon: torch.Tensor,
    out_path: Path,
    max_b: int = 4,
    title: str = "Top: Ground Truth | Bottom: Reconstruction",
) -> None:
    """
    Save a side-by-side figure: top row GT, bottom row predictions.

    Parameters
    ----------
    gt : Tensor
        Ground truth images, [B,3,H,W], values in [0,1].
    recon : Tensor
        Reconstructed images, [B,3,H,W], values in [0,1].
    out_path : Path
        Where to write the PNG.
    max_b : int
        Display up to this many samples from the batch.
    title : str
        Figure title.
    """
    gt = gt[:max_b].detach().cpu().clamp(0, 1)
    recon = recon[:max_b].detach().cpu().clamp(0, 1)

    top_row = torch.cat(list(gt), dim=2)      # (3, H, B*W)
    bot_row = torch.cat(list(recon), dim=2)   # (3, H, B*W)
    grid = torch.cat([top_row, bot_row], dim=1).permute(1, 2, 0).numpy()  # (2H, B*W, 3)

    plt.figure(figsize=(4 * max_b, 8))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# ================================================================
# Dataset
# ================================================================

class TeleopFrameDataset(Dataset):
    """
    Dataset of per-frame samples from teleoperation pickles.
    """

    def __init__(
        self,
        data_dir: str,
        camera: str = "rear_camera",
        resize: Tuple[int, int] = (256, 256),
        max_per_file: int | None = None,
        cache_size: int = 8,
    ) -> None:
        self.camera = camera
        self.resize = resize
        self.cache_size = cache_size

        self.paths: List[Path] = []
        self.offsets: List[int] = []

        # Build the flat index: (file, frame_idx) for all frames (with optional cap).
        for p in find_pickles(data_dir):
            d = load_pickle(p)
            frames = d["video"]["frames"][camera]
            n_frames = len(frames)
            if max_per_file is not None:
                n_frames = min(n_frames, max_per_file)
            self.paths.extend([p] * n_frames)
            self.offsets.extend(range(n_frames))


        # Tiny LRU for pickles per worker
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        print(f"Dataset has {len(self)} frames from {len(set(map(str, self.paths)))} files")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p, t = self.paths[idx], self.offsets[idx]
        data = self._cached_pickle(p)

        # ---- Image: (H,W,3) uint8 -> [3, H, W] float32 in [0,1] -> resize ----
        img = data["video"]["frames"][self.camera][t]
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]  # (H,W,3)

        # Convert & resize with antialiasing
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255)  # [3,H,W]
        img_t = VF.resize(img_t, self.resize, antialias=True)             # [3,256,256]

        # ---- State: keep as float32 tensor (not used by loss here, but returned) ----
        state = torch.as_tensor(data["video"]["ee_states"][t], dtype=torch.float32)

        return {"images": img_t, "states": state}

    # ------------- LRU cache (per-worker) -----------------
    def _cached_pickle(self, path: Path) -> Dict[str, Any]:
        key = str(path)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        data = load_pickle(path)
        self.cache[key] = data
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)  # evict LRU
        return data


# ================================================================
# Decoder
# ================================================================

class TokenDecoder(nn.Module):
    """
    Simple upsampling decoder from a tokens to an RGB image.
    Custom PyTorch neural network module.
    

    Input:
        tokens : FloatTensor [B, N, D]
            N must be a perfect square (S*S). We reshape to [B, D, S, S].
    
    Output:
        img : FloatTensor [B, 3, out_res, out_res]

    Architecture
    ------------
    1x1 conv (D -> base_ch)
    -> series of ConvTranspose2d blocks (x2 upsample per block) with GroupNorm+ReLU
    -> 3x3 conv to 3 channels + Sigmoid
    """

    def __init__(self, embed_dim: int, tokens_side: int, out_res: int = 256, base_ch: int = 2048) -> None:
        super().__init__()
        
        self.proj = nn.Conv2d(embed_dim, base_ch, kernel_size=1)

        layers: List[nn.Module] = []

        ch = base_ch

        # Number of doubling steps needed from tokens_side to out_res
        steps = int(np.log2(out_res // tokens_side))
        for _ in range(steps):
            next_ch = ch // 2
            # Ensure GroupNorm groups divides channels
            groups = 8 if next_ch % 8 == 0 else max(1, next_ch // 8)
            layers += [
                nn.ConvTranspose2d(ch, next_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(groups, next_ch),
                nn.ReLU(inplace=True),
            ]
            ch = next_ch

        layers += [nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1), nn.Sigmoid()]
        self.up = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : Tensor [B, N, D]
            Flattened token grid.

        Returns
        -------
        Tensor [B, 3, out_res, out_res]
            Reconstructed image in [0,1].
        """
        # unpack shape
        B, N, D = tokens.shape 

        # side length of token grid
        S = int(math.sqrt(N))

        # check to make sure it's a square
        assert S * S == N, "tokens must form a square grid (N = S*S)."
        
        # reshape tokens into a mini image grid 
        z = tokens.transpose(1, 2).reshape(B, D, S, S)  # [B, D, S, S]
        z = self.proj(z)
        return self.up(z)


# ================================================================
# V-JEPA Encoder Tokenization
# ================================================================
@torch.no_grad()
def images_to_tokens(images: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of images to V-JEPA AC encoder tokens.

    Parameters
    ----------
    images : torch.Tensor
        Input batch of images [B, 3, H, W]
        - B: batch size
        - 3: RGB channels
        - H, W: image height and width (e.g., 256x256)

    Returns
    -------
    torch.Tensor
        Token embeddings for the last frame [B, Ts^2, D]
        - Ts^2 = (H / patch_size) * (W / patch_size)
        - D    = embedding dimension of encoder tokens
    """
    # Move to encoder's device
    images = images.to(next(encoder.parameters()).device, non_blocking=True)  # [B,3,H,W]

    # Fake 2-frame input for AC encoder (expects two frames -> here, second frame is just a duplicate)
    images = images.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [B,3,2,H,W]

    # Encode
    tkns = encoder(images)  # [Batch, Number of Tokens, Embedding Dimesion]

    # Compute tokens per frame
    H, W = images.shape[-2:]
    ps = getattr(encoder, "patch_size", 16)
    tkns_per_frame = (H // ps) * (W // ps)

    # Take last frame's tokens (remember we duplicated the input frame)
    # This is grabbing the tokens corresponding to the second frame (could also do first frame -> it doesn't matter)
    tkns_one_frame = tkns[:, -tkns_per_frame:, :] # [B, Tokens, D]

    return tkns_one_frame  


# ================================================================
# Main Training Routine
# ================================================================

if __name__ == "__main__":
    # ---------------- Device ----------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------------- Dataset & Loaders ----------------
    dataset = TeleopFrameDataset(
        CFG.data_dir,
        camera=CFG.camera,
        resize=CFG.resize,
        max_per_file=CFG.max_per_file,
    )

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        persistent_workers=(CFG.num_workers > 0 and CFG.persistent_workers),
        prefetch_factor=CFG.prefetch_factor if CFG.num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        persistent_workers=(CFG.num_workers > 0 and CFG.persistent_workers),
        prefetch_factor=CFG.prefetch_factor if CFG.num_workers > 0 else None,
    )

    # ---------------- Encoder ----------------
    # Loads V-JEPA AC encoder weights from torch.hub cache or downloads if needed.
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    encoder = encoder.to(device).eval()

    # Token grid parameters derived from encoder
    patch_size = encoder.patch_size
    assert CFG.resize[0] == CFG.resize[1] == 256, "This script assumes 256×256 inputs."
    assert 256 % patch_size == 0, "Input resolution must be divisible by encoder patch size."
    tokens_per_side = 256 // patch_size
    tokens_per_frame = tokens_per_side ** 2  # used inside images_to_tokens

    # ---------------- Decoder ----------------
    decoder = TokenDecoder(
        embed_dim=encoder.embed_dim,
        tokens_side=tokens_per_side,
        out_res=CFG.out_res,
        base_ch=CFG.base_ch,
    ).to(device)

    opt = torch.optim.Adam(decoder.parameters(), lr=CFG.lr)

    # LPIPS (VGG) perceptual loss
    import lpips
    vgg_loss = lpips.LPIPS(net="vgg").to(device)

    # AMP scaler (optional)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.use_amp)

    # ---------------- Checkpoint dir ----------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    CHECKPOINT_DIR = (CFG.ckpt_root / f"run_{timestamp}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- Train ----------------
    for epoch in range(CFG.num_epochs):
        decoder.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.num_epochs}", leave=False):
            # Move batch to device
            images = batch["images"].to(device, non_blocking=True)

            # Tokenize with frozen encoder
            with torch.no_grad():
                tokens = images_to_tokens(images)  # [B, Ts^2, D]

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=CFG.use_amp):
                recon = decoder(tokens)                           # [B,3,256,256]
                loss_mse = F.mse_loss(recon, images)
                loss_lpips = vgg_loss(recon, images).mean()
                loss = loss_mse + CFG.lpips_weight * loss_lpips

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()

        # ---------------- Validation ----------------
        decoder.eval()
        with torch.no_grad():
            val_batch = next(iter(test_loader))
            val_images = val_batch["images"].to(device, non_blocking=True)
            val_tokens = images_to_tokens(val_images)
            val_recon = decoder(val_tokens)

            val_mse = F.mse_loss(val_recon, val_images).item()
            val_lp = vgg_loss(val_recon, val_images).mean().item()

            print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f} | "
                  f"Val MSE: {val_mse:.4f} | LPIPS: {val_lp:.4f}")

            # Save qualitative grid
            save_eval_images(val_images, val_recon, CHECKPOINT_DIR / f"eval_epoch_{epoch+1}.png")

        # ---------------- Checkpoint ----------------
        torch.save(decoder.state_dict(), CHECKPOINT_DIR / f"epoch_{epoch+1}.pth")
