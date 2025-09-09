"""
Train a ConvTranspose2d decoder (original architecture) on DROID-100 RLDS.

Pipeline (matches your pickle script):
1) Dataset: one frame per sample, resized to 256x256.
2) Encoder: V-JEPA AC (T=2; we duplicate the frame).
3) Decoder: ConvTranspose2d pyramid -> 256x256 RGB.
4) Loss: MSE + 0.1 * LPIPS(VGG).
5) Saves eval grids and checkpoints per epoch.

Note: For strict parity with your original setup, LPIPS is called on [0,1] tensors.
(If you want sharper results, call LPIPS on [-1,1]: vgg_loss(2*img-1, 2*recon-1))
"""

import os
import io
import math
import numpy as np
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


# ================================================================
# Dataset (RLDS → frames)
# ================================================================
class DroidRLDSFrames(Dataset):
    """
    Loads DROID RLDS TFRecords and preloads *all* frames (as JPEG bytes) into RAM,
    then decodes to tensors on __getitem__.

    Args:
      root: folder that contains "1.0.0" with TFRecord shards.
      camera: e.g. "exterior_image_1_left", "exterior_image_2_left", "wrist_image_left".
      target_size: output image size (H, W), default (256, 256).
    """
    def __init__(self, root, camera="exterior_image_1_left", target_size=(256, 256)):
        self.root = Path(root) / "1.0.0"
        self.files = sorted(self.root.glob("*.tfrecord*"))
        if not self.files:
            raise FileNotFoundError(f"No tfrecord files found under {self.root}")
        self.camera = camera
        self.size = target_size

        # Preload all JPEG bytes for the chosen camera into a python list
        feat = {f"steps/observation/{self.camera}": tf.io.VarLenFeature(tf.string)}
        frames = []

        # Iterate every Example; each one holds a *sequence* of images in that feature
        ds = tf.data.TFRecordDataset([str(f) for f in self.files])
        for raw in ds:
            ex = tf.io.parse_single_example(raw, feat)
            seq = tf.sparse.to_dense(ex[f"steps/observation/{self.camera}"])  # [T]
            # extend with each frame's bytes
            for b in seq.numpy():
                frames.append(b)  # 'b' is a bytes object

        self._jpeg_bytes = frames
        print(f"DROID RLDS: loaded {len(self._jpeg_bytes)} frames from '{self.camera}'")

        # transforms: PIL -> tensor in [0,1]
        self.to_tensor = T.Compose([
            T.Resize(self.size, antialias=True),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self._jpeg_bytes)

    def __getitem__(self, idx):
        # decode JPEG from bytes using PIL
        img = Image.open(io.BytesIO(self._jpeg_bytes[idx])).convert("RGB")
        img_t = self.to_tensor(img)  # [3,256,256] in [0,1]
        return {"images": img_t}


# ================================================================
# Decoder (original ConvTranspose2d version)
# ================================================================
class TokenDecoder(nn.Module):
    """
    Original upsampling decoder:
    1x1 Conv (D -> base_ch)
      -> [ConvTranspose2d, GroupNorm, ReLU] x k (doubling each step)
      -> 3x3 Conv (to 3) + Sigmoid
    """
    def __init__(self, embed_dim: int, tokens_side: int, out_res: int = 256, base_ch: int = 2048):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, base_ch, kernel_size=1)

        layers, ch = [], base_ch
        steps = int(np.log2(out_res // tokens_side))
        for _ in range(steps):
            next_ch = ch // 2
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
        B, N, D = tokens.shape
        S = int(math.sqrt(N))
        assert S * S == N, "tokens must form a square grid (N=S*S)."
        z = tokens.transpose(1, 2).reshape(B, D, S, S)  # [B,D,S,S]
        z = self.proj(z)
        return self.up(z)


# ================================================================
# V-JEPA tokenization (T=2: duplicate the single frame)
# ================================================================
@torch.no_grad()
def images_to_tokens(images: torch.Tensor, encoder) -> torch.Tensor:
    dev = next(encoder.parameters()).device
    x = images.to(dev, non_blocking=True).unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [B,3,2,H,W]
    tkns = encoder(x)  # [B, ND, D]

    H, W = images.shape[-2:]
    ps = getattr(encoder, "patch_size", 16)
    Ts2 = (H // ps) * (W // ps)
    return tkns[:, -Ts2:, :]  # last frame’s tokens


# ================================================================
# Save side-by-side grids
# ================================================================
def save_eval_images(gt: torch.Tensor, recon: torch.Tensor, out_path: Path, max_b: int = 4,
                     title: str = "Top: Ground Truth | Bottom: Reconstruction"):
    gt = gt[:max_b].detach().cpu().clamp(0, 1)
    recon = recon[:max_b].detach().cpu().clamp(0, 1)
    top = torch.cat(list(gt), dim=2)
    bot = torch.cat(list(recon), dim=2)
    grid = torch.cat([top, bot], dim=1).permute(1, 2, 0).numpy()

    plt.figure(figsize=(4 * max_b, 8))
    plt.imshow(grid); plt.axis("off"); plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path); plt.close()


# ================================================================
# Main training + eval
# ================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Data -----
    data_root = "/data/maddie/vjepa2/droid/droid_100"   # <-- change if needed
    camera = "exterior_image_1_left"
    dataset = DroidRLDSFrames(data_root, camera=camera, target_size=(256, 256))

    # split
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # ----- Encoder -----
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    encoder = encoder.to(device).eval()

    patch_size = encoder.patch_size
    assert 256 % patch_size == 0, "Input 256 must be divisible by encoder.patch_size"
    tokens_per_side = 256 // patch_size

    # ----- Decoder (original) -----
    decoder = TokenDecoder(embed_dim=encoder.embed_dim, tokens_side=tokens_per_side,
                           out_res=256, base_ch=2048).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=1e-4)

    # LPIPS (VGG)
    import lpips
    vgg_loss = lpips.LPIPS(net="vgg").to(device)

    # ckpt dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = Path("checkpoints") / f"droid_origdec_{timestamp}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----- Train -----
    num_epochs = 20
    for epoch in range(num_epochs):
        decoder.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = batch["images"].to(device, non_blocking=True)

            with torch.no_grad():
                tokens = images_to_tokens(images, encoder)  # [B, Ts^2, D]

            opt.zero_grad()
            recon = decoder(tokens)
            loss_mse = F.mse_loss(recon, images)
            loss_lp = vgg_loss(recon, images).mean()      # (kept identical to your original)
            loss = loss_mse + 0.1 * loss_lp
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        # quick val on one batch
        decoder.eval()
        with torch.no_grad():
            vb = next(iter(test_loader))
            vimg = vb["images"].to(device, non_blocking=True)
            vtoks = images_to_tokens(vimg, encoder)
            vrec = decoder(vtoks)
            vmse = F.mse_loss(vrec, vimg).item()
            vlp  = vgg_loss(vrec, vimg).mean().item()
            print(f"[Epoch {epoch+1}] TrainLoss {epoch_loss:.4f} | Val MSE {vmse:.4f} | LPIPS {vlp:.4f}")
            save_eval_images(vimg, vrec, ckpt_dir / f"val_epoch_{epoch+1}.png")

        torch.save(decoder.state_dict(), ckpt_dir / f"epoch_{epoch+1}.pth")

    # ----- Full test eval -----
    decoder.eval()
    test_mse, test_lp, nb = 0.0, 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            imgs = batch["images"].to(device)
            toks = images_to_tokens(imgs, encoder)
            recs = decoder(toks)
            test_mse += F.mse_loss(recs, imgs).item()
            test_lp  += vgg_loss(recs, imgs).mean().item()
            nb += 1
            if i < 3:
                save_eval_images(imgs, recs, ckpt_dir / f"test_batch_{i}.png", title="Test: GT | Recon")
    print(f"\nFinal Test over {nb} batches: MSE={test_mse/nb:.4f}  LPIPS={test_lp/nb:.4f}")
