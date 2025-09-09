"""
train_decoder_droid_split.py

Train a convolutional decoder on DROID-100 RLDS frames with train/test split.
Evaluates both during training (per-epoch) and after training (full test set).
"""

import os
import math
import numpy as np
from pathlib import Path
from datetime import datetime

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# ================================================================
# Dataset
# ================================================================
class DroidRLDSFrames(Dataset):
    def __init__(self, root, camera="exterior_image_1_left", target_size=(256, 256)):
        self.root = Path(root) / "1.0.0"
        self.files = sorted(self.root.glob("*.tfrecord*"))
        if not self.files:
            raise FileNotFoundError(f"No tfrecord files found under {self.root}")

        self.camera = camera
        self.transform = T.Compose([
            T.Resize(target_size, antialias=True),
            T.ToTensor(),
        ])

        dataset = tf.data.TFRecordDataset([str(f) for f in self.files])
        self.records = list(dataset.map(self._parse_fn).as_numpy_iterator())
        print(f"DROID RLDS: {len(self.records)} frames from camera '{camera}'")

    def _parse_fn(self, example_proto):
        feature_desc = {
            f"steps/observation/{self.camera}": tf.io.VarLenFeature(tf.string),
        }
        example = tf.io.parse_single_example(example_proto, feature_desc)
        img_bytes = tf.sparse.to_dense(example[f"steps/observation/{self.camera}"])[0]
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        return img

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img = self.records[idx]
        pil = T.functional.to_pil_image(img)
        img_t = self.transform(pil)
        return {"images": img_t}


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


# class TokenDecoder(nn.Module):
#     """
#     Token grid [B,N,D] -> RGB [B,3,256,256] using Upsample+Conv blocks
#     to avoid checkerboard artifacts from ConvTranspose2d.
#     """
#     def __init__(self, embed_dim: int, tokens_side: int, out_res: int = 256, base_ch: int = 2048):
#         super().__init__()
#         self.tokens_side = tokens_side
#         self.out_res = out_res

#         # project D -> base_ch on the token grid
#         self.proj = nn.Conv2d(embed_dim, base_ch, kernel_size=1)

#         def block(cin, cout):
#             return nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#                 nn.Conv2d(cin, cout, 3, padding=1),
#                 nn.GroupNorm(8 if cout % 8 == 0 else max(1, cout // 8), cout),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(cout, cout, 3, padding=1),
#                 nn.ReLU(inplace=True),
#             )


#         layers = []
#         ch = base_ch
#         steps = int(np.log2(out_res // tokens_side))
#         for _ in range(steps):
#             next_ch = max(64, ch // 2)
#             layers.append(block(ch, next_ch))
#             ch = next_ch

#         self.up = nn.Sequential(*layers)
#         self.head = nn.Sequential(
#             nn.Conv2d(ch, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, tokens: torch.Tensor) -> torch.Tensor:
#         B, N, D = tokens.shape
#         S = int(math.sqrt(N))
#         assert S * S == N, "tokens must form a square grid"
#         x = tokens.transpose(1, 2).reshape(B, D, S, S)  # [B,D,S,S]
#         x = self.proj(x)
#         x = self.up(x)
#         x = self.head(x)
#         return x



# ================================================================
# V-JEPA tokenization
# ================================================================
@torch.no_grad()
def images_to_tokens(images: torch.Tensor, encoder) -> torch.Tensor:
    dev = next(encoder.parameters()).device
    images = images.to(dev, non_blocking=True)
    images = images.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # fake T=2
    tkns = encoder(images)

    H, W = images.shape[-2:]
    ps = getattr(encoder, "patch_size", 16)
    Ts2 = (H // ps) * (W // ps)

    return tkns[:, -Ts2:, :]


# ================================================================
# Save sample reconstructions
# ================================================================
def save_eval_images(gt, recon, out_path, max_b=4, title="Top: GT | Bottom: Recon"):
    gt = gt[:max_b].detach().cpu().clamp(0, 1)
    recon = recon[:max_b].detach().cpu().clamp(0, 1)
    top = torch.cat(list(gt), dim=2)
    bot = torch.cat(list(recon), dim=2)
    grid = torch.cat([top, bot], dim=1).permute(1, 2, 0).numpy()

    plt.figure(figsize=(4*max_b, 8))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# ================================================================
# Main training + eval
# ================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset + split
    dataset = DroidRLDSFrames(
        "/data/maddie/vjepa2/droid/droid_100",
        camera="exterior_image_1_left"
    )
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Encoder
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    encoder = encoder.to(device).eval()
    patch_size = encoder.patch_size
    tokens_per_side = 256 // patch_size

    # Decoder
    decoder = TokenDecoder(embed_dim=encoder.embed_dim, tokens_side=tokens_per_side).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=1e-4)

    # LPIPS
    import lpips
    vgg_loss = lpips.LPIPS(net="vgg").to(device)

    # Checkpoint dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = Path("checkpoints") / f"droid_run_{timestamp}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        decoder.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = batch["images"].to(device)

            with torch.no_grad():
                tokens = images_to_tokens(images, encoder)

            opt.zero_grad()
            recon = decoder(tokens)

            mse = F.mse_loss(recon, images)
            lp  = vgg_loss(2*recon - 1, 2*images - 1).mean()   # <-- fix
            loss = mse + 0.3 * lp   # try 0.3–0.5

            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        # Quick validation on one batch
        decoder.eval()
        with torch.no_grad():
            val_batch = next(iter(test_loader))
            val_images = val_batch["images"].to(device)
            val_tokens = images_to_tokens(val_images, encoder)
            val_recon = decoder(val_tokens)
            val_mse = F.mse_loss(val_recon, val_images).item()
            val_lp = vgg_loss(val_recon, val_images).mean().item()
            print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f} | "
                  f"Val MSE (1 batch): {val_mse:.4f} | LPIPS: {val_lp:.4f}")

            save_eval_images(val_images, val_recon, ckpt_dir / f"val_epoch_{epoch+1}.png")

        torch.save(decoder.state_dict(), ckpt_dir / f"epoch_{epoch+1}.pth")

    # ============================================================
    # Final evaluation on full test set
    # ============================================================
    decoder.eval()
    test_mse_total, test_lp_total, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["images"].to(device)
            tokens = images_to_tokens(images, encoder)
            recon = decoder(tokens)

            test_mse_total += F.mse_loss(recon, images).item()
            test_lp_total += vgg_loss(recon, images).mean().item()
            n_batches += 1

            # Save a couple of batches of reconstructions
            if i < 3:
                save_eval_images(images, recon, ckpt_dir / f"test_batch_{i}.png", title="Test: GT | Recon")

    print(f"\nFinal Test Results over {n_batches} batches:")
    print(f"  Avg MSE:   {test_mse_total / n_batches:.4f}")
    print(f"  Avg LPIPS: {test_lp_total / n_batches:.4f}")


