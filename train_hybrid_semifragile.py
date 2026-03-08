import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WatermarkDataset
from models_hybrid_qcnn import WatermarkHybridQCNN
from attacks import benign_attack, tamper_attack

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-3

ALPHA = 0.05

# Loss weights
LAMBDA_WM = 1.0
LAMBDA_IMG = 2.0
LAMBDA_BENIGN = 1.0   # enforce watermark under benign transforms
LAMBDA_TAMPER = 1.0   # enforce drop / mismatch under tamper

WEIGHTS_DIR = "outputs/weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

bce = nn.BCELoss()
mse = nn.MSELoss()

def main():
    ds = WatermarkDataset()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = WatermarkHybridQCNN(alpha=ALPHA).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("Device:", DEVICE)
    print("Dataset:", len(ds), "frames")
    print("Batches/epoch:", len(dl))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        pbar = tqdm(dl, desc=f"Hybrid SemiFrag Epoch {epoch}/{EPOCHS}")
        for frame, wm_logo, _ in pbar:
            frame = frame.to(DEVICE)
            wm_logo = wm_logo.to(DEVICE)

            # 1) Embed
            watermarked, wm_hat_clean, _ = model(frame, wm_logo)

            # 2) Attacks on watermarked
            wmk_b = benign_attack(watermarked)
            wmk_t = tamper_attack(watermarked, patch_frac=0.25)

            # 3) Decode after attacks
            wm_hat_b = model.decoder(wmk_b)
            wm_hat_t = model.decoder(wmk_t)

            # --- Loss terms ---
            loss_img = mse(watermarked, frame)
            loss_clean = bce(wm_hat_clean, wm_logo)
            loss_benign = bce(wm_hat_b, wm_logo)

            # Tamper: we want extracted watermark to NOT match
            # Target of 0.5 pushes it toward "uncertain / destroyed"
            target_t = torch.full_like(wm_hat_t, 0.5)
            loss_tamper = mse(wm_hat_t, target_t)

            loss = (LAMBDA_IMG * loss_img +
                    LAMBDA_WM * loss_clean +
                    LAMBDA_BENIGN * loss_benign +
                    LAMBDA_TAMPER * loss_tamper)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            pbar.set_postfix(total=loss.item(),
                             img=loss_img.item(),
                             clean=loss_clean.item(),
                             benign=loss_benign.item(),
                             tamper=loss_tamper.item())

        avg = running / len(dl)
        print(f"Epoch {epoch} avg loss: {avg:.4f}")

        ckpt = os.path.join(WEIGHTS_DIR, f"hybrid_semifrag_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt)
        print("Saved:", ckpt)

    print("✅ Hybrid QCNN Semi-fragile training complete.")

if __name__ == "__main__":
    main()