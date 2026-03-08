import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from dataset import WatermarkDataset
from models_hybrid_qcnn import WatermarkHybridQCNN
from attacks import benign_attack, tamper_attack

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "outputs/weights/hybrid_semifrag_epoch5.pt"
ALPHA = 0.05

def normalized_correlation(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

def to_uint8_img(x):
    x = x.detach().cpu().clamp(0,1).permute(1,2,0).numpy()
    return (x*255).round().astype(np.uint8)

def main():
    ds = WatermarkDataset()
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    model = WatermarkHybridQCNN(alpha=ALPHA).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    N = 10
    psnr_c, ssim_c, nc_c = [], [], []
    psnr_b, ssim_b, nc_b = [], [], []
    psnr_t, ssim_t, nc_t = [], [], []

    with torch.no_grad():
        for i, (frame, wm_logo, _) in enumerate(dl):
            if i >= N:
                break
            frame = frame.to(DEVICE)
            wm_logo = wm_logo.to(DEVICE)

            watermarked, wm_hat_clean, _ = model(frame, wm_logo)

            wmk_b = benign_attack(watermarked)
            wmk_t = tamper_attack(watermarked, patch_frac=0.25)

            wm_hat_b = model.decoder(wmk_b)
            wm_hat_t = model.decoder(wmk_t)

            orig_u8 = to_uint8_img(frame[0])
            clean_u8 = to_uint8_img(watermarked[0])
            benign_u8 = to_uint8_img(wmk_b[0])
            tamper_u8 = to_uint8_img(wmk_t[0])

            psnr_c.append(psnr(orig_u8, clean_u8, data_range=255))
            ssim_c.append(ssim(orig_u8, clean_u8, channel_axis=2, data_range=255))

            psnr_b.append(psnr(orig_u8, benign_u8, data_range=255))
            ssim_b.append(ssim(orig_u8, benign_u8, channel_axis=2, data_range=255))

            psnr_t.append(psnr(orig_u8, tamper_u8, data_range=255))
            ssim_t.append(ssim(orig_u8, tamper_u8, channel_axis=2, data_range=255))

            w_true = wm_logo[0,0].cpu().numpy()
            nc_c.append(normalized_correlation(w_true, wm_hat_clean[0,0].cpu().numpy()))
            nc_b.append(normalized_correlation(w_true, wm_hat_b[0,0].cpu().numpy()))
            nc_t.append(normalized_correlation(w_true, wm_hat_t[0,0].cpu().numpy()))

    def avg(x): return float(sum(x)/len(x))

    print("\n=== Hybrid Semi-Fragile QCNN Metrics ===")
    print("Avg PSNR (clean): ", avg(psnr_c))
    print("Avg SSIM (clean): ", avg(ssim_c))
    print("Avg NC   (clean): ", avg(nc_c))
    print("Avg PSNR (benign):", avg(psnr_b))
    print("Avg SSIM (benign):", avg(ssim_b))
    print("Avg NC   (benign):", avg(nc_b))
    print("Avg PSNR (tamper):", avg(psnr_t))
    print("Avg SSIM (tamper):", avg(ssim_t))
    print("Avg NC   (tamper):", avg(nc_t))

if __name__ == "__main__":
    main()