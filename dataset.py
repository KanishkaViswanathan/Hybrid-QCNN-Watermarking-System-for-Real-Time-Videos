import os, glob, hashlib, cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from crypto_quantum import derive_seed, chaotic_key, encrypt_bits
from wm_codec import logo_to_bits, bits_to_logo

FRAMES_DIR = "data/frames/input"
WM_LOGO_PATH = "data/watermarks/wm_logo.png"
VIDEO_PATH = "data/raw_videos/input.mp4"

IMG_SIZE = 256
WM_SIZE = 64
USER_SECRET = "kanishka-secret-key"

def compute_video_hash(video_path, sample_frames=5):
    cap = cv2.VideoCapture(video_path)
    h = hashlib.sha256()
    count = 0

    while cap.isOpened() and count < sample_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h.update(frame.tobytes())
        count += 1

    cap.release()
    return h.hexdigest()

class WatermarkDataset(Dataset):
    def __init__(self, frames_dir=FRAMES_DIR):
        self.frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if len(self.frame_paths) == 0:
            raise RuntimeError("No frames found")

        wm_img = Image.open(WM_LOGO_PATH).convert("L").resize((WM_SIZE, WM_SIZE))
        self.wm_logo = np.array(wm_img, dtype=np.float32) / 255.0

        self.video_hash = compute_video_hash(VIDEO_PATH)

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img = Image.open(self.frame_paths[idx]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        frame = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2,0,1)

        # 1) logo → bits
        wm_bits = logo_to_bits(self.wm_logo)

        # 2) encrypt bits (chaos + permutation + XOR diffusion)
        seed = derive_seed(USER_SECRET, self.video_hash, idx, use_time=False)
        key = chaotic_key(seed, len(wm_bits))
        enc_bits, _ = encrypt_bits(wm_bits, key)

        # 3) bits → encrypted logo image
        enc_logo = bits_to_logo(enc_bits, (WM_SIZE, WM_SIZE))
        wm_logo_enc = torch.from_numpy(enc_logo).unsqueeze(0).float()
        
        return frame, wm_logo_enc, torch.from_numpy(wm_bits).float()