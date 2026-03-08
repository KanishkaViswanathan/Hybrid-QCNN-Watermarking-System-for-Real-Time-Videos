import cv2
import os

VIDEO_PATH = "data/raw_videos/input.mp4"
OUT_DIR = "data/frames/input"
TARGET_SIZE = (256, 256)
SAVE_EVERY_N = 1  # later we’ll set 5 for real-time

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

idx, saved = 0, 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if idx % SAVE_EVERY_N == 0:
        frame = cv2.resize(frame, TARGET_SIZE)
        out_path = os.path.join(OUT_DIR, f"frame_{saved:05d}.png")
        cv2.imwrite(out_path, frame)
        saved += 1

    idx += 1

cap.release()
print(f"Done. Saved {saved} frames to {OUT_DIR}")