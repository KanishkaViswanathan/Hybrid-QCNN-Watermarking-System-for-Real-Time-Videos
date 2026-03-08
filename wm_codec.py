import numpy as np

def logo_to_bits(wm_logo_np):
    return (wm_logo_np > 0.5).astype(np.uint8).reshape(-1)

def bits_to_logo(bits, size):
    return bits.reshape(size).astype(np.float32)