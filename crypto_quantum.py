import numpy as np
import hashlib
import time

def derive_seed(user_secret: str, video_hash: str, frame_idx: int, use_time=True):
    t = int(time.time()) if use_time else "fixed-session"
    raw = f"{user_secret}|{video_hash}|{frame_idx}|{t}"
    h = hashlib.sha256(raw.encode()).hexdigest()
    x0 = (int(h[:8], 16) % 10**6) / 10**6
    return x0

def chaotic_key(seed, bits_len, r=3.99):
    x = seed
    key = np.zeros(bits_len, dtype=np.uint8)
    for i in range(bits_len):
        x = r * x * (1 - x)
        key[i] = 1 if x > 0.5 else 0
    return key

def permute_bits(bits, key):
    rng = np.random.default_rng(int(np.packbits(key).sum()))
    idx = rng.permutation(len(bits))
    return bits[idx], idx

def invert_permutation(bits_p, idx):
    inv = np.zeros_like(idx)
    inv[idx] = np.arange(len(idx))
    return bits_p[inv]

def encrypt_bits(bits, key):
    bits_p, perm = permute_bits(bits, key)
    enc = np.bitwise_xor(bits_p, key[:len(bits)])
    return enc, perm

def decrypt_bits(enc_bits, key, perm):
    bits_p = np.bitwise_xor(enc_bits, key[:len(enc_bits)])
    dec = invert_permutation(bits_p, perm)
    return dec