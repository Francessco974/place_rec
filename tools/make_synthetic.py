"""Synthetic TUM-format RGB-D sequence for smoke tests (no download needed).

A camera looks at a textured wall (plane z = 2.5 m), drives 6 m along x at 0.2 m/s and,
in 'loop' mode, drives back on a slightly offset path -> true revisits.
usage: python3 make_synthetic.py <out_dir> <seed> loop|noloop
"""
import os
import sys

import cv2
import numpy as np

out, seed, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(seed)

S, Zp = 250.0, 2.5                         # texture px per metre, wall depth [m]
W, H = int(12 * S), int(4 * S)
tex = np.full((H, W), 128, np.uint8)
for _ in range(9000):                      # random shapes -> many distinct corners
    c, x, y, r = int(rng.integers(0, 256)), int(rng.integers(0, W)), int(rng.integers(0, H)), int(rng.integers(4, 40))
    if rng.random() < 0.5:
        cv2.circle(tex, (x, y), r, c, -1)
    else:
        cv2.rectangle(tex, (x, y), (x + r, y + int(rng.integers(4, 40))), c, -1)
tex = cv2.GaussianBlur(tex, (3, 3), 0)

K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
uu, vv = np.meshgrid(np.arange(640), np.arange(480))
rays = np.linalg.inv(K) @ np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)])  # z = 1


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


xs = np.arange(0.0, 6.0, 0.02)             # 10 Hz at 0.2 m/s
poses = [(np.array([x - 3.0, 0.0, 0.0]), 0.05 * np.sin(x)) for x in xs]
if mode == "loop":                          # come back 8 cm lower, 15 cm closer, yawed
    poses += [(np.array([x - 3.0, 0.08, 0.15]), 0.05 * np.sin(x) + 0.08) for x in xs[::-1]]

os.makedirs(f"{out}/rgb", exist_ok=True)
os.makedirs(f"{out}/depth", exist_ok=True)
files = {n: open(f"{out}/{n}.txt", "w") for n in ("rgb", "depth", "groundtruth")}
for f in files.values():
    f.write("# synthetic sequence\n")
for k, (t, yaw) in enumerate(poses):
    d = Ry(yaw) @ rays                      # rays in the world frame
    lam = (Zp - t[2]) / d[2]                # ray-plane intersection; = depth since ray z_c = 1
    X = t[:, None] + lam * d
    img = cv2.remap(tex, (X[0] * S + W / 2).reshape(480, 640).astype(np.float32),
                    (X[1] * S + H / 2).reshape(480, 640).astype(np.float32), cv2.INTER_LINEAR)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    ts, td = 1000.0 + 0.1 * k, 1000.0 + 0.1 * k + 0.005
    cv2.imwrite(f"{out}/rgb/{ts:.6f}.png", img)
    cv2.imwrite(f"{out}/depth/{td:.6f}.png", (lam.reshape(480, 640) * 5000).astype(np.uint16))
    files["rgb"].write(f"{ts:.6f} rgb/{ts:.6f}.png\n")
    files["depth"].write(f"{td:.6f} depth/{td:.6f}.png\n")
    files["groundtruth"].write(f"{ts:.6f} {t[0]} {t[1]} {t[2]} 0 {np.sin(yaw/2)} 0 {np.cos(yaw/2)}\n")
print(f"{out}: {len(poses)} frames")