import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

imgs = os.listdir("imgs")

PATCH_W = 32
PATCH_H = 32
PATCH_C = 3
rows, cols = 8, 8
m = rows * cols
n = PATCH_H*PATCH_W*PATCH_C
W = np.random.rand(n, m)
s = np.zeros(m)
e = 0.0001

fig, axs = plt.subplots(rows, cols)

for img in imgs:
    img = Image.open('imgs/' + img)
    img = np.array(img)
    h, w, c = img.shape
    for _ in range(30):
        y, x = np.random.randint((0, 0), (h - PATCH_H, w - PATCH_W))
        patch = img[y:y + PATCH_H, x:x + PATCH_W]
        assert patch.shape == (PATCH_H, PATCH_W, PATCH_C)
        patch = np.random.rand(*patch.shape) < patch / 255
        x = patch.reshape(-1)
        k = np.argmax(x @ W + s)
        s[k] -= e
        W[x, k] += e/patch.sum()
        W[:, k] /= np.linalg.norm(W[:, k])
    for col in range(cols):
        for row in range(rows):
            s = W[:, col + row * w].reshape(PATCH_H, PATCH_W, PATCH_C)
            axs[col, row].imshow(s)
    plt.pause(0.01)
