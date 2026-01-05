from scipy import misc, datasets, ndimage
import numpy as np
import matplotlib
from scipy.fft import dctn, idctn

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# VIDEO
# 5 cadre, fiecare cadru e imaginea face() rotita diferit
frames = []
X = datasets.face().astype(float)

for angle in [0, 10, 20, -10, -20]:
    frame = ndimage.rotate(X, angle, reshape=False)
    frames.append(frame)

video = np.array(frames)  # shape = (5, H, W, 3)

# JPEG PE CANALUL Y

Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

compressed_video = []

for idx, frame in enumerate(video):

    R = frame[:, :, 0]
    G = frame[:, :, 1]
    B = frame[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

    Y_out = np.zeros_like(Y)

    H, W = Y.shape
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block = Y[i:i + 8, j:j + 8]
            if block.shape != (8, 8):
                continue

            D = dctn(block)
            Dq = Q * np.round(D / Q)
            rec = idctn(Dq)

            Y_out[i:i + 8, j:j + 8] = rec

    # reconstruim RGB simplu
    R2 = Y_out + 1.402 * (Cr - 128)
    G2 = Y_out - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B2 = Y_out + 1.772 * (Cb - 128)

    frame_new = np.stack([R2, G2, B2], axis=2)
    frame_new = np.clip(frame_new, 0, 255).astype(np.uint8)

    compressed_video.append(frame_new)

    print("Cadru", idx, "comprimat!")

compressed_video = np.array(compressed_video)

# 2 CADRE ca exemplu

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.imshow(video[0].astype(np.uint8))
plt.title("Original cadru 0")

plt.subplot(122)
plt.imshow(compressed_video[0])
plt.title("Comprimat cadru 0")

# SAVE FIGURE
plt.savefig("ex4ss1.png", dpi=300, bbox_inches="tight")

plt.show()

