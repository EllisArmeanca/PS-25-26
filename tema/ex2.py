from scipy import misc, ndimage, datasets
import numpy as np
import matplotlib
from scipy.fft import idctn, dctn

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# imaginea color
X = misc.face()
plt.imshow(X)
plt.title("Original RGB")
plt.show()

# ---------------------------------------------------
# 1. Conversie RGB -> Y, Cb, Cr (FOARTE simplu)
# ---------------------------------------------------

Xf = X.astype(float)

R = Xf[:,:,0]
G = Xf[:,:,1]
B = Xf[:,:,2]

# formule simple, standard
Y  = 0.299*R + 0.587*G + 0.114*B
Cb = -0.168736*R - 0.331264*G + 0.5*B + 128
Cr =  0.5*R - 0.418688*G - 0.081312*B + 128

# ---------------------------------------------------
# 2. JPEG aplicat DOAR pe canalul Y (codul tau)
# ---------------------------------------------------

Q_jpeg = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Y_encoded = Y.copy()
Y_decoded = np.zeros_like(Y)

H, W = Y.shape

# acelasi cod ca la sarcina 1
for i in range(0, H, 8):
    for j in range(0, W, 8):
        block = Y_encoded[i:i+8, j:j+8]

        if block.shape != (8, 8):
            continue

        Y_dct = dctn(block)
        Y_q = Q_jpeg * np.round(Y_dct / Q_jpeg)
        block_rec = idctn(Y_q)

        Y_decoded[i:i+8, j:j+8] = block_rec


# ---------------------------------------------------
# 3. Reconstruire RGB -> imagine color
# ---------------------------------------------------

R_new = Y_decoded + 1.402*(Cr - 128)
G_new = Y_decoded - 0.344136*(Cb - 128) - 0.714136*(Cr - 128)
B_new = Y_decoded + 1.772*(Cb - 128)

X_new = np.zeros_like(Xf)
X_new[:,:,0] = R_new
X_new[:,:,1] = G_new
X_new[:,:,2] = B_new

X_new = np.clip(X_new, 0, 255).astype(np.uint8)

# ---------------------------------------------------
# 4. Afisare imagine JPEG color
# ---------------------------------------------------

plt.imshow(X_new)
plt.title("JPEG Color (canalul Y comprimat)")
plt.show()
