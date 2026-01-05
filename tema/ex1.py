from scipy import misc, ndimage, datasets
import numpy as np
import matplotlib
from scipy.fft import idctn, dctn

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


X = datasets.ascent()
plt.imshow(X, cmap=plt.cm.gray)
plt.title("Original")
plt.show()

# Matricea de cuantizare JPEG

Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

# Copii pentru encoding si decoding
X_encoded = X.copy().astype(float)
X_decoded = np.zeros_like(X_encoded)

# Dimensiunea imaginii
H, W = X.shape

#  imaginea pe blocuri de 8x8
for i in range(0, H, 8):
    for j in range(0, W, 8):
        block = X_encoded[i:i+8, j:j+8]

        if block.shape != (8, 8):
            continue

        # DCT pe bloc
        Y = dctn(block)

        # Cuantizare
        Y_q = Q_jpeg * np.round(Y / Q_jpeg)

        # iDCT
        block_rec = idctn(Y_q)

        # Salvare
        X_decoded[i:i+8, j:j+8] = block_rec


plt.subplot(121).imshow(X, cmap=plt.cm.gray)
plt.title("Original")
plt.subplot(122).imshow(X_decoded, cmap=plt.cm.gray)
plt.title("JPEG blocuri 8x8")

# SAVE FIGURE
plt.savefig("ex1ss1.png", dpi=300, bbox_inches="tight")

plt.show()

#dct - transformarea blocului in frevente#
# cuantizare cu !
#decoding = reconstruirea imaginii
#1. inmultesti inapoi cu Q
#idct - din frecvente in pixeli
