from scipy import datasets
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# imaginea cu raton
X = datasets.face(gray=True)

# afisare original
plt.imshow(X, cmap='gray')
plt.title("Original")
plt.show()

# zgomot
pixel_noise = 200
noise = np.random.randint(-pixel_noise, pixel_noise+1, size=X.shape)
X_noisy = X + noise

plt.imshow(X_noisy, cmap='gray')
plt.title("Noisy")
plt.show()

# FFT
Y = np.fft.fft2(X_noisy)
Y_shift = np.fft.fftshift(Y)

# -------------------------------------------------------
# FILTRU FOARTE SIMPLU: taiem frecventele indepartate
# -------------------------------------------------------
rows, cols = X.shape
cx, cy = rows // 2, cols // 2
cutoff = 80

for i in range(rows):
    for j in range(cols):
        if np.sqrt((i - cx)**2 + (j - cy)**2) > cutoff:
            Y_shift[i, j] = 0

# revenim la format normal
Y_clean = np.fft.ifftshift(Y_shift)

# reconstructie imagine
X_denoised = np.real(np.fft.ifft2(Y_clean))

plt.imshow(X_denoised, cmap='gray')
plt.title("Denoised")
plt.show()
