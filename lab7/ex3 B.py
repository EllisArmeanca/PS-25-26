from scipy import datasets
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# imaginea cu raton
X = datasets.face(gray=True)

# adaug zgomot
pixel_noise = 200
noise = np.random.randint(-pixel_noise, pixel_noise+1, size=X.shape)
X_noisy = X + noise

plt.imshow(X, cmap='gray')
plt.title("Original")
plt.show()

plt.imshow(X_noisy, cmap='gray')
plt.title("Noisy")
plt.show()

# SNR inainte
suma_semnal = np.sum(X ** 2)
suma_zgomot = np.sum((X - X_noisy) ** 2)
print("SNR înainte filtrare:", 10*np.log10(suma_semnal/suma_zgomot))

# FFT - IMPORTANT: folosim abs + log1p corect
Y_noisy = np.fft.fft2(X_noisy)
freq_db = 20 * np.log10(np.abs(Y_noisy) + 1e-8)   # evităm log(0)

plt.imshow(np.log1p(np.abs(np.fft.fftshift(Y_noisy))), cmap='gray')
plt.title("Spectru înainte filtrare")
plt.colorbar()
plt.show()

# cutoff EXACT ca in laborator (pe amplitudine, nu pe radial)
freq_cutoff = 120

Y_cutoff = Y_noisy.copy()
Y_cutoff[freq_db > freq_cutoff] = 0   # ACEASTA ERA PROBLEMA: freq_db era greșit

# afisare spectru filtrat
plt.imshow(np.log1p(np.abs(np.fft.fftshift(Y_cutoff))), cmap='gray')
plt.title("Spectru după filtrare")
plt.colorbar()
plt.show()

# reconstructie
X_denoised = np.fft.ifft2(Y_cutoff)
X_denoised = np.real(X_denoised)

plt.imshow(X_denoised, cmap='gray')
plt.title("Denoised")
plt.show()

# SNR dupa filtrare
suma_zgomot2 = np.sum((X - X_denoised)**2)
print("SNR după filtrare:", 10*np.log10(suma_semnal/suma_zgomot2))
