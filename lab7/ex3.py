from scipy import misc, ndimage, datasets
import numpy as np
import matplotlib
matplotlib.use('TkAgg')        # setam backend inainte de pyplot
import matplotlib.pyplot as plt

# imaginea cu raton
X = datasets.face(gray=True)
plt.imshow(X, cmap=plt.cm.gray)
plt.title("Original")
plt.show()

# adaug zgomot ( lab )
pixel_noise = 200
noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise

plt.imshow(X_noisy, cmap=plt.cm.gray)
plt.title("Noisy")
plt.show()

# calculam SNR inainte
suma_semnal = np.sum(X ** 2)
suma_zgomot = np.sum((X - X_noisy) ** 2)
SNR_inainte = 10 * np.log10(suma_semnal / suma_zgomot)
print("SNR inainte filtrare:", SNR_inainte, "dB")

# transformata fourier a imaginii zgomotoase
Y_noisy = np.fft.fft2(X_noisy)
freq_db_noisy = 20*np.log10(abs(Y_noisy))

# afisam spectrul inainte de filtrare
plt.figure()
plt.imshow(np.log1p(np.abs(np.fft.fftshift(Y_noisy))), cmap='gray')
plt.title("Y_noisy - spectru inainte de filtrare")
plt.colorbar()
plt.show()


# prag de frecventa
freq_cutoff = 150

# eliminam frecventele inalte
Y_clean = Y_noisy.copy()
Y_clean[freq_db_noisy > freq_cutoff] = 0

# afisam spectrul dupa filtrare
plt.figure()
plt.imshow(np.log1p(np.abs(np.fft.fftshift(Y_clean))), cmap='gray')
plt.title("Y_clean - spectru DUPA filtrare")
plt.colorbar()
plt.show()


X_denoised = np.fft.ifft2(Y_clean)
X_denoised = np.real(X_denoised)

plt.imshow(X_denoised, cmap=plt.cm.gray)
plt.title("Denoised")
plt.show()

# calculam SNR dupa filtrare
suma_zgomot2 = np.sum((X - X_denoised) ** 2)
SNR_dupa = 10 * np.log10(suma_semnal / suma_zgomot2)
print("SNR dupa filtrare:", SNR_dupa, "dB")
