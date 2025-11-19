from scipy import misc, ndimage, datasets
import numpy as np
import matplotlib
matplotlib.use('TkAgg')        # setam backend inainte de pyplot
import matplotlib.pyplot as plt

#SignalToNoiseRatio =
#10 log 10 ( suma x ^ 2 / suma (x - x_rec) ^ 2 )


#incarc ratonul
X = datasets.face(gray=True)
plt.imshow(X, cmap=plt.cm.gray)
plt.show()


#transformata fourier
Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y))

plt.imshow(freq_db)
plt.colorbar()
plt.show()

#pragul de frecventa
freq_cutoff = 120


#copiez spectrul si anulez frecventele inalte
Y_cutoff = Y.copy()
Y_cutoff[freq_db > freq_cutoff] = 0

#recomstruisc imaginea prin trasnformta inversa

X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)    # avoid rounding erros in the complex domain,
                                # in practice use irfft2

#afisez imaginea
plt.imshow(X_cutoff, cmap=plt.cm.gray)
plt.title("IMAGINE COMPRIMATA")
plt.show()

#calculez SNR
suma_semnal = np.sum(X ** 2)
suma_zgomot = np.sum((X - X_cutoff) ** 2)
SNR = 10 * np.log10(suma_semnal / suma_zgomot)
print("SNR =", SNR, "dB")
# cu cat pragul de frecventa e mai mare, cu atat SNR e mai mare

#salvez imaginea
plt.imsave("raton_comprimat.png", X_cutoff, cmap=plt.cm.gray)


###cutofful imi spune cate elemente utile ale imaginii sa PASTREZ... daca e mai mic inseamna ca sunt eliminate mai multe