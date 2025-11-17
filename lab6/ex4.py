import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math
import time
import wavfile

import numpy as np


n = 20

x = np.cos(2 * np.pi * 2 * np.arange(n) / n)

#  deplasarea circulara d
d = 5
y = np.roll(x, d)

# FFT
X = np.fft.fft(x)
Y = np.fft.fft(y)

# 1) IFFT(FFT(x) * FFT(y))
c1 = np.fft.ifft(X * Y)

# 2) IFFT(FFT(y) / FFT(x))
c2 = np.fft.ifft(Y / X)

#  deplasarea gasita din c1 (maximul)
d_est = np.argmax(np.abs(c1))

print("d real =", d)
print("d estimat din corelatie =", d_est)

plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.stem(np.real(c1))
plt.title("Re{ IFFT( FFT(x) * FFT(y) ) }")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.stem(np.real(c2))
plt.title("Re{ IFFT( FFT(y) / FFT(x) ) }")
plt.grid(True)

plt.tight_layout()
plt.savefig("ex4.pdf", dpi=300)
plt.show()
