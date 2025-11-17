import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math
import time
import wavfile

import numpy as np


#  fereastra dreptunghiulara
def rect_window(N):
    # toate valorile 1
    return np.ones(N)

#  Hanning
def hanning_window(N):
    n = np.arange(N)
    w = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    return w

Nw = 200

# sinusoida cu f = 100 Hz, A = 1, phi = 0
fs = 1000.0           # frecventa de esantionare aleasa
n = np.arange(Nw)
t = n / fs
x = np.sin(2 * np.pi * 100 * t)

# ferestre
w_rect = rect_window(Nw)
w_hann = hanning_window(Nw)

# semnal ferestrat
x_rect = x * w_rect
x_hann = x * w_hann

plt.figure(figsize=(10, 8))

# sinusoida originala
plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.title("Sinusoida originala (f = 100 Hz)")
plt.grid(True)

# cu fereastra dreptunghiulara
plt.subplot(3, 1, 2)
plt.plot(t, x_rect)
plt.title("Sinusoida cu fereastra dreptunghiulara")
plt.grid(True)

# cu fereastra Hanning
plt.subplot(3, 1, 3)
plt.plot(t, x_hann)
plt.title("Sinusoida cu fereastra Hanning")
plt.grid(True)

plt.tight_layout()
plt.savefig("ex5.pdf", dpi=300)
plt.show()
