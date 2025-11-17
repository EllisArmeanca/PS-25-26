import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math
import time
import wavfile

import numpy as np


# definim x(t) = sinc^2(Bt)
B = 1
t = np.linspace(-3, 3, 2000)
x_t = np.sinc(B * t) ** 2

# reconstructie sinc simpla
def reconstruct(t, t_n, x_n, Ts):
    x_hat = np.zeros_like(t)
    for i in range(len(t_n)):
        x_hat += x_n[i] * np.sinc((t - t_n[i]) / Ts)
    return x_hat

Fs_list = [1, 1.5, 2, 4]

plt.figure(figsize=(10, 12))

for idx, Fs in enumerate(Fs_list, 1):
    Ts = 1 / Fs

    # instante de esantionare
    n_min = int(np.ceil(-3 / Ts))
    n_max = int(np.floor(3 / Ts))
    n = np.arange(n_min, n_max + 1)
    t_n = n * Ts

    # valori esantionate
    x_n = np.sinc(B * t_n) ** 2

    # reconstructie sinc
    x_hat = reconstruct(t, t_n, x_n, Ts)

    # plot
    plt.subplot(4, 1, idx)
    plt.plot(t, x_t, label='x(t) original')              # semnal continuu
    plt.plot(t_n, x_n, 'ro', label='esantioane')         # puncte rosii
    plt.plot(t, x_hat, '--', label='reconstructie sinc') # reconstructie

    plt.title(f'Fs = {Fs} Hz')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig("ex1.pdf", dpi=300)
plt.show()
