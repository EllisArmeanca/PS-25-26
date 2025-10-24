# import librariile necesare pentru grafice, calcule si sunet
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math
import numpy as np

# setam parametrii de esantionare
fs = 1000        # frecventa de esantionare
dur = 1.0        # durata semnalului (1 secunda)
N = int(fs * dur)  # numarul total de esantioane
k = np.arange(N)   # vectorul frecventelor discrete
n = np.arange(N)   # vectorul esantioanelor

T = 1 / fs        # perioada de esantionare

# definim cele 3 frecvente componente ale semnalului
f1 = 5
f2 = 20
f3 = 75

# semnal compus din 3 cosinusuri cu amplitudini diferite
x = np.cos(2 * np.pi * f1 * n * T) + 2 * np.cos(2 * np.pi * f2 * n * T) + 0.5 * np.cos(2 * np.pi * f3 * n * T)

# construim matricea Fourier (relatia 1 din teorie)
mat_four = np.exp(-2j * np.pi * np.outer(k, n) / N)

# calculam transformata Fourier prin inmultire matrice-vector
X = mat_four @ x

# vectorul de frecvente (in Hz)
fk = k / N * fs

# luam doar jumatatea pozitiva din spectru (pentru semnal real)
mag = np.abs(X)          # modulul transformatei
kpos = np.arange(0, N//2 + 1)
fk_pos = fk[kpos]
mag_pos = mag[kpos]

# afisam rezultatele
plt.figure(figsize = (10,4))

# semnalul in timp
plt.subplot(1,2,1)
plt.plot(n/fs, x, color = "blue")
plt.xlabel('Timp (s)')
plt.ylabel('x(t)')
plt.grid(True)

# spectrul in frecventa (modulul transformatei)
plt.subplot(1,2,2)
plt.stem(fk_pos, mag_pos)
plt.xlabel('Frecventa (Hz)')
plt.ylabel('|X(ω)|')
plt.grid(True)

plt.suptitle('Transformata Fourier pentru un semnal cu 3 componente de frecventa')
plt.tight_layout()
plt.show()
