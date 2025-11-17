import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math
import time
import wavfile
import numpy as np

# 6(a)  Train.csv si  primele 72 valori (3 zile)
data = np.genfromtxt("Train.csv", delimiter=",", skip_header=1)
x_full = data[:, -1]    # presupunem ca ultima coloana contine valorile
x = x_full[:72]         # 3 zile = 72 ore

t = np.arange(len(x))

# lista ferestrelor cerute
windows = [5, 9, 13, 17]

plt.figure(figsize=(10, 10))

# subplot 1: semnal brut
plt.subplot(5, 1, 1)
plt.plot(t, x)
plt.title("Semnal brut - 3 zile (72 ore)")
plt.grid(True)

# subplot-urile pentru medii alunecatoare
for i, w in enumerate(windows):
    x_smooth = np.convolve(x, np.ones(w), "valid") / w
    plt.subplot(5, 1, i + 2)     # i+2 => subplots 2, 3, 4, 5
    plt.plot(x_smooth)
    plt.title(f"Media alunecatoare, w = {w}")
    plt.grid(True)

plt.tight_layout()
plt.savefig("ex6.pdf", dpi=300)
plt.show()

# 6(c) Frecventa de taiere
Fs = 1.0              # 1 esantion / ora
f_cut = 1 / 24        # frecventa zilnica
Wn = f_cut / (Fs / 2) # normalizare

print("Frecventa de taiere (Hz):", f_cut)
print("Frecventa normalizata Wn:", Wn)
