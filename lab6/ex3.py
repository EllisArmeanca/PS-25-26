import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math
import time
import wavfile

import numpy as np



N = 5  # grad maxim (poti schimba daca vrei)

#  coeficienti intregi  in  [-5, 5]
p = np.random.randint(-5, 6, size=N + 1)
q = np.random.randint(-5, 6, size=N + 1)

print("p(x) coef:", p)
print("q(x) coef:", q)

# 1) Inmultirea directa (convolutie in timp)
def poly_mult_direct(p, q):
    # dimensiunea rezultatului: len(p) + len(q) - 1
    r = np.zeros(len(p) + len(q) - 1, dtype=int)
    for i in range(len(p)):
        for j in range(len(q)):
            r[i + j] += p[i] * q[j]
    return r

r_direct = poly_mult_direct(p, q)
print("r(x) direct:", r_direct)

# 2) Inmultirea folosind FFT (produs in frecventa)
def poly_mult_fft(p, q):
    m = len(p) + len(q) - 1  # nr coeficienti rezultat
    size = 1
    while size < m:
        size *= 2

    # FFT pe coeficienti (completam cu zerouri pana la size)
    P = np.fft.fft(p, size)
    Q = np.fft.fft(q, size)

    # produs in frecventa
    R = P * Q

    # intoarcem in timp prin IFFT
    r = np.fft.ifft(R)

    # rotunjim la cel mai apropiat intreg
    r_real = np.rint(r.real).astype(int)

    # pastram doar primii m coeficienti
    return r_real[:m]

r_fft = poly_mult_fft(p, q)
print("r(x) FFT   :", r_fft)

# verificam daca cele doua rezultate coincid
print("Sunt egale direct vs FFT? ->", np.array_equal(r_direct, r_fft))
