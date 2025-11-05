
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math
import time

import numpy as np


def fft_recursive(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft_recursive(x[0::2]) # de la poz 0 din 2 in 2
    odd = fft_recursive(x[1::2]) # de la poz 1 din 2 in 2
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return np.array([even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)])


N1 = [128, 256, 512, 1024, 2048, 4096, 8192]

timp_dft_ms = []
timp_fft_ms = []
timp_fftmy_ms = []

for N in N1:
    x = np.random.rand(N)
    k = np.arange(N)  # indexul frecventei - fk = k/N fs numarul componentei de frecventa din X[k]
    n = np.arange(N)  # indexul timpului - pozitia fiecarui esantion din semnalul de intrare x[n]

    mat_four = np.exp(-2j * np.pi * np.outer(k, n) / N)

    # efectuam DFT
    start_dft = time.perf_counter()
    X = mat_four @ x
    end_dft = time.perf_counter()

    dft_time = end_dft - start_dft
    print(f'Timp DFT N={N}: {dft_time * 1000:.6f} ms')
    timp_dft_ms.append(dft_time * 1000)

    # efectuam FFT utilizand functia din numpy
    start_fft = time.perf_counter()
    X_fft = np.fft.fft(x)
    end_fft = time.perf_counter()

    fft_time = end_fft - start_fft
    print(f'Timp FFT N={N}: {fft_time * 1000:.6f} ms')
    timp_fft_ms.append(fft_time * 1000)

    # trebuie sa implementez FFT manual pentru a compara timpii de executie
    start_my = time.perf_counter()
    X_my = fft_recursive(x)
    end_my = time.perf_counter()

    my_time = end_my - start_my
    timp_fftmy_ms.append(my_time * 1000)
    print(f'Timp FFT recursiv N={N}: {my_time * 1000:.6f} ms')
    print('---')

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(N1, timp_dft_ms, marker='o', label='DFT')
plt.plot(N1, timp_fft_ms, marker='o', label='FFT (numpy)')
plt.plot(N1, timp_fftmy_ms, marker='o', label='FFT (recursiv)')
plt.xlabel('Numar de esantioane N')
plt.ylabel('Timp de executie (ms)')
plt.title('Comparatie timp de executie DFT vs FFT')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xticks(N1, N1)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
#vreau sa salvez acest plot ca pdf
plt.savefig('fft_dft_comparison.pdf')
plt.show()
