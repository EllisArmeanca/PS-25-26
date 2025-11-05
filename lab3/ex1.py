import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math

import numpy as np

N=8
k = np.arange(N)
n = np.arange(N)

mat_four = np.exp(-2j * np.pi * np.outer(k, n) / N)

# fig, axs = plt.subplots(8, figsize=(8, 10))
# fig.suptitle('REAL(F)')
#
# for i in range(N):
#     axs[i].set_title(f'k={i}')
#     axs[i].set_xlabel('n')
#     axs[i].set_ylabel('Re(F[k])')
#     axs[i].grid(True)
#     axs[i].plot(n, np.real(mat_four[i, :]))
#
# plt.tight_layout()
# plt.show()
#
# fig2, axs2 = plt.subplots(8, figsize=(8, 10))
# fig2.suptitle('IMAG(F)')
#
# for i in range(N):
#     axs2[i].set_title(f'k={i}')
#     axs2[i].set_xlabel('n')
#     axs2[i].set_ylabel('Imag(F[k])')
#     axs2[i].grid(True)
#     axs2[i].plot(n, np.imag(mat_four[i, :]))
#
# plt.tight_layout()
# plt.show()

fig, axs = plt.subplots(8, figsize=(8, 10))
fig.suptitle('REAL vs IMAG (F)')

for i in range(N):
    axs[i].set_title(f'k={i}')
    axs[i].set_xlabel('n')
    axs[i].set_ylabel('F[k]')
    axs[i].grid(True)

    # Partea reala - albastru
    axs[i].plot(n, np.real(mat_four[i, :]), label='Re(F[k])', color='blue')

    # Partea imaginara - rosu
    axs[i].plot(n, np.imag(mat_four[i, :]), label='Im(F[k])', color='red', linestyle='--')

    axs[i].legend(loc='upper right')

plt.tight_layout()
plt.show()

F_H = mat_four.conj().T #hermitiana si transpusa si conjugata
# transpusa conjugata a matricei dft
G =  F_H @  mat_four

v = np.allclose(G, N * np.eye(N))

print(v)

#pe acelasi subplot cu alta culoare sa l pubn pe celalat