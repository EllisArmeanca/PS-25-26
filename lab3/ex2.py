import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math

import numpy as np

#definirea semnalului x n
f0 = 3 # frecventa semnalului
fs = 1000 # frecventa de esantionare
T = 1 / fs # periaoda de esantionare, 1000 easntioane

n = np.arange(0, 1, T) #1000 esantionae, 1 secunda de semnal, timp discret
x = np.sin(2*np.pi*f0*n) # semnal sinusoidal cu 3 oscilatii complete pe 1 secunda

#plotam figura 1 stanga

fig, ax = plt.subplots()
fig.suptitle("Plotez x(n)")
ax.grid(True)
ax.plot(n, x)
ax.set_xlabel("Timp (S)")
ax.set_ylabel("Amplitutinde")
plt.show()

y = x * np.exp(-2j * np.pi * n) # consruiesc infasurarea semnalului
plt.figure()
plt.plot(np.real(y), np.imag(y))
plt.xlabel("Real")
plt.ylabel("Imaginar")
plt.title("Reprezentarea in planul complex")
plt.grid(True)
plt.axis("equal") # plotul nu e deformat
plt.show()

#am terminat figura 1

#incep figura 2

w = [1, 2, 5, 7]

fig2, axs2 = plt.subplots(4, figsize=(6, 10))
plt.subplots_adjust(hspace=0.6)


for index,w1 in enumerate(w):
    z = x * np.exp(-2j * np.pi * w1 * n)
    dist = np.abs(z)
    axs2[index].scatter(np.real(z), np.imag(z), c=dist, cmap='viridis', s=2)
    # scatter deseneaza punctele (real, imaginar) si le coloreaza in functie de distanta fata de origine (c=dist), folosind paleta 'viridis' si dimensiunea punctelor s=2
        # punctele mai aproape de centru sunt colorate diferit fata de cele mai departe - c = dist
    # s=2 puncte foarte mici ca sa nu se suprapuna
    axs2[index].grid(True)
    axs2[index].set_aspect("equal")
    axs2[index].set_title(f"w = {w1} ")
    fig2.colorbar(axs2[index].collections[0], ax=axs2[index])
    # colorbar adauga o bara laterala care arata legenda culorilor folosite in scatter (valorile amplitudinilor)

plt.tight_layout()
plt.show()