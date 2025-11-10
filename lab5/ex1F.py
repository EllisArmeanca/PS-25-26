#frecventele principale continute in semnal , in fft
# determinati primele 4 cele mai mari alori ale modulului transformatei si specificati carore frecvente in Hz le corespuind
# caror fenomente periodice din semnal se asociaza fiecare

# gasim cele mai mari 4 varfuri din fft in afara de 0
# afisam frecventele corespunzatoare

import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math
import time
import wavfile

import numpy as np

# citim fisier csv si salvam in variabila x
x = np.genfromtxt('Train.csv', delimiter=',', skip_header=1, usecols=2)

x = x - np.mean(x)

# transformata fourier a semnalului x
X = np.fft.fft(x)

# setam frecventele de baza
Fs = 1/3600
N = len(x)

# modulul transformatei
X = abs(X/ N )

#utilizam doar jumatate din spectru
X = X[:N//2]
#generam vectorul de frecvente pentru care e calculata trasnformata
f = Fs * np.linspace(0, N/2, N//2) / N

#gasim primele 4 varfuri
idx_sort = np.argsort(X)[::-1] # sortam de la cel mai mare la cel mai mic
idx_sort = idx_sort[f[idx_sort] > 0 ]
# f [idx_sort] obtine frecventa si X [idx_sort] obtina amplitutindea
top_k = 4

print("Primele 4 varfuri:")
for i in range(top_k):
    fr_hz = f[idx_sort[i]]
    period_hours = 1 / fr_hz / 3600
    period_days = period_hours / 24

    print(f"{i+1}: Frecventa = {fr_hz:.8f} Hz, Perioada = {period_hours:.2f} ore ({period_days:.2f} zile)")

# plot fft frecventa
plt.figure()
plt.plot(f,X )

plt.xlabel("Frecventa [Hz]")
plt.ylabel("X(w)") # transformata fourier a semanluli x(t)
plt.title('Modulul transformatei Fourier (jumatatea pozitiva)')
#pana la ~5 cicluri/zi
plt.xlim(0, 5/86400)
plt.tight_layout()
plt.savefig("ex1f.pdf", dpi=300)
plt.show()
#salvam figura
