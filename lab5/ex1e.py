# semnalul prezinta o componenta continua deoarece in grafic apare o valoare mult ridicata a modulului pentru frecventa 0
# modificarea se va face asa : x = x - np.mean(x)

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

#MODIFICAREA
# x = x - np.mean(x)

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

# plot fft frecventa
plt.figure()
plt.plot(f,X )

plt.xlabel("Frecventa [Hz]")
plt.ylabel("X(w)") # transformata fourier a semanluli x(t)
plt.title('Modulul transformatei Fourier (jumatatea pozitiva)')
#un zoom apropiat de 0:
plt.xlim(0, 1e-6)
plt.tight_layout()
plt.savefig("ex1eNemodificata.pdf", dpi=300)
plt.show()
#salvam figura
