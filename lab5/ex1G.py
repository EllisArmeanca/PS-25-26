# datasetul incepe cu 25 aug 2012 - sambata
# o zi are 24 de esantionae
# start = 24* 2 = 48

import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math
import time
import wavfile

import numpy as np

x = np.genfromtxt("Train.csv", delimiter=',', skip_header=1, usecols=2)

Fs = 1/3600

esantioane_per_zi = 24
start = 48 + 24 * 7 * 6
Luna = 30 * 24

segment = x[start : start + Luna]

#vector timp in zile
t = np.arange(Luna) / esantioane_per_zi

plt.figure()
plt.plot(t, segment)
plt.xlabel("Timp [zile]")
plt.ylabel("nr / ora")
plt.title("Trafic o luna")
plt.xticks(np.arange(0, 31, 1))  # afiseaza de la 0 la 30, din 1 in 1 zi
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("ex1G.pdf", dpi=300)
plt.show()
