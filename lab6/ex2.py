import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math
import time
import wavfile

import numpy as np


N = 100

x0 = np.random.random(N)

#  convolutii
x1 = np.convolve(x0, x0)      # prima convolutie
x2 = np.convolve(x1, x1)      # a doua convolutie
x3 = np.convolve(x2, x2)      # a treia convolutie

plt.figure(figsize=(8, 10))

# x initial
plt.subplot(4, 1, 1)
plt.plot(x0)
plt.title("Semnal aleator initial")
plt.grid(True)

# dupa prima convolutie
plt.subplot(4, 1, 2)
plt.plot(x1)
plt.title("Dupa 1 convolutie: x * x")
plt.grid(True)

# dupa a doua convolutie
plt.subplot(4, 1, 3)
plt.plot(x2)
plt.title("Dupa 2 convolutii: (x * x) * (x * x)")
plt.grid(True)

# dupa a treia convolutie
plt.subplot(4, 1, 4)
plt.plot(x3)
plt.title("Dupa 3 convolutii")
plt.grid(True)

plt.tight_layout()
plt.savefig("ex2A.pdf", dpi=300)
plt.show()


# semnal bloc: 0 peste tot, 1 pe un interval scurt
x0_rect = np.zeros(N)
x0_rect[40:60] = 1.0   # bloc de 1 intre index 40 si 59

x1_rect = np.convolve(x0_rect, x0_rect)
x2_rect = np.convolve(x1_rect, x1_rect)
x3_rect = np.convolve(x2_rect, x2_rect)

plt.figure(figsize=(8, 10))

plt.subplot(4, 1, 1)
plt.plot(x0_rect)
plt.title("Semnal bloc rectangular initial")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(x1_rect)
plt.title("Bloc dupa 1 convolutie")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(x2_rect)
plt.title("Bloc dupa 2 convolutii")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(x3_rect)
plt.title("Bloc dupa 3 convolutii")
plt.grid(True)

plt.tight_layout()
plt.savefig("ex2B.pdf", dpi=300)
plt.show()
