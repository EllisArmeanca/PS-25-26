# serii de timp python
import numpy as np
import scipy as sp
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

# generarea unei serii de timp aleatoare de dimensiune N=1000
N = 1000
t = np.arange(N)

# trend - ecuatie de gradul 2
trend = 0.01 * t**2

# sezon - doua frecvente
frecventa1 = 0.05
frecventa2 = 0.1
sezon = 5 * np.sin(2 * np.pi * frecventa1 * t) + 3 * np.sin(2 * np.pi * frecventa2 * t + np.pi/4)

# zgomot alb gaussian
zgomot = np.random.normal(0, 2, N)

# seria finala
serie_timp = trend + sezon + zgomot

# grafice separate
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(serie_timp)
plt.title('Seria de timp completa')

plt.subplot(4, 1, 2)
plt.plot(trend)
plt.title('Trend')

plt.subplot(4, 1, 3)
plt.plot(sezon)
plt.title('Sezon')

plt.subplot(4, 1, 4)
plt.plot(zgomot)
plt.title('Zgomot alb')

plt.tight_layout()
plt.show()
plt.savefig("ex1a.pdf", dpi=300)