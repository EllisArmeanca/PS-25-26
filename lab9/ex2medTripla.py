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
plt.savefig("ex1a.pdf", dpi=300)
plt.show()

def mediere_tripla(x, alpha, beta, gamma, L):
    N = len(x)
    s = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)

    s[0] = x[0]
    b[0] = x[1] - x[0]

    # initializare sezonalitate pe primii L termeni
    for i in range(L):
        c[i] = x[i] - s[0]

    for t in range(1, N):
        s[t] = alpha * (x[t] - c[t - L]) + (1 - alpha) * (s[t - 1] + b[t - 1])
        b[t] = beta * (s[t] - s[t - 1]) + (1 - beta) * b[t - 1]
        c[t] = gamma * (x[t] - s[t] - b[t - 1]) + (1 - gamma) * c[t - L]

    return s


alpha = 0.3
beta = 0.2
gamma = 0.1
L = 50

s_tripla = mediere_tripla(serie_timp, alpha, beta, gamma, L)

plt.figure(figsize=(12, 6))
plt.plot(serie_timp, label="Original")
plt.plot(s_tripla, label="Mediere Exponentiala Tripla")
plt.legend()
plt.title("Holt-Winters Aditiv (mediere exponentiala tripla)")
plt.savefig("ex2_tripla.pdf", dpi=300)
plt.show()