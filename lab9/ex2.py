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

def mediere_exponentiala(x, alpha): #termenii mai recenti primesc pondere mai mare iar termenii vechi primesc pondere exponential mai mica
    s = np.zeros_like(x)
    s[0] = x[0]
    for t in range(1, len(x)):
        s[t] = alpha * x[t] + (1 - alpha) * s[t - 1]
    return s

alpha = 0.3
s_alpha = mediere_exponentiala(serie_timp, alpha)

#alpha intre 0 si 1
lista_alpha = np.linspace(0.01, 0.99, 50)
best_alpha = None
best_err = 1e18


for a in lista_alpha:
    s_temp = mediere_exponentiala(serie_timp, a)
    err = np.sum((s_temp - serie_timp)**2)
    if err < best_err:
        best_err = err
        best_alpha = a

s_best = mediere_exponentiala(serie_timp, best_alpha)


print("Alpha optim =", best_alpha)

# plot
plt.figure(figsize=(12,6))
plt.plot(serie_timp, label="Original")
plt.plot(s_alpha, label=f"Exp smoothing α={alpha}")
plt.plot(s_best, label=f"Exp smoothing α optim={best_alpha:.2f}")
plt.legend()
plt.title("Mediere Exponentiala Simpla")
plt.savefig("ex2_simpla.pdf", dpi=300)
plt.show()