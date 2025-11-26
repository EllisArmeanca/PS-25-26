# serii de timp python
import numpy as np
import scipy as sp
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def autocor(y, max_k):
    N = len(y)
    val = []
    for k in range(max_k + 1):
        suma = 0
        for t in range(k, N):
            suma += y[t] * y[t - k]
        val.append(suma / (N - k))
    return np.array(val)

# generarea unei serii de timp aleatoare de dimensiune N=1000
N = 1000
t = np.arange(N)

# trend - ecuatie de gradul 2
trend = 0.01 * t**2

# sezon - doua frecvente
frecventa1 = 0.05
frecventa2 = 0.1
sezon = 5 * np.sin(2 * np.pi * frecventa1 * t) + \
        3 * np.sin(2 * np.pi * frecventa2 * t + np.pi/4)

# zgomot alb gaussian
zgomot = np.random.normal(0, 2, N)

# seria finala
serie_timp = trend + sezon + zgomot

# calcul autocorelatie
v = autocor(serie_timp, 200)

# desen
plt.figure(figsize=(10, 5))
plt.plot(v)
plt.title('Autocorelatia seriei de timp')
plt.xlabel("Lag")
plt.ylabel("Autocorelatie")
plt.tight_layout()
plt.savefig("ex1b.pdf", dpi=300)
plt.show()

