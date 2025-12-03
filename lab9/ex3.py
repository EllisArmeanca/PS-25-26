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
trend = 0.00000001 * t**2

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

def build_MA(y, q):
    mu = np.mean(y)
    eps = y - mu

    Y = []
    T = []

    for i in range(q, len(y)):
        trecut = []
        for j in range(q):
            trecut.append(eps[i - j - 1])
        Y.append(trecut)

        T.append(y[i] - mu)

    return np.array(Y), np.array(T), mu, eps

def train_MA(y, q):
    Y, T, mu, eps = build_MA(y, q)
    theta = np.linalg.inv(Y.T @ Y) @ (Y.T @ T)
    return theta, mu, eps

def predict_MA(y, q, theta, mu, eps):
    predictii = []

    # primele q valori nu pot fi prezise
    for i in range(q):
        predictii.append(y[i])

    for i in range(q, len(y)):
        suma = mu
        for j in range(q):
            suma += theta[j] * eps[i - j - 1]
        predictii.append(suma)

    return np.array(predictii)


q = 3
theta, mu, eps = train_MA(serie_timp, q)
predictii = predict_MA(serie_timp, q, theta, mu, eps)

plt.plot(serie_timp, label="Original")
plt.plot(predictii, label="MA manual")
plt.legend()
plt.savefig("ex3.pdf", dpi=300)
plt.show()
