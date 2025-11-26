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
        val.append(suma / (N - k))     # important!
    return np.array(val)

def build_AR(y, p):
    Y = []
    T = []

    for i in range(p, len(y)):
        trecut = []
        for j in range(p):
            trecut.append(y[i - j - 1])
        Y.append(trecut)
        T.append(y[i])

    return np.array(Y), np.array(T)

def train_AR(y, p):
    Y, T = build_AR(y, p)
    w = np.linalg.inv(Y.T @ Y) @ (Y.T @ T)
    return w

def predict_AR(y, p, coef):
    predictii = []

    for i in range(p):
        predictii.append(y[i])

    for i in range(p, len(y)):
        suma = 0
        for j in range(p):
            suma += coef[j] * predictii[i - j - 1]

        predictii.append(suma)

    return np.array(predictii)


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

p=5

coef = train_AR(serie_timp, p)

pred = predict_AR(serie_timp, p, coef)

# desenam seria originala + predictia
plt.figure(figsize=(12, 5))
plt.plot(serie_timp, label="Original")
plt.plot(pred, label="Predictie AR(p)", alpha=0.8)
plt.title("Model AR(p) simplu, p = " + str(p))
plt.legend()
plt.tight_layout()
plt.savefig("ex1c.pdf", dpi=300)
plt.show()
