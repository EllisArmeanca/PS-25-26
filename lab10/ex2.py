import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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

def predict_AR_onedata(y, p, w):

    pred = np.zeros_like(y, dtype=float)
    pred[:p] = y[:p]
    for i in range(p, len(y)):
        x = np.array([y[i - j - 1] for j in range(p)])
        pred[i] = w @ x
    return pred

N = 1000
t = np.arange(N)
trend = 0.01 * t**2
frecventa1 = 0.05
frecventa2 = 0.1
sezon = 5 * np.sin(2 * np.pi * frecventa1 * t) + 3 * np.sin(2 * np.pi * frecventa2 * t + np.pi/4)
zgomot = np.random.normal(0, 2, N)
serie_timp = trend + sezon + zgomot

p = 5
w = train_AR(serie_timp, p)
pred = predict_AR_onedata(serie_timp, p, w)

mse = np.mean((serie_timp[p:] - pred[p:])**2)
print("Coef AR:", w)
print("MSE (1-step):", mse)

plt.figure(figsize=(12,5))
plt.plot(serie_timp, label="Original")
plt.plot(pred, label="Predictie AR(p) 1-step", alpha=0.8)
plt.title(f"Ex2: AR(p) cu LS, p={p}")
plt.legend()
plt.tight_layout()
plt.savefig("ex2.pdf", dpi=300)
plt.show()
