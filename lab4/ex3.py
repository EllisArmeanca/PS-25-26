
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math
import time

import numpy as np

# parametrii semnalelor
f0 = 9.0      # frecventa originala
fe = 20.0     # frecventa de esantionare peste Nyquist (fe > 2*f0 = 18)
Te = 1.0 / fe # perioada de esantionare

# alte doua frecvente pentru comparatie (folosite si la ex 2)
f1 = 1.0
f2 = 19.0

# interval timp astfel incat sa avem ~8 esantioane
t_end = 7 * Te
t = np.linspace(0, t_end, 2000)        # timp continuu pt desenare
tn = np.arange(0, t_end + 1e-12, Te)   # momentele de esantionare

# semnale continue
x0 = np.sin(2*np.pi*f0*t)  # f0 = 9 Hz
x1 = np.sin(2*np.pi*f1*t)  # f1 = 1 Hz
x2 = np.sin(2*np.pi*f2*t)  # f2 = 19 Hz

# esantioane pentru fiecare semnal
xn0 = np.sin(2*np.pi*f0*tn)  # esantioane f0
xn1 = np.sin(2*np.pi*f1*tn)  # esantioane f1
xn2 = np.sin(2*np.pi*f2*tn)  # esantioane f2

# plot
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Ex 3 - Fara aliasing (fe = 20 Hz)")

# semnal continuu f0
axs[0].plot(t, x0)
axs[0].set_title("Semnal f0 = 9 Hz (continu)")
axs[0].set_ylabel("Amp")
axs[0].grid(True)

# f0 + esantioane
axs[1].plot(t, x0)
axs[1].scatter(tn, xn0, s=80, color="gold", edgecolors="k")
axs[1].set_title("f0 = 9 Hz cu esantioane (fe = 20 Hz)")
axs[1].set_ylabel("Amp")
axs[1].grid(True)

# f1 + esantioane
axs[2].plot(t, x1, color="purple")
axs[2].scatter(tn, xn1, s=80, color="gold", edgecolors="k")
axs[2].set_title("f1 = 1 Hz cu propriile esantioane")
axs[2].set_ylabel("Amp")
axs[2].grid(True)

# f2 + esantioane
axs[3].plot(t, x2, color="green")
axs[3].scatter(tn, xn2, s=80, color="gold", edgecolors="k")
axs[3].set_title("f2 = 19 Hz cu propriile esantioane")
axs[3].set_ylabel("Amp")
axs[3].set_xlabel("Timp [s]")
axs[3].grid(True)

# limite pe grafice
for ax in axs:
    ax.set_xlim(0, t_end)
    ax.set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig("ex3_no_aliasing.pdf", bbox_inches="tight")
plt.show()
