import matplotlib
matplotlib.use('TkAgg')

#

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd


import numpy as np
fs = 1000
dur = 0.5
t = np.arange(int(fs * dur)) / fs

f0 = 50
x = np.sin(2 * np.pi * f0 * t)

y_a = x[::4]
t_a = t[::4]

z_b = x[1::4]
t_b = t[1::4]

fig1, axs1 = plt.subplots(3, figsize=(10,6))
fig1.suptitle("Ex.7(a) — Decimare la 1/4 incepand de la index 0")

axs1[0].stem(t, x, basefmt=" ")
axs1[0].set_title("x[n] — semnal original (fs = 1000 Hz)")
axs1[0].set_xlabel("Timp [s]")
axs1[0].set_ylabel("Amplitudine")
axs1[0].grid(True)

axs1[1].stem(t_a, y_a, basefmt=" ")
axs1[1].set_title("y[n] — semnal decimat (fs' = 250 Hz)")
axs1[1].set_xlabel("Timp [s]")
axs1[1].set_ylabel("Amplitudine")
axs1[1].grid(True)

axs1[2].plot(t, x, label="x[n] original")
axs1[2].plot(t_a, y_a, "o-", markersize=3, label="y[n] decimat")
axs1[2].set_title("Comparatie: original vs decimat")
axs1[2].set_xlabel("Timp [s]")
axs1[2].set_ylabel("Amplitudine")
axs1[2].grid(True)
axs1[2].legend()

plt.tight_layout()
plt.show()

fig2, axs2 = plt.subplots(3, figsize=(10,6))
fig2.suptitle("Ex.7(b) — Decimare la 1/4 pornind de la al doilea element (index 1)")

axs2[0].stem(t, x, basefmt=" ")
axs2[0].set_title("x[n] — semnal original (fs = 1000 Hz)")
axs2[0].set_xlabel("Timp [s]")
axs2[0].set_ylabel("Amplitudine")
axs2[0].grid(True)

axs2[1].stem(t_b, z_b, basefmt=" ")
axs2[1].set_title("z[n] — decimat cu offset de faza")
axs2[1].set_xlabel("Timp [s]")
axs2[1].set_ylabel("Amplitudine")
axs2[1].grid(True)

axs2[2].plot(t_a, y_a, "o-", markersize=3, label="y[n] (start index 0)")
axs2[2].plot(t_b, z_b, "o-", markersize=3, label="z[n] (start index 1)")
axs2[2].set_title("Comparatie: y[n] vs z[n] (aceeasi frecventa, faze diferite)")
axs2[2].set_xlabel("Timp [s]")
axs2[2].set_ylabel("Amplitudine")
axs2[2].grid(True)
axs2[2].legend()

plt.tight_layout()
plt.show()

