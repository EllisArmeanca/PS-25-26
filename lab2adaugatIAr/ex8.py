import matplotlib
matplotlib.use('TkAgg')

#

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd


import numpy as np

x = np.linspace(-np.pi/2, np.pi/2, 2000)
y_true = np.sin(x)

y_taylor1 = x
y_pade = (x - 7*(x**3)/60.0) / (1.0 + (x**2)/20.0)

err_taylor1 = np.abs(y_true - y_taylor1)
err_pade = np.abs(y_true - y_pade)

fig, axs = plt.subplots(3, figsize=(10,8))
fig.suptitle("Ex.8 — sin(x) vs aproximari (Taylor1 si Pade) pe [-pi/2, pi/2]")

axs[0].plot(x, y_true, label="sin(x)")
axs[0].plot(x, y_taylor1, "--", label="Taylor: x")
axs[0].plot(x, y_pade, "-.", label="Padé: (x - 7x^3/60)/(1 + x^2/20)")
axs[0].set_title("Functii")
axs[0].set_xlabel("x [rad]"); axs[0].set_ylabel("valoare"); axs[0].grid(True); axs[0].legend()

axs[1].semilogy(x, err_taylor1)
axs[1].set_title("Eroare abs. Taylor (sin(x) - x)")
axs[1].set_xlabel("x [rad]"); axs[1].set_ylabel("|eroare| (log)"); axs[1].grid(True, which="both")

axs[2].semilogy(x, err_pade)
axs[2].set_title("Eroare abs. Padé")
axs[2].set_xlabel("x [rad]"); axs[2].set_ylabel("|eroare| (log)"); axs[2].grid(True, which="both")

plt.tight_layout(); plt.show()
