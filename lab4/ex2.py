
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math
import time

import numpy as np

#Parametrii semnalului
f0 = 9 # frecventa semnalului
fe = 10 # sub Nyquist, fe < 2f0
Te = 1.0 / fe  # perioada de esantionare

#aliasuri la fe=9 :1 si 19 hz

f1 = 1.0
f2 = 19.0

#doar primele 0.03 secunde
t_end = 7*Te
t = np.linspace(0, t_end, 2000)

tn = np.arange(0.0, t_end + 1e-12, Te)  # 0, Te, 2Te, ... <= 0.03

#semnalele continue:
x0 = np.sin(2 * np.pi * f0 * t)
x1 = np.sin(2 * np.pi * f1 * t)
x2 = np.sin(2 * np.pi * f2 * t)

#semnal original esantioane
xn = np.sin(2 * np.pi * f0 * tn)

#plotting:
fig, axs = plt.subplots(4, figsize=(10, 8))
fig.suptitle("Ex.2 — Efectul de aliasing")

#1 9hz continuu
axs[0].plot(t, x0)
axs[0].set_ylabel("Amp.")
axs[0].grid(True)
axs[0].set_title("Semnal continuu — f0 = 9 Hz")

#2 9hz + esantion
axs[1].plot(t,x0)
axs[1].scatter(tn, xn, s=80, color="gold", edgecolors="k", zorder=3)
axs[1].set_ylabel("Amp.")
axs[1].grid(True)
axs[1].set_title("esantioane fe = 10 hz pe f0 = 9hz")

#1 hz si aceleasi esantioane
axs[2].plot(t, x1, color="purple")
axs[2].scatter(tn, xn, s=80, color="gold", edgecolors="k", zorder=3)
axs[2].set_ylabel("Amp.")
axs[2].grid(True)
axs[2].set_title("f1 = 1 Hz si aceleasi esantioane")

#19 hz si aceleasi esantioane
axs[3].plot(t, x2, color="green")
axs[3].scatter(tn, xn, s=80, color="gold", edgecolors="k", zorder=3)
axs[3].set_xlabel("Timp [s]")
axs[3].set_ylabel("Amp.")
axs[3].grid(True)
axs[3].set_title("f2 = 19 Hz si aceleasi esantioane")


for ax in axs:
    ax.set_xlim(0, t_end)
    ax.set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig("ex2_aliasing.pdf", bbox_inches="tight")
plt.show()














#
# import matplotlib
#
# matplotlib.use('TkAgg')
#
# import matplotlib.pyplot as plt
# import scipy
# import sounddevice as sd
# import math
# import time
#
# import numpy as np
#
# #trebuie sa construiesc un semnal sinusoidal
#
# t = np.linspace(0, 0.03, 500)
# x = np.sin(2*np.pi * 9 * t)
#
# fe = 10 # frecventa de esantionare sub Nquiest fe < 2f0
# Te = 1/fe
#
# tn = np.arange(0, 1, Te)
# X = np.sin(2 * np.pi * 9 * tn)
#
# f1 = 1
# f2 = 19
#
# #desenez plotul cu cele 4 grafice
# fig, axs = plt.subplots(4, figsize=(10, 8))
# fig.suptitle("Ex.2 — Efectul de aliasing")
# axs[0].plot(t, x)
# axs[0].set_title("Semnal original (f0 = 9 Hz)")
# axs[0].set_xlabel("Timp [s]")
# axs[0].set_ylabel("Amplitudine")
# axs[0].grid(True)
#
# axs[1].stem(tn, X, basefmt=" ")
# axs[1].set_title("Semnal esantionat (fe = 10 Hz < 2f0)")
# axs[1].set_xlabel("Timp [s]")
# axs[1].set_ylabel("Amplitudine")
# axs[1].grid(True)
#
# #semnalul aliased
# x_alias = np.sin(2 * np.pi * f1 * t)
# axs[2].plot(t, x_alias, 'r')
# axs[2].set_title(f"Semnal aliased (f_alias = |{f1} - {f2}| = {f_alias} Hz)")
# axs[2].set_xlabel("Timp [s]")
# axs[2].set_ylabel("Amplitudine")
# axs[2].grid(True)
#
# x_alias = np.sin(2 * np.pi * f2 * t)
# axs[3].stem(tn, X, basefmt=" ")
# axs[3].plot(t, x_alias, 'r', alpha=0.5)
# axs[3].set_title("Comparatie: semnal esantionat vs semnal aliased")
# axs[3].set_xlabel("Timp [s]")
# axs[3].set_ylabel("Amplitudine")
# axs[3].grid(True)
# plt.tight_layout()
# #salvez plotul intr un pdf
# fig.savefig("ex2_aliasing.pdf")
# plt.show()

