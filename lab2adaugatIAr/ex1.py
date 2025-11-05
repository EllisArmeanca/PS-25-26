import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

def sinusoidal(A, f, phi, t):
    return A * np.sin(2 * np.pi * f * t + phi)

A = 4
f = 50
phi = np.pi / 3
t = np.linspace(0, 0.1, 1000)

x_sin = sinusoidal(A, f, phi, t)
x_cos = A * np.cos(2 * np.pi * f * t + phi - np.pi/2)


fig, axs = plt.subplots(2)
fig.suptitle('Titlu principal')

axs[0].plot(t, x_sin)
axs[0].set_title('Semnal sinusoidal sinus: x(t)')
axs[0].grid(True)

axs[1].plot(t, x_cos)
axs[1].set_title('Semnal sinusoidal cosinus: x(t)')
axs[1].grid(True)

plt.tight_layout()
plt.savefig("ex1.pdf", dpi=300)
plt.show()