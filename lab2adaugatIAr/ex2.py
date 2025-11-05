import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

def sinusoidal(A, f, phi, t):
    return A * np.sin(2 * np.pi * f * t + phi)

A = 1
f = 5
phi = [np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

t= np.linspace(0, 1, 1000)

for p in phi:
    x_t = sinusoidal(A, f, p, t)
    plt.plot(t, x_t, label=f'phi={p:.2f} rad')

plt.legend()
plt.grid(True)
plt.title('Semnale sinusoidale cu faze diferite')
plt.show()
plt.savefig("ex2.pdf", dpi=300)

x = sinusoidal(A, f, np.pi/2, t)
z = np.random.normal(0,1,len(x))

e_x = np.linalg.norm(x)**2
e_z = np.linalg.norm(z)**2

snr_t = {0.1, 1, 10 ,100}
for s in snr_t:
    k = np.sqrt(e_x / (s * e_z))
    y = x + k * z
    plt.plot(t, y, label=f'SNR={s} ')
    plt.title(f'Semnal cu zgomot, SNR={s}')
    plt.xlabel('Timp [s]')
    plt.ylabel('Amplitudine')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig(f"ex2_snr_{s}.pdf", dpi=300)

#o auzi mai mult, pune-o de o lungime mai mare