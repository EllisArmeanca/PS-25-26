import matplotlib
matplotlib.use('TkAgg')

#

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd


import numpy as np

fs = 44100
dur = 1.5
t = np.arange(int(dur * fs)) / fs

x_n = np.sin(2 * np.pi * 300 * t)

y_n = np.sin(2 * np.pi * 600 * t)

z_n = np.concatenate([x_n, y_n])
n = np.arange(len(z_n))

fig2, axs2 = plt.subplots(3, figsize=(10,6))
fig2.suptitle('Ex.5 — Sinusoide cu frecvențe diferite (fs = 44100 Hz)')

axs2[0].plot(t, x_n)
axs2[0].set_title('x[n] — 300 Hz')
axs2[0].set_xlabel('Timp [s]')
axs2[0].set_ylabel('Amplitudine')
axs2[0].grid(True)

axs2[1].plot(t, y_n)
axs2[1].set_title('y[n] — 600 Hz')
axs2[1].set_xlabel('Timp [s]')
axs2[1].set_ylabel('Amplitudine')
axs2[1].grid(True)

axs2[2].plot(n / fs, z_n)
axs2[2].set_title('z[n] — concatenare (300 Hz urmat de 600 Hz)')
axs2[2].set_xlabel('Timp [s]')
axs2[2].set_ylabel('Amplitudine')
axs2[2].grid(True)

plt.tight_layout()
plt.show()

rate = int(10e5)
scipy.io.wavfile.write('ex5_concat.wav', rate, z_n.astype(np.float32))
