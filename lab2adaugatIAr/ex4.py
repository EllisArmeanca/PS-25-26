import matplotlib
matplotlib.use('TkAgg')

#

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd


import numpy as np


fs = 44100
dur = 1.0
t = np.arange(int(dur * fs)) / fs

x_n = np.sin(2 * np.pi * 440 * t)

y_n = 2 * (240 * t - np.floor(240 * t)) - 1

z_n = x_n + y_n

fig, axs2 = plt.subplots(3, figsize=(10,6))
fig.suptitle('Ex.4 — Semnale cu forme de undă diferite (fs = 44100 Hz)')

axs2[0].plot(t, x_n)
axs2[0].set_title('x[n] — sinus 440 Hz')
axs2[0].set_xlabel('Timp [s]')
axs2[0].set_ylabel('Amplitudine')
axs2[0].grid(True)

axs2[1].plot(t, y_n)
axs2[1].set_title('y[n] — sawtooth 240 Hz')
axs2[1].set_xlabel('Timp [s]')
axs2[1].set_ylabel('Amplitudine')
axs2[1].grid(True)

axs2[2].plot(t, z_n)
axs2[2].set_title('z[n] = x[n] + y[n]')
axs2[2].set_xlabel('Timp [s]')
axs2[2].set_ylabel('Amplitudine')
axs2[2].grid(True)

plt.tight_layout()
plt.show()

rate = int(10e5)
scipy.io.wavfile.write('ex4_sum.wav', rate, z_n.astype(np.float32))
