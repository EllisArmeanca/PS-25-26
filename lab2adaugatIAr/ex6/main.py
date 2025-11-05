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

x_n = np.sin(2 * np.pi * (fs/2) * t)

y_n = np.sin(2 * np.pi * (fs/4) * t)

z_n = np.zeros_like(t)

fig2, axs2 = plt.subplots(3, figsize=(10,6))
fig2.suptitle('Ex.6 — Semnale sinusoidale cu f = fs/2, fs/4, 0 Hz')

axs2[0].stem(t[:200], x_n[:200])
axs2[0].set_title('x[n]: f = fs/2 (Nyquist)')
axs2[0].set_xlabel('Timp [s]')
axs2[0].set_ylabel('Amplitudine')
axs2[0].grid(True)

axs2[1].stem(t[:800], y_n[:800])
axs2[1].set_title('y[n]: f = fs/4')
axs2[1].set_xlabel('Timp [s]')
axs2[1].set_ylabel('Amplitudine')
axs2[1].grid(True)

axs2[2].stem(t[:800], z_n[:800])
axs2[2].set_title('z[n]: f = 0 Hz')
axs2[2].set_xlabel('Timp [s]')
axs2[2].set_ylabel('Amplitudine')
axs2[2].grid(True)

plt.tight_layout()
plt.show()

rate = int(10e5)
scipy.io.wavfile.write('ex6a_fs2.wav', rate, x_n.astype(np.float32))
scipy.io.wavfile.write('ex6b_fs4.wav', rate, y_n.astype(np.float32))
scipy.io.wavfile.write('ex6c_0Hz.wav', rate, z_n.astype(np.float32))