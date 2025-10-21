import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

#a
fs = 40000              # frecventa de esantionare
N = 1600               # numar de esantioane
t = np.arange(N) / fs  # vector de timp discret
x = np.sin(2 * np.pi * 400 * t)  # semnal sinusoidal

plt.plot(t, x)
plt.title('Semnal sinusoidal 400 Hz, 1600 esantioane')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.show()

#b
fs = 800              # frecventa de esantionare
N = 3 * fs # 3 secunde
t = np.arange(N) / fs
x = np.sin(2 * np.pi * 800 * t)
plt.plot(t, x)
plt.title('Semnal sinusoidal 800 Hz, 3 secunde')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.show()

#c
fs = 24000
#sawtooth - x(t) = 2(ft - floor(ft)) - 1
N = 0.05 * fs
t = np.arange(N) / fs
x = 2 * (240 * t - np.floor(240 * t)) - 1
plt.plot(t, x)
plt.title('Semnal sawtooth 240 Hz, 0.05 secunde')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.show()

#d
#square - oscileaza intre doua valori constante (1 si -1)
# x(T) = sign(sin(2πft))
fs = 5000
N = 0.05 * fs
t = np.arange(N) / fs
x = np.sign(np.sin(2 * np.pi * 300 * t))
plt.plot(t, x)
plt.title('Semnal square 300 Hz, 0.05 secunde')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.show()

#e
I = np.random.rand(128,128 )
plt.imshow(I, cmap='gray')
plt.colorbar()
plt.title('Semnal 2D aleator 128×128')
plt.show()

#f
# 128x128 imagine RGB (3 canale)
I = np.zeros((128, 128, 3))

# Partea de sus - rosu (R=1, G=0, B=0)
I[0:64, :, 0] = 1

# Partea de jos - albastru (R=0, G=0, B=1)
I[64:128, :, 2] = 1

plt.imshow(I)
plt.title('Steaua')
plt.axis('off')
plt.show()